"""SSD 磁盘存储层。

负责把 ComfyUI 节点的输出（张量 / 容器 / 标量）安全地序列化到磁盘，
并提供原子写入、目录分片、容量统计与清理能力。

设计要点
--------
* 只接受 ``torch.Tensor`` 与基本容器，遇到 MODEL / CLIP 这类业务对象直接跳过，
  绝不会把整个模型权重拖进缓存。
* 写入先落到同目录的临时文件，再用 ``os.replace`` 原子替换，进程中途崩掉
  也不会留下可被误读的半截文件。
* 读取使用 ``weights_only=True``，即使缓存目录被第三方投毒也无法执行任意代码。
* 目录按 hash 前 4 位分两级，避免单目录文件过多拖慢文件系统。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any, Optional, Tuple

import torch

log = logging.getLogger("SSD-Cache")

DATA_SUFFIX = ".pt"
META_SUFFIX = ".json"
TMP_PREFIX = ".tmp-"

#: 认为可以直接序列化的 Python 标量类型
_PRIMITIVES = (str, int, float, bool, bytes, type(None))


class UnsupportedValue(Exception):
    """节点输出里含有无法（或不应）持久化的对象。"""


def to_saveable(value: Any) -> Any:
    """递归把节点输出转换成 ``torch.save`` 可安全持久化的结构。

    * 张量 -> detach 到 CPU、连续化
    * dict    -> 保留键（仅允许标量键）
    * list/tuple -> 递归，保留容器类型
    * 其它类型 -> 抛 :class:`UnsupportedValue`
    """
    if torch.is_tensor(value):
        if value.is_sparse:
            raise UnsupportedValue("稀疏张量")
        if value.device.type == "meta":
            raise UnsupportedValue("meta 张量")
        if value.is_quantized:
            raise UnsupportedValue("量化张量")
        tensor = value.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor

    if isinstance(value, _PRIMITIVES):
        return value

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, _PRIMITIVES):
                raise UnsupportedValue(f"字典键类型 {type(key).__name__}")
            result[key] = to_saveable(item)
        return result

    if isinstance(value, (list, tuple)):
        items = [to_saveable(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items

    raise UnsupportedValue(f"不支持的类型 {type(value).__name__}")


def estimate_size(value: Any) -> int:
    """估算一个输出结构里张量占用的字节数（用于单条上限判断）。"""
    total = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if torch.is_tensor(item):
            try:
                total += item.numel() * item.element_size()
            except Exception:  # noqa: BLE001
                pass
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return total


def format_bytes(num: int) -> str:
    """人类可读的字节数。"""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


class SSDCacheStore:
    """一个缓存目录的读写封装。线程安全，可跨事件循环在线程池里调用。"""

    def __init__(self, cache_dir: str) -> None:
        self._dir: str = ""
        self._lock = threading.RLock()
        self._size_cache: Optional[Tuple[int, int]] = None
        self._size_cache_time: float = 0.0
        self.set_dir(cache_dir)

    # ------------------------------------------------------------------ 目录
    @property
    def cache_dir(self) -> str:
        return self._dir

    def set_dir(self, cache_dir: str) -> None:
        """切换缓存目录；目录不存在时自动创建。"""
        path = os.path.abspath(os.path.expanduser(str(cache_dir)))
        with self._lock:
            if path == self._dir:
                return
            os.makedirs(path, exist_ok=True)
            self._dir = path
            self._size_cache = None
            self._size_cache_time = 0.0

    def _entry_path(self, key: str) -> str:
        return os.path.join(self._dir, key[:2], key[2:4], key + DATA_SUFFIX)

    def _meta_path(self, key: str) -> str:
        return os.path.join(self._dir, key[:2], key[2:4], key + META_SUFFIX)

    # ------------------------------------------------------------------ 读写
    def has(self, key: str) -> bool:
        return os.path.isfile(self._entry_path(key))

    def save(
        self,
        key: str,
        outputs: Any,
        *,
        class_type: str = "",
        node_id: str = "",
    ) -> int:
        """保存一条缓存，返回写入的字节数。不可序列化时抛 UnsupportedValue。"""
        payload = to_saveable(outputs)
        size = estimate_size(payload)

        path = self._entry_path(key)
        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(prefix=TMP_PREFIX, suffix=DATA_SUFFIX, dir=parent)
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        meta = {
            "key": key,
            "class_type": class_type,
            "node_id": node_id,
            "size": size,
            "created": time.time(),
            "plugin": "Comfyui-SSD-Cache",
            "version": 1,
        }
        try:
            with open(self._meta_path(key), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False)
        except OSError as exc:
            log.debug("SSD-Cache: 写入元数据失败 %s: %s", key, exc)

        with self._lock:
            self._size_cache = None
        return size

    def load(self, key: str) -> Optional[Any]:
        """读取一条缓存；不存在或损坏时返回 None（损坏文件会被删除）。"""
        path = self._entry_path(key)
        if not os.path.isfile(path):
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=True, mmap=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("SSD-Cache: 删除损坏的缓存条目 %s: %s", key, exc)
            try:
                os.unlink(path)
            except OSError:
                pass
            return None

    # ------------------------------------------------------------------ 统计
    def disk_usage(self, max_age: float = 5.0) -> Tuple[int, int]:
        """返回 ``(总字节数, 条目数)``，结果缓存 ``max_age`` 秒。"""
        with self._lock:
            now = time.time()
            if self._size_cache is not None and now - self._size_cache_time < max_age:
                return self._size_cache

        total = 0
        count = 0
        for root, _dirs, files in os.walk(self._dir):
            for name in files:
                if not name.endswith(DATA_SUFFIX) or name.startswith(TMP_PREFIX):
                    continue
                try:
                    total += os.path.getsize(os.path.join(root, name))
                    count += 1
                except OSError:
                    pass

        with self._lock:
            self._size_cache = (total, count)
            self._size_cache_time = time.time()
        return total, count

    def invalidate_size_cache(self) -> None:
        with self._lock:
            self._size_cache = None

    # ------------------------------------------------------------------ 清理
    def clear(self) -> int:
        """删除缓存目录下的所有条目与临时文件，返回删除的文件数。"""
        removed = 0
        with self._lock:
            for root, _dirs, files in os.walk(self._dir):
                for name in files:
                    if not (
                        name.endswith(DATA_SUFFIX)
                        or name.endswith(META_SUFFIX)
                        or name.startswith(TMP_PREFIX)
                    ):
                        continue
                    try:
                        os.unlink(os.path.join(root, name))
                        removed += 1
                    except OSError:
                        pass
            self._size_cache = None
            self._size_cache_time = 0.0
        return removed

    def purge_temp_files(self) -> int:
        """清理上一次崩溃留下的临时文件。"""
        removed = 0
        with self._lock:
            for root, _dirs, files in os.walk(self._dir):
                for name in files:
                    if name.startswith(TMP_PREFIX):
                        try:
                            os.unlink(os.path.join(root, name))
                            removed += 1
                        except OSError:
                            pass
        return removed
