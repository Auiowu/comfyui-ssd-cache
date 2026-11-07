"""把 ComfyUI 节点输出持久化到 SSD 的 CacheProvider。

ComfyUI 0.44 起内置了外部缓存提供者机制（``comfy_execution.cache_provider``）：
每次某个节点在内存缓存里未命中时，会依次询问已注册的 provider；节点执行完
写入内存缓存后，也会异步通知 provider 落盘。我们利用这个官方扩展点，把节点
输出镜像到 SSD，从而让"内存不够大"和"跑一半崩掉"都不必重算已完成的部分。

缓存键是 ComfyUI 自己算出的 ``cache_key_hash``（节点及其全部上游输入的确定性
SHA256），因此跨进程、跨重启都保持一致。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from comfy_execution.cache_provider import (
    CacheContext,
    CacheProvider,
    CacheValue,
    register_cache_provider,
)

from .config import DEFAULT_CACHE_DIR, SSDCacheConfig, load_config, save_config
from .store import SSDCacheStore, UnsupportedValue, estimate_size, format_bytes

log = logging.getLogger("SSD-Cache")

#: 读 / 写线程池并发度。写盘独立于 ComfyUI 的 prompt 事件循环 —— 后者每个
#: prompt 都会 asyncio.run() 一次并在结束时关闭，放在循环里的 create_task
#: 会被直接取消，导致缓存永远落不了盘。
_MAX_WRITE_WORKERS = 2
_MAX_READ_WORKERS = 3


@dataclass
class CacheStats:
    """一次运行（或进程生命周期累计）的缓存统计。"""

    prompt_id: str = ""
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    stores: int = 0
    stores_pending: int = 0
    store_failed: int = 0
    skipped_unsupported: int = 0
    skipped_too_large: int = 0
    bytes_written: int = 0
    started: float = field(default_factory=time.time)

    def reset(self, prompt_id: str = "") -> None:
        self.prompt_id = prompt_id
        self.lookups = 0
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.stores_pending = 0
        self.store_failed = 0
        self.skipped_unsupported = 0
        self.skipped_too_large = 0
        self.bytes_written = 0
        self.started = time.time()

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.lookups) if self.lookups else 0.0

    @property
    def elapsed(self) -> float:
        return max(0.0, time.time() - self.started)


class SSDCacheProvider(CacheProvider):
    """SSD 缓存提供者（全局单例）。"""

    def __init__(self, config: Optional[SSDCacheConfig] = None) -> None:
        self.config: SSDCacheConfig = config or load_config()
        self.store = SSDCacheStore(self.config.cache_dir)
        self.enabled: bool = self.config.enabled
        self.max_entry_bytes: int = max(0, int(self.config.max_entry_mb)) * 1024 * 1024

        self.session = CacheStats()
        self.total = CacheStats()

        self._output_node_cache: Dict[str, bool] = {}
        self._stats_lock = threading.Lock()
        self._futures_lock = threading.Lock()
        self._pending_futures: set = set()
        self._write_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._read_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None

    # -------------------------------------------------------------- 配置
    def configure(self, cache_dir: str, enabled: bool, max_entry_mb: int) -> bool:
        """由 Enable 节点调用：切换目录 / 开关 / 单条上限，并持久化。"""
        cache_dir = os.path.abspath(os.path.expanduser(str(cache_dir or DEFAULT_CACHE_DIR)))
        try:
            self.store.set_dir(cache_dir)
        except OSError as exc:
            log.error("SSD-Cache: 无法创建缓存目录 %s: %s", cache_dir, exc)
            self.enabled = False
            return False

        self.enabled = bool(enabled)
        self.max_entry_bytes = max(0, int(max_entry_mb)) * 1024 * 1024
        self.config.cache_dir = cache_dir
        self.config.enabled = bool(enabled)
        self.config.max_entry_mb = max(0, int(max_entry_mb))
        save_config(self.config)
        log.info(
            "SSD-Cache: %s，目录=%s，单条上限=%s",
            "已启用" if enabled else "已禁用",
            cache_dir,
            "不限" if self.max_entry_bytes == 0 else format_bytes(self.max_entry_bytes),
        )
        return True

    # -------------------------------------------------------------- 线程池
    def _write_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._write_pool is None:
            self._write_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=_MAX_WRITE_WORKERS, thread_name_prefix="ssd-cache-write"
            )
        return self._write_pool

    def _read_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._read_pool is None:
            self._read_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=_MAX_READ_WORKERS, thread_name_prefix="ssd-cache-read"
            )
        return self._read_pool

    def _forget_future(self, future) -> None:
        with self._futures_lock:
            self._pending_futures.discard(future)

    def flush(self, timeout: Optional[float] = None) -> bool:
        """等待所有排队中的写盘任务结束（测试 / 关停时使用）。"""
        with self._futures_lock:
            futures = list(self._pending_futures)
        if not futures:
            return True
        _done, not_done = concurrent.futures.wait(futures, timeout=timeout)
        return not not_done

    def shutdown(self) -> None:
        for pool in (self._write_pool, self._read_pool):
            if pool is not None:
                pool.shutdown(wait=True)
        self._write_pool = None
        self._read_pool = None

    def _finish_store(
        self,
        success: bool,
        size: int = 0,
        unsupported: bool = False,
        error: Optional[BaseException] = None,
    ) -> None:
        with self._stats_lock:
            self.session.stores_pending = max(0, self.session.stores_pending - 1)
            self.total.stores_pending = max(0, self.total.stores_pending - 1)
            if success:
                self.session.stores += 1
                self.total.stores += 1
                self.session.bytes_written += size
                self.total.bytes_written += size
            elif unsupported:
                self.session.skipped_unsupported += 1
                self.total.skipped_unsupported += 1
            else:
                self.session.store_failed += 1
                self.total.store_failed += 1

    # -------------------------------------------------------------- 过滤
    def _is_output_node(self, class_type: str) -> bool:
        """输出节点（SaveImage/PreviewImage 等）不缓存，否则恢复时会跳过它们的副作用。"""
        cached = self._output_node_cache.get(class_type)
        if cached is not None:
            return cached
        result = False
        try:
            import nodes as comfy_nodes

            node_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_type)
            if node_cls is not None:
                result = bool(getattr(node_cls, "OUTPUT_NODE", False))
        except Exception:  # noqa: BLE001
            result = False
        self._output_node_cache[class_type] = result
        return result

    def should_cache(self, context: CacheContext, value: Optional[CacheValue] = None) -> bool:
        if not self.enabled:
            return False
        if self._is_output_node(context.class_type):
            return False
        if value is not None:
            if not value.outputs:
                return False
            if self.max_entry_bytes > 0:
                try:
                    if estimate_size(value.outputs) > self.max_entry_bytes:
                        self.session.skipped_too_large += 1
                        self.total.skipped_too_large += 1
                        return False
                except Exception:  # noqa: BLE001
                    pass
            # 写盘是异步线程任务，可能晚于 prompt 结束；这里先记一笔"排队中"，
            # 让 Stats 节点在它执行时也能看到本轮真实写入量。
            with self._stats_lock:
                self.session.stores_pending += 1
                self.total.stores_pending += 1
        return True

    # -------------------------------------------------------------- 查询
    async def on_lookup(self, context: CacheContext) -> Optional[CacheValue]:
        if not self.enabled:
            return None

        with self._stats_lock:
            self.session.lookups += 1
            self.total.lookups += 1

        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            self._read_executor(), self.store.load, context.cache_key_hash
        )

        with self._stats_lock:
            if payload is None:
                self.session.misses += 1
                self.total.misses += 1
            else:
                self.session.hits += 1
                self.total.hits += 1

        if payload is None:
            return None

        log.debug("SSD-Cache: 命中 %s (%s)", context.class_type, context.cache_key_hash[:12])
        return CacheValue(outputs=list(payload), ui=None)

    # -------------------------------------------------------------- 写入
    async def on_store(self, context: CacheContext, value: CacheValue) -> None:
        """提交写盘后立即返回。

        关键：不能把写盘放在 prompt 的事件循环里（ComfyUI 每个 prompt 都
        ``asyncio.run()`` 一次，结束时循环关闭，``create_task`` 的任务会被取消），
        所以这里提交到进程级线程池。
        """
        if not self.enabled or value is None or value.outputs is None:
            return

        try:
            future = self._write_executor().submit(self._store_sync, context, value)
        except RuntimeError as exc:  # 线程池已关闭
            log.warning("SSD-Cache: 写线程池不可用: %s", exc)
            self._finish_store(False, error=exc)
            return

        with self._futures_lock:
            self._pending_futures.add(future)
        future.add_done_callback(self._forget_future)

    def _store_sync(self, context: CacheContext, value: CacheValue) -> None:
        """在写线程里执行真正的落盘（同步，不受 prompt 事件循环影响）。"""
        try:
            size = self.store.save(
                context.cache_key_hash,
                value.outputs,
                class_type=context.class_type,
                node_id=str(context.node_id),
            )
        except UnsupportedValue:
            self._finish_store(False, unsupported=True)
            log.debug("SSD-Cache: 跳过不可序列化输出 %s", context.class_type)
            return
        except Exception as exc:  # noqa: BLE001 - 缓存失败不能影响主流程
            self._finish_store(False, error=exc)
            log.warning("SSD-Cache: 写入 %s 失败: %s", context.class_type, exc)
            return

        self._finish_store(True, size=size)
        log.debug(
            "SSD-Cache: 写入 %s (%s, %s)",
            context.class_type,
            context.cache_key_hash[:12],
            format_bytes(size),
        )

    # -------------------------------------------------------------- 生命周期
    def on_prompt_start(self, prompt_id: str) -> None:
        self.session.reset(prompt_id)
        # 注意：这里刻意不改动 self.enabled —— 它来自持久化配置 / 上一次 Enable 节点，
        # 这样 ComfyUI 在 prompt 开始时的批量预取也能直接命中 SSD。

    def on_prompt_end(self, prompt_id: str) -> None:
        stats = self.session
        if stats.lookups == 0 and stats.stores == 0 and stats.stores_pending == 0:
            return
        log.info(
            "SSD-Cache 运行结束 %s：SSD 命中 %d/%d (%.1f%%)，写入 %d（排队 %d），"
            "跳过(不可序列化) %d，耗时 %.1fs",
            prompt_id[:8] if prompt_id else "?",
            stats.hits,
            stats.lookups,
            stats.hit_rate * 100.0,
            stats.stores,
            stats.stores_pending,
            stats.skipped_unsupported,
            stats.elapsed,
        )

    # -------------------------------------------------------------- 报告
    def stats_snapshot(self) -> Dict[str, Any]:
        size, count = self.store.disk_usage()
        return {
            "enabled": self.enabled,
            "cache_dir": self.store.cache_dir,
            "max_entry_mb": self.config.max_entry_mb,
            "session": {
                "prompt_id": self.session.prompt_id,
                "lookups": self.session.lookups,
                "hits": self.session.hits,
                "misses": self.session.misses,
                "hit_rate": self.session.hit_rate,
                "stores": self.session.stores,
                "stores_pending": self.session.stores_pending,
                "store_failed": self.session.store_failed,
                "skipped_unsupported": self.session.skipped_unsupported,
                "skipped_too_large": self.session.skipped_too_large,
                "bytes_written": self.session.bytes_written,
                "elapsed": self.session.elapsed,
            },
            "total": {
                "lookups": self.total.lookups,
                "hits": self.total.hits,
                "misses": self.total.misses,
                "hit_rate": self.total.hit_rate,
                "stores": self.total.stores,
                "stores_pending": self.total.stores_pending,
                "store_failed": self.total.store_failed,
                "skipped_unsupported": self.total.skipped_unsupported,
                "skipped_too_large": self.total.skipped_too_large,
                "bytes_written": self.total.bytes_written,
            },
            "disk": {
                "bytes": size,
                "bytes_human": format_bytes(size),
                "entries": count,
            },
        }

    def format_stats(self) -> str:
        """Stats 节点显示用的多行文本。"""
        snap = self.stats_snapshot()
        session = snap["session"]
        total = snap["total"]
        disk = snap["disk"]

        def _write_text(stat: dict) -> str:
            queued = stat.get("stores_pending", 0)
            total_written = stat["stores"] + queued
            return f"{total_written}（{queued} 排队）" if queued else str(total_written)

        lines = []
        state = "已启用" if snap["enabled"] else "未启用"
        lines.append(f"SSD-Cache: {state}")
        lines.append(
            f"本轮命中: {session['hits']}/{session['lookups']} "
            f"({session['hit_rate'] * 100:.1f}%) · 写入 {_write_text(session)} "
            f"· 跳过 {session['skipped_unsupported'] + session['skipped_too_large']}"
        )
        lines.append(
            f"累计命中: {total['hits']}/{total['lookups']} "
            f"({total['hit_rate'] * 100:.1f}%) · 写入 {_write_text(total)}"
        )
        lines.append(f"磁盘占用: {disk['bytes_human']} · {disk['entries']} 个条目")
        lines.append(f"目录: {snap['cache_dir']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- 单例
_provider: Optional[SSDCacheProvider] = None


def get_provider() -> SSDCacheProvider:
    """获取（并在必要时注册）全局 provider。"""
    global _provider
    if _provider is None:
        _provider = SSDCacheProvider()
        register_cache_provider(_provider)
        try:
            removed = _provider.store.purge_temp_files()
            if removed:
                log.info("SSD-Cache: 清理了 %d 个上次残留的临时文件", removed)
        except Exception:  # noqa: BLE001
            pass
        log.info(
            "SSD-Cache provider 已注册（初始状态: %s，目录: %s）",
            "启用" if _provider.enabled else "禁用",
            _provider.store.cache_dir,
        )
    return _provider
