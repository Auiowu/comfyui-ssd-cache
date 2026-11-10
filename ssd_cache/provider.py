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
    #: 本轮被 ComfyUI 内存/显存缓存直接跳过的节点（来自 execution_cached 事件）
    cached_node_ids: set = field(default_factory=set)
    #: 本轮真正执行的节点（来自 executing 事件）
    executed_node_ids: set = field(default_factory=set)
    #: 累计跳过的节点数（on_prompt_end 时汇总）
    cached_node_count: int = 0
    #: 累计实际执行的节点数
    executed_node_count: int = 0

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
        self.cached_node_ids = set()
        self.executed_node_ids = set()

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.lookups) if self.lookups else 0.0

    @property
    def node_hit_rate(self) -> float:
        """总命中率：跳过计算的节点 / 本轮涉及的全部节点。"""
        total = len(self.cached_node_ids) + len(self.executed_node_ids)
        return (len(self.cached_node_ids) / total) if total else 0.0

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
        #: 是否允许节点结果保留在内存/显存中
        self.memory_cache_enabled: bool = bool(self.config.memory_cache)
        #: 内存/显存缓存上限（字节），0 = 不限
        self.memory_cache_max_bytes: int = max(0, int(self.config.memory_cache_mb)) * 1024 * 1024
        self._memory_cache = None
        #: 当前生效的缓存层：``ssd`` 或 ``official``
        self.active_cache_kind = "official"

        self.session = CacheStats()
        self.total = CacheStats()

        self._output_node_cache: Dict[str, bool] = {}
        self._stats_lock = threading.Lock()
        self._futures_lock = threading.Lock()
        self._pending_futures: set = set()
        self._write_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._read_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None

        # 键稳定性诊断：同一节点在多次 miss 中出现过哪些不同的 cache_key_hash。
        # 跨 prompt 保留，这样用户连跑两次后就能看到是哪个节点在变。
        self._key_history: Dict[str, Dict[str, Any]] = {}
        self._hooked = False

    # -------------------------------------------------------------- 配置
    def configure(
        self,
        cache_dir: str,
        enabled: bool,
        max_entry_mb: int,
        memory_cache: bool = True,
        memory_cache_mb: int = 0,
    ) -> bool:
        """由 Enable 节点调用：切换目录 / 开关 / 单条上限 / 内存上限，并持久化。"""
        cache_dir = os.path.abspath(os.path.expanduser(str(cache_dir or DEFAULT_CACHE_DIR)))
        try:
            self.store.set_dir(cache_dir)
        except OSError as exc:
            log.error("SSD-Cache: 无法创建缓存目录 %s: %s", cache_dir, exc)
            self.enabled = False
            return False

        self.enabled = bool(enabled)
        self.max_entry_bytes = max(0, int(max_entry_mb)) * 1024 * 1024
        self.memory_cache_enabled = bool(memory_cache)
        self.memory_cache_max_bytes = max(0, int(memory_cache_mb)) * 1024 * 1024

        self.config.cache_dir = cache_dir
        self.config.enabled = bool(enabled)
        self.config.max_entry_mb = max(0, int(max_entry_mb))
        self.config.memory_cache = bool(memory_cache)
        self.config.memory_cache_mb = max(0, int(memory_cache_mb))
        save_config(self.config)

        log.info(
            "SSD-Cache: %s，目录=%s，单条上限=%s，内存缓存=%s",
            "已启用" if enabled else "已禁用",
            cache_dir,
            "不限" if self.max_entry_bytes == 0 else format_bytes(self.max_entry_bytes),
            (
                "不限" if self.memory_cache_max_bytes == 0
                else format_bytes(self.memory_cache_max_bytes)
            )
            if self.memory_cache_enabled
            else "关闭",
        )
        return True

    def attach_memory_cache(self, cache) -> None:
        """由缓存层 patch 回调，保存当前内存缓存实例用于统计。"""
        self._memory_cache = cache

    # -------------------------------------------------------------- 工作流作用域
    ENABLE_NODE_TYPE = "SSDCacheEnable"

    def begin_prompt(self, prompt) -> bool:
        """每次运行开始前调用：只有工作流里放了 Enable 节点才启用缓存。

        返回本次是否启用。这里只调整运行时状态，不写 config.json ——
        持久化配置由 Enable 节点执行时负责。
        """
        node = None
        if isinstance(prompt, dict):
            for item in prompt.values():
                if isinstance(item, dict) and item.get("class_type") == self.ENABLE_NODE_TYPE:
                    node = item
                    break

        if node is None:
            self.enabled = False
            log.debug("SSD-Cache: 本次工作流未包含 Enable 节点，缓存已禁用")
            return False

        inputs = node.get("inputs") or {}

        enabled = inputs.get("enable")
        if not isinstance(enabled, bool):
            # 输入是链接时无法静态求值，回退到持久化配置
            enabled = bool(self.config.enabled)

        cache_dir = inputs.get("cache_dir")
        if isinstance(cache_dir, str) and cache_dir.strip():
            try:
                self.store.set_dir(cache_dir)
            except OSError as exc:
                log.warning("SSD-Cache: 无法切换到缓存目录 %s: %s", cache_dir, exc)

        max_entry_mb = inputs.get("max_entry_mb")
        if isinstance(max_entry_mb, (int, float)):
            self.max_entry_bytes = max(0, int(max_entry_mb)) * 1024 * 1024

        memory_cache = inputs.get("memory_cache")
        if isinstance(memory_cache, bool):
            self.memory_cache_enabled = memory_cache

        memory_cache_mb = inputs.get("memory_cache_mb")
        if isinstance(memory_cache_mb, (int, float)):
            self.memory_cache_max_bytes = max(0, int(memory_cache_mb)) * 1024 * 1024

        self.enabled = enabled
        return enabled

    def memory_stats(self) -> Dict[str, Any]:
        if getattr(self, "active_cache_kind", "official") == "official":
            return {
                "kind": "official",
                "enabled": True,
                "bytes": 0,
                "bytes_human": "0 B",
                "entries": 0,
                "limit_bytes": 0,
                "limit_human": "官方默认",
            }
        cache = self._memory_cache
        if cache is None:
            return {
                "kind": "ssd",
                "enabled": self.memory_cache_enabled,
                "bytes": 0,
                "bytes_human": "0 B",
                "entries": 0,
                "limit_bytes": self.memory_cache_max_bytes,
                "limit_human": "不限" if self.memory_cache_max_bytes == 0 else format_bytes(self.memory_cache_max_bytes),
            }
        return {
            "kind": "ssd",
            "enabled": self.memory_cache_enabled,
            "bytes": cache.memory_bytes,
            "bytes_human": format_bytes(cache.memory_bytes),
            "entries": cache.memory_entries,
            "limit_bytes": self.memory_cache_max_bytes,
            "limit_human": "不限" if self.memory_cache_max_bytes == 0 else format_bytes(self.memory_cache_max_bytes),
        }

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

    # -------------------------------------------------------------- 执行统计钩子
    def _install_hooks(self) -> None:
        """挂接 PromptServer.send_sync，记录本轮「跳过 / 实际执行」的节点。

        ComfyUI 的 ``execution_cached`` 事件列出被内存缓存直接跳过的节点，
        ``executing`` 事件则每个真正要执行的节点发一次。有了这两个数字，
        Stats 才能区分「内存命中」「SSD 恢复」「实际计算」。
        """
        if self._hooked:
            return
        try:
            from server import PromptServer

            server = getattr(PromptServer, "instance", None)
            if server is None:
                return
            if getattr(server, "_ssd_cache_hooked", False):
                self._hooked = True
                return

            original = server.send_sync
            provider = self

            def _send_sync(event, data, sid=None):
                try:
                    if isinstance(data, dict):
                        if event == "execution_cached":
                            nodes = data.get("nodes") or []
                            with provider._stats_lock:
                                provider.session.cached_node_ids = {str(n) for n in nodes}
                        elif event == "executing":
                            node_id = data.get("node")
                            if node_id is not None:
                                with provider._stats_lock:
                                    provider.session.executed_node_ids.add(str(node_id))
                except Exception:  # noqa: BLE001 - 统计失败绝不能影响执行
                    pass
                return original(event, data, sid)

            server.send_sync = _send_sync
            server._ssd_cache_hooked = True
            self._hooked = True
            log.debug("SSD-Cache: 已挂接执行统计钩子")
        except Exception as exc:  # noqa: BLE001
            log.debug("SSD-Cache: 无法挂接执行统计钩子: %s", exc)

    # -------------------------------------------------------------- 键稳定性诊断
    def _record_miss_key(self, context: CacheContext) -> None:
        label = f"{context.class_type}#{context.node_id}"
        with self._stats_lock:
            entry = self._key_history.get(label)
            if entry is None:
                entry = {"hashes": set(), "last": 0.0}
                self._key_history[label] = entry
            if len(entry["hashes"]) < 8:
                entry["hashes"].add(context.cache_key_hash)
            entry["last"] = time.time()

    def unstable_nodes(self, min_distinct: int = 2, limit: int = 5) -> list:
        """返回反复 miss 且每次键都不同的节点，如 ``SamplerCustomAdvanced#504(3)``。"""
        rows = []
        with self._stats_lock:
            for label, entry in self._key_history.items():
                distinct = len(entry["hashes"])
                if distinct >= min_distinct:
                    rows.append((distinct, entry["last"], label))
        rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [f"{label}({distinct})" for distinct, _last, label in rows[:limit]]

    def clear_diagnostics(self) -> None:
        with self._stats_lock:
            self._key_history.clear()

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
            self._record_miss_key(context)
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
        self._install_hooks()
        # 注意：这里刻意不改动 self.enabled —— 它来自持久化配置 / 上一次 Enable 节点，
        # 这样 ComfyUI 在 prompt 开始时的批量预取也能直接命中 SSD。

    def on_prompt_end(self, prompt_id: str) -> None:
        stats = self.session
        # 把本轮的节点跳过/执行数累加进 total（用于累计命中率）
        with self._stats_lock:
            self.total.cached_node_count += len(stats.cached_node_ids)
            self.total.executed_node_count += len(stats.executed_node_ids)

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
                "cached_nodes": len(self.session.cached_node_ids),
                "executed_nodes": len(self.session.executed_node_ids),
                "node_hit_rate": self.session.node_hit_rate,
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
                "cached_nodes": self.total.cached_node_count,
                "executed_nodes": self.total.executed_node_count,
            },
            "disk": {
                "bytes": size,
                "bytes_human": format_bytes(size),
                "entries": count,
            },
            "memory": self.memory_stats(),
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

        cached = session.get("cached_nodes", 0)
        executed = session.get("executed_nodes", 0)
        if cached or executed:
            total_nodes = cached + executed
            ssd_restored = min(session["hits"], cached)
            memory_only = max(0, cached - ssd_restored)
            lines.append(
                f"本轮命中率: {cached}/{total_nodes} "
                f"({session.get('node_hit_rate', 0.0) * 100:.1f}%)"
                f" = SSD 恢复 {ssd_restored} + 内存/显存 {memory_only}"
                f" · 实际计算 {executed}"
            )

        total_cached = total.get("cached_nodes", 0)
        total_executed = total.get("executed_nodes", 0)
        if total_cached or total_executed:
            total_nodes = total_cached + total_executed
            rate = (total_cached / total_nodes * 100) if total_nodes else 0.0
            lines.append(
                f"累计命中率: {total_cached}/{total_nodes} ({rate:.1f}%)"
                f" · SSD 恢复 {total['hits']} · 实际计算 {total_executed}"
            )

        lines.append(
            f"本轮 SSD: {session['hits']}/{session['lookups']} 命中 "
            f"({session['hit_rate'] * 100:.1f}%) · 写入 {_write_text(session)} "
            f"· 不可缓存 {session['skipped_unsupported'] + session['skipped_too_large']}"
        )
        lines.append(
            f"累计 SSD: {total['hits']}/{total['lookups']} 命中 "
            f"({total['hit_rate'] * 100:.1f}%) · 写入 {_write_text(total)}"
        )
        lines.append(f"磁盘: {disk['bytes_human']} · {disk['entries']} 条 | 目录: {snap['cache_dir']}")

        memory = snap.get("memory") or {}
        if memory.get("kind") == "official":
            lines.append("内存/显存: ComfyUI 官方默认缓存（本工作流未放 Enable 节点）")
        elif memory.get("enabled"):
            lines.append(
                f"内存/显存: {memory.get('bytes_human', '0 B')} / "
                f"{memory.get('limit_human', '不限')} · {memory.get('entries', 0)} 条"
            )
        else:
            lines.append("内存/显存: 已关闭（只写 SSD）")

        unstable = self.unstable_nodes()
        if unstable:
            lines.append("⚠ 键不稳定（同一节点每次键不同）: " + ", ".join(unstable))
            lines.append("  常见原因：随机种子（RandomNoise / KSampler 的 control_after_generate）、时间戳类节点")
        elif session["lookups"] and session["hits"] == 0 and not cached:
            lines.append("提示：内存命中的节点不会查询 SSD，验证 SSD 命中请重启 ComfyUI 后再跑")

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
