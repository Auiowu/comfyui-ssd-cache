"""内存/显存缓存层：可开关、可设上限，超出上限时按 LRU 淘汰。

替换 ComfyUI 的 ``CacheSet.outputs``（运行时 patch，不改官方源码）：

* ``memory_cache=False``  -> 节点结果只写 SSD，读取时直接查 SSD；
* ``memory_cache=True``   -> 结果先进内存，超过 ``memory_cache_mb`` 时淘汰最旧的；
* 无论哪种模式，结果都会异步写入 SSD，所以重启后仍能恢复。

读取顺序：内存 -> SSD -> 返回 None（交给 ComfyUI 真正计算）。
"""

from __future__ import annotations

import collections
import logging
from typing import Any, Optional

from comfy_execution.caching import HierarchicalCache

log = logging.getLogger("SSD-Cache")


def value_bytes(value: Any) -> int:
    """估算一个缓存值（CacheEntry 或裸输出）占用的字节数。"""
    if value is None:
        return 0
    outputs = getattr(value, "outputs", value)
    total = 0
    stack = [outputs]
    while stack:
        item = stack.pop()
        try:
            import torch

            if torch.is_tensor(item):
                total += item.numel() * item.element_size()
                continue
        except Exception:  # noqa: BLE001
            pass
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return total


class SSDMemoryCache(HierarchicalCache):
    """带字节上限的 LRU 内存缓存，未命中时回落到 SSD provider。"""

    def __init__(self, key_class, provider) -> None:
        super().__init__(key_class, enable_providers=True)
        self.provider = provider
        self._lru: "collections.OrderedDict[str, None]" = collections.OrderedDict()
        self._sizes: dict = {}
        self._bytes = 0
        # ComfyUI 的 RAM_PRESSURE 分支会读取这两个字段（仅用于日志），保留兼容
        self.active_evictions = False
        self.full_evictions = False

    # ------------------------------------------------------------------ 配置
    def _memory_on(self) -> bool:
        return bool(getattr(self.provider, "memory_cache_enabled", False))

    def _limit(self) -> int:
        return int(getattr(self.provider, "memory_cache_max_bytes", 0) or 0)

    # ------------------------------------------------------------------ 内存维护
    def _touch(self, key: str) -> None:
        if key in self._lru:
            self._lru.move_to_end(key)

    def _put(self, key: str, value: Any) -> None:
        size = value_bytes(value)
        if key in self.cache:
            self._bytes -= self._sizes.get(key, 0)
        self.cache[key] = value
        self._sizes[key] = size
        self._bytes += size
        self._lru[key] = None
        self._lru.move_to_end(key)
        self._enforce_limit()

    def _drop(self, key: str) -> None:
        self.cache.pop(key, None)
        self._bytes -= self._sizes.pop(key, 0)
        self._lru.pop(key, None)

    def _drop_all(self) -> None:
        self.cache.clear()
        self._sizes.clear()
        self._lru.clear()
        self._bytes = 0

    def _enforce_limit(self) -> None:
        limit = self._limit()
        if limit <= 0:
            return
        while self._bytes > limit and self._lru:
            key, _ = self._lru.popitem(last=False)
            self.cache.pop(key, None)
            self._bytes -= self._sizes.pop(key, 0)

    def _clean_cache(self) -> None:
        if not self.initialized:
            return
        preserve = set(self.cache_key_set.get_used_keys())
        for key in list(self.cache):
            if key not in preserve:
                self._drop(key)

    # ------------------------------------------------------------------ 读写
    async def _get_immediate(self, node_id):
        if not self.initialized:
            return None

        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key is None:
            return None

        if not self._memory_on():
            # 从「允许内存」切到「禁用内存」时，把旧内存条目清掉
            if self.cache:
                self._drop_all()
        elif cache_key in self.cache:
            self._touch(cache_key)
            return self.cache[cache_key]

        # 内存没有 -> 查 SSD
        entry = await self._check_providers_lookup(node_id, cache_key)
        if entry is not None and self._memory_on():
            self._put(cache_key, entry)
        return entry

    async def _set_immediate(self, node_id, value) -> None:
        if not self.initialized:
            return

        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key is None:
            return

        if self._memory_on():
            self._put(cache_key, value)
        elif self.cache:
            self._drop_all()

        # 无论是否进内存，都异步落盘；这样重启后仍可恢复
        await self._notify_providers_store(node_id, cache_key, value)

    # ------------------------------------------------------------------ 兼容接口
    def get_local(self, node_id):
        if not self.initialized:
            return None
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key is None:
            return None
        value = self.cache.get(cache_key)
        if value is not None:
            self._touch(cache_key)
        return value

    def set_local(self, node_id, value):
        if not self.initialized:
            return
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key is not None:
            self._put(cache_key, value)

    def ram_release(self, target, free_active: bool = False, min_entry_size: int = 0) -> int:
        """ComfyUI 在 RAM 压力模式下会调用；这里由用户设定的上限接管，直接返回 0。"""
        return 0

    async def _ensure_subcache(self, node_id, children_ids):
        subcache_key = self.cache_key_set.get_subcache_key(node_id)
        subcache = self.subcaches.get(subcache_key)
        if subcache is None:
            subcache = SSDMemoryCache(self.key_class, self.provider)
            self.subcaches[subcache_key] = subcache
        await subcache.set_prompt(self.dynprompt, children_ids, self.is_changed_cache)
        return subcache

    # ------------------------------------------------------------------ 报告
    @property
    def memory_bytes(self) -> int:
        return self._bytes

    @property
    def memory_entries(self) -> int:
        return len(self.cache)

    @property
    def memory_limit_bytes(self) -> int:
        return self._limit()


def install_cache_override(provider) -> bool:
    """把 ComfyUI 的 ``CacheSet.outputs`` 替换成 :class:`SSDMemoryCache`。

    在 custom node 加载时调用即可：ComfyUI 的 ``PromptExecutor`` 之后才创建，
    所以会直接使用我们这一层。整个过程不修改任何官方源文件。
    """
    try:
        import execution
    except Exception as exc:  # noqa: BLE001
        log.debug("SSD-Cache: 无法导入 execution，跳过缓存层替换: %s", exc)
        return False

    cache_set_cls = getattr(execution, "CacheSet", None)
    if cache_set_cls is None or getattr(cache_set_cls, "_ssd_cache_patched", False):
        return getattr(cache_set_cls, "_ssd_cache_patched", False)

    original_init = cache_set_cls.__init__

    def patched_init(self, cache_type=None, cache_args=None):
        original_init(self, cache_type, cache_args or {})
        try:
            memory_cache = SSDMemoryCache(execution.CacheKeySetInputSignature, provider)
            self.outputs = memory_cache
            self.all = [self.outputs, self.objects]
            provider.attach_memory_cache(memory_cache)
        except Exception as exc:  # noqa: BLE001
            log.warning("SSD-Cache: 替换缓存层失败，沿用 ComfyUI 原实现: %s", exc)

    cache_set_cls.__init__ = patched_init
    cache_set_cls._ssd_cache_patched = True
    log.info("SSD-Cache: 已接管 ComfyUI 缓存层（内存/显存上限可控）")
    return True
