"""memory_cache.py 的单元测试：内存开关、字节上限 LRU 淘汰。"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

_COMFY_ROOT = os.environ.get(
    "COMFYUI_ROOT", r"D:\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
)
if os.path.isdir(_COMFY_ROOT) and _COMFY_ROOT not in sys.path:
    sys.path.insert(0, _COMFY_ROOT)

import torch  # noqa: E402
from comfy_execution.caching import CacheKeySetID  # noqa: E402

from ssd_cache.memory_cache import SSDMemoryCache, value_bytes  # noqa: E402


class FakeProvider:
    def __init__(self, enabled=True, limit_bytes=0):
        self.memory_cache_enabled = enabled
        self.memory_cache_max_bytes = limit_bytes


class FakeDynPrompt:
    def __init__(self, nodes):
        self.nodes = nodes

    def has_node(self, node_id):
        return node_id in self.nodes

    def get_node(self, node_id):
        return self.nodes[node_id]

    def get_parent_node_id(self, node_id):
        return None


def _run(coro):
    return asyncio.run(coro)


class _Entry:
    """模拟 comfy_execution.execution.CacheEntry。"""

    def __init__(self, outputs, ui=None):
        self.outputs = outputs
        self.ui = ui


def _make_cache(provider, node_ids=("1", "2")):
    nodes = {node_id: {"class_type": "KSampler", "inputs": {}} for node_id in node_ids}
    cache = SSDMemoryCache(CacheKeySetID, provider)
    _run(cache.set_prompt(FakeDynPrompt(nodes), list(node_ids), None))
    return cache


class TestValueBytes(unittest.TestCase):
    def test_tensor_size(self):
        entry = _Entry([{"samples": torch.zeros(10, dtype=torch.float32)}])
        self.assertEqual(value_bytes(entry), 40)

    def test_nested(self):
        entry = _Entry([[torch.zeros(2, dtype=torch.float32), {"x": torch.zeros(3, dtype=torch.float32)}]])
        self.assertEqual(value_bytes(entry), 20)


class TestMemoryCache(unittest.TestCase):
    def test_memory_enabled_keeps_entry(self):
        provider = FakeProvider(enabled=True, limit_bytes=0)
        cache = _make_cache(provider)
        entry = _Entry([torch.zeros(4, dtype=torch.float32)])

        _run(cache._set_immediate("1", entry))
        self.assertEqual(cache.memory_entries, 1)
        self.assertEqual(cache.memory_bytes, 16)
        self.assertIs(_run(cache._get_immediate("1")), entry)

    def test_memory_disabled_does_not_keep_entry(self):
        provider = FakeProvider(enabled=False, limit_bytes=0)
        cache = _make_cache(provider)
        entry = _Entry([torch.zeros(4, dtype=torch.float32)])

        _run(cache._set_immediate("1", entry))
        self.assertEqual(cache.memory_entries, 0)
        self.assertEqual(cache.memory_bytes, 0)
        # 没有内存也没有 SSD provider 时会 miss
        self.assertIsNone(_run(cache._get_immediate("1")))

    def test_lru_evicts_oldest_when_over_limit(self):
        # 每条 16 字节，上限 24 字节 -> 只能放 1 条
        provider = FakeProvider(enabled=True, limit_bytes=24)
        cache = _make_cache(provider)
        e1 = _Entry([torch.zeros(4, dtype=torch.float32)])
        e2 = _Entry([torch.zeros(4, dtype=torch.float32)])

        _run(cache._set_immediate("1", e1))
        _run(cache._set_immediate("2", e2))

        self.assertEqual(cache.memory_entries, 1)
        self.assertLessEqual(cache.memory_bytes, 24)
        # 最早插入的 1 被淘汰
        self.assertIsNone(cache.get_local("1"))
        self.assertIs(_run(cache._get_immediate("2")), e2)

    def test_touch_updates_lru_order(self):
        provider = FakeProvider(enabled=True, limit_bytes=24)
        cache = _make_cache(provider)
        e1 = _Entry([torch.zeros(4, dtype=torch.float32)])
        e2 = _Entry([torch.zeros(4, dtype=torch.float32)])

        _run(cache._set_immediate("1", e1))
        _run(cache._set_immediate("2", e2))  # 淘汰 1
        self.assertIsNone(cache.get_local("1"))

        _run(cache._set_immediate("1", e1))  # 现在 1 是最新
        _run(cache._set_immediate("2", e2))  # 淘汰 1
        self.assertIsNone(cache.get_local("1"))
        self.assertIs(_run(cache._get_immediate("2")), e2)

    def test_switch_to_disabled_clears_memory(self):
        provider = FakeProvider(enabled=True, limit_bytes=0)
        cache = _make_cache(provider)
        _run(cache._set_immediate("1", _Entry([torch.zeros(4, dtype=torch.float32)])))
        self.assertEqual(cache.memory_entries, 1)

        provider.memory_cache_enabled = False
        _run(cache._get_immediate("1"))  # 触发清理
        self.assertEqual(cache.memory_entries, 0)

    def test_ram_release_is_noop(self):
        provider = FakeProvider(enabled=True, limit_bytes=0)
        cache = _make_cache(provider)
        self.assertEqual(cache.ram_release(0, free_active=True), 0)

    def test_memory_stats_properties(self):
        provider = FakeProvider(enabled=True, limit_bytes=64)
        cache = _make_cache(provider)
        _run(cache._set_immediate("1", _Entry([torch.zeros(4, dtype=torch.float32)])))
        self.assertEqual(cache.memory_entries, 1)
        self.assertEqual(cache.memory_limit_bytes, 64)


class TestCacheScopeSwitch(unittest.TestCase):
    """没有 Enable 节点时必须回到 ComfyUI 官方缓存层。"""

    def _executor(self):
        from ssd_cache.memory_cache import apply_cache_scope  # noqa: F401

        class Caches:
            def __init__(self):
                self.objects = object()
                self._ssd_outputs = "ssd-layer"
                self._official_outputs = "official-layer"
                self.outputs = self._official_outputs
                self.all = None

        class Executor:
            def __init__(self):
                self.caches = Caches()

        return Executor()

    def test_switch_to_ssd(self):
        from ssd_cache.memory_cache import apply_cache_scope

        executor = self._executor()
        provider = type("P", (), {"active_cache_kind": None})()
        self.assertTrue(apply_cache_scope(executor, True, provider))
        self.assertEqual(executor.caches.outputs, "ssd-layer")
        self.assertEqual(provider.active_cache_kind, "ssd")

    def test_switch_back_to_official(self):
        from ssd_cache.memory_cache import apply_cache_scope

        executor = self._executor()
        provider = type("P", (), {"active_cache_kind": None})()
        apply_cache_scope(executor, True, provider)
        self.assertTrue(apply_cache_scope(executor, False, provider))
        self.assertEqual(executor.caches.outputs, "official-layer")
        self.assertEqual(provider.active_cache_kind, "official")

    def test_no_cache_set_is_safe(self):
        from ssd_cache.memory_cache import apply_cache_scope

        provider = type("P", (), {"active_cache_kind": None})()
        executor = type("E", (), {})()
        self.assertFalse(apply_cache_scope(executor, True, provider))


if __name__ == "__main__":
    unittest.main(verbosity=2)
