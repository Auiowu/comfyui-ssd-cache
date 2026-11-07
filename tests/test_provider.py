"""provider.py 的单元测试：缓存查询 / 写入 / 过滤 / 统计。

运行方式::

    python tests/test_provider.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from unittest import mock

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

_COMFY_ROOT = os.environ.get(
    "COMFYUI_ROOT", r"D:\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
)
if os.path.isdir(_COMFY_ROOT) and _COMFY_ROOT not in sys.path:
    sys.path.insert(0, _COMFY_ROOT)

import torch  # noqa: E402

try:
    from comfy_execution.cache_provider import CacheContext, CacheValue  # noqa: E402

    _HAS_COMFY = True
    _COMFY_IMPORT_ERROR = ""
except Exception as _exc:  # noqa: BLE001 - CI 环境没有 ComfyUI 时自动跳过
    CacheContext = CacheValue = None  # type: ignore[assignment]
    _HAS_COMFY = False
    _COMFY_IMPORT_ERROR = str(_exc)

from ssd_cache.config import SSDCacheConfig  # noqa: E402
from ssd_cache.provider import SSDCacheProvider  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _context(node_id="1", class_type="KSampler", key="a" * 64):
    return CacheContext(node_id=node_id, class_type=class_type, cache_key_hash=key)


@unittest.skipUnless(_HAS_COMFY, f"需要 ComfyUI 环境: {_COMFY_IMPORT_ERROR}")
class TestProvider(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._tmpdir = self._tmp.name
        # 避免测试写坏真实的 config.json
        self._patch = mock.patch("ssd_cache.provider.save_config", lambda _cfg: None)
        self._patch.start()
        self.provider = SSDCacheProvider(
            config=SSDCacheConfig(
                cache_dir=os.path.join(self._tmpdir, "cache"),
                enabled=True,
                max_entry_mb=0,
            )
        )

    def tearDown(self):
        self.provider.flush(10)
        self.provider.shutdown()
        self._patch.stop()
        self._tmp.cleanup()

    def test_configure_creates_directory(self):
        target = os.path.join(self._tmpdir, "brand", "new", "dir")
        self.assertTrue(self.provider.configure(target, True, 512))
        self.assertTrue(os.path.isdir(target))
        self.assertTrue(self.provider.enabled)
        self.assertEqual(self.provider.max_entry_bytes, 512 * 1024 * 1024)

    def test_store_then_lookup_roundtrip(self):
        ctx = _context()
        outputs = [{"samples": torch.zeros(1, 4, 8, 8)}]

        self.assertTrue(self.provider.should_cache(ctx, CacheValue(outputs=outputs, ui=None)))
        _run(self.provider.on_store(ctx, CacheValue(outputs=outputs, ui=None)))
        self.assertTrue(self.provider.flush(10))
        self.assertEqual(self.provider.session.stores, 1)
        self.assertEqual(self.provider.session.stores_pending, 0)

        result = _run(self.provider.on_lookup(ctx))
        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(result.outputs[0]["samples"], outputs[0]["samples"]))

    def test_should_cache_marks_pending_until_store_finishes(self):
        ctx = _context(key="k" * 64)
        value = CacheValue(outputs=[torch.zeros(2)], ui=None)
        self.assertTrue(self.provider.should_cache(ctx, value))
        self.assertEqual(self.provider.session.stores_pending, 1)

        _run(self.provider.on_store(ctx, value))
        self.assertTrue(self.provider.flush(10))
        self.assertEqual(self.provider.session.stores_pending, 0)
        self.assertEqual(self.provider.session.stores, 1)

    def test_lookup_miss_counts(self):
        self.assertIsNone(_run(self.provider.on_lookup(_context(key="b" * 64))))
        self.assertEqual(self.provider.session.lookups, 1)
        self.assertEqual(self.provider.session.hits, 0)
        self.assertEqual(self.provider.session.misses, 1)
        self.assertEqual(self.provider.session.hit_rate, 0.0)

    def test_hit_rate(self):
        hit_ctx = _context(key="c" * 64)
        _run(self.provider.on_store(hit_ctx, CacheValue(outputs=[torch.zeros(2)], ui=None)))
        self.assertTrue(self.provider.flush(10))
        _run(self.provider.on_lookup(hit_ctx))
        _run(self.provider.on_lookup(_context(key="d" * 64)))

        self.assertEqual(self.provider.session.lookups, 2)
        self.assertEqual(self.provider.session.hits, 1)
        self.assertEqual(self.provider.session.hit_rate, 0.5)

    def test_disabled_provider_does_nothing(self):
        self.provider.enabled = False
        ctx = _context(key="e" * 64)
        self.assertFalse(self.provider.should_cache(ctx))
        _run(self.provider.on_store(ctx, CacheValue(outputs=[torch.zeros(2)], ui=None)))
        self.assertEqual(self.provider.session.stores, 0)
        self.assertIsNone(_run(self.provider.on_lookup(ctx)))
        self.assertEqual(self.provider.session.lookups, 0)

    def test_output_nodes_are_not_cached(self):
        self.provider._output_node_cache["SaveImage"] = True
        self.assertFalse(self.provider.should_cache(_context(class_type="SaveImage")))
        self.assertTrue(self.provider.should_cache(_context(class_type="KSampler")))

    def test_unsupported_output_is_skipped(self):
        class _Model:
            pass

        ctx = _context(key="f" * 64)
        _run(self.provider.on_store(ctx, CacheValue(outputs=[_Model()], ui=None)))
        self.assertTrue(self.provider.flush(10))
        self.assertEqual(self.provider.session.stores, 0)
        self.assertEqual(self.provider.session.skipped_unsupported, 1)
        self.assertIsNone(self.provider.store.load(ctx.cache_key_hash))

    def test_empty_outputs_are_not_stored(self):
        ctx = _context(key="g" * 64)
        self.assertFalse(self.provider.should_cache(ctx, CacheValue(outputs=[], ui=None)))

    def test_max_entry_size_guard(self):
        self.provider.max_entry_bytes = 1024
        ctx = _context(key="h" * 64)
        big = CacheValue(outputs=[torch.zeros(1024, dtype=torch.float32)], ui=None)  # 4 KiB
        self.assertFalse(self.provider.should_cache(ctx, big))
        self.assertEqual(self.provider.session.skipped_too_large, 1)

    def test_prompt_lifecycle_resets_session(self):
        self.provider.on_prompt_start("prompt-1")
        _run(self.provider.on_lookup(_context(key="i" * 64)))
        self.assertEqual(self.provider.session.lookups, 1)

        self.provider.on_prompt_start("prompt-2")
        self.assertEqual(self.provider.session.lookups, 0)
        self.assertEqual(self.provider.session.prompt_id, "prompt-2")

    def test_format_stats_contains_key_fields(self):
        self.provider.on_prompt_start("prompt-x")
        _run(self.provider.on_lookup(_context(key="j" * 64)))
        text = self.provider.format_stats()
        self.assertIn("SSD-Cache", text)
        self.assertIn("本轮 SSD", text)
        self.assertIn("磁盘", text)
        self.assertIn(self.provider.store.cache_dir, text)

    def test_unstable_node_detected_after_repeated_misses(self):
        ctx_a = _context(node_id="504", class_type="SamplerCustomAdvanced", key="a" * 64)
        ctx_b = _context(node_id="504", class_type="SamplerCustomAdvanced", key="b" * 64)
        _run(self.provider.on_lookup(ctx_a))
        _run(self.provider.on_lookup(ctx_b))

        unstable = self.provider.unstable_nodes()
        self.assertEqual(len(unstable), 1)
        self.assertIn("SamplerCustomAdvanced#504", unstable[0])
        self.assertIn("(2)", unstable[0])
        self.assertIn("键不稳定", self.provider.format_stats())

    def test_stable_node_not_reported_unstable(self):
        ctx = _context(node_id="7", class_type="KSampler", key="c" * 64)
        _run(self.provider.on_lookup(ctx))
        _run(self.provider.on_lookup(ctx))  # 同一个键 miss 两次 = 稳定
        self.assertEqual(self.provider.unstable_nodes(), [])

    def test_clear_diagnostics(self):
        _run(self.provider.on_lookup(_context(node_id="1", key="d" * 64)))
        _run(self.provider.on_lookup(_context(node_id="1", key="e" * 64)))
        self.assertTrue(self.provider.unstable_nodes())
        self.provider.clear_diagnostics()
        self.assertEqual(self.provider.unstable_nodes(), [])

    def test_node_hit_rate_includes_memory_skipped(self):
        self.provider.on_prompt_start("p")
        # 模拟：10 个节点被跳过（其中 4 个来自 SSD），7 个实际执行
        self.provider.session.cached_node_ids = {str(i) for i in range(10)}
        self.provider.session.executed_node_ids = {str(i) for i in range(100, 107)}
        self.provider.session.hits = 4
        self.assertAlmostEqual(self.provider.session.node_hit_rate, 10 / 17)

        text = self.provider.format_stats()
        self.assertIn("本轮命中率: 10/17 (58.8%)", text)
        self.assertIn("SSD 恢复 4", text)
        self.assertIn("内存/显存 6", text)
        self.assertIn("实际计算 7", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
