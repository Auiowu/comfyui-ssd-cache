"""store.py 的单元测试（纯磁盘逻辑，不依赖 ComfyUI 服务）。

运行方式::

    python tests/test_store.py
    # 或
    python -m unittest tests.test_store -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

import torch  # noqa: E402

from ssd_cache.store import (  # noqa: E402
    SSDCacheStore,
    UnsupportedValue,
    estimate_size,
    format_bytes,
    to_saveable,
)


class _FakeModel:
    """模拟 MODEL / CLIP 这类不应被持久化的业务对象。"""


class TestSerialize(unittest.TestCase):
    def test_tensor_moves_to_cpu_and_detaches(self):
        result = to_saveable(torch.ones(4, 4, requires_grad=True))
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.device.type, "cpu")
        self.assertFalse(result.requires_grad)

    def test_nested_latent(self):
        latent = {"samples": torch.zeros(1, 4, 8, 8), "noise_mask": torch.ones(1, 1, 8, 8)}
        result = to_saveable(latent)
        self.assertEqual(set(result), {"samples", "noise_mask"})
        self.assertEqual(result["samples"].shape, (1, 4, 8, 8))

    def test_rejects_model_objects(self):
        with self.assertRaises(UnsupportedValue):
            to_saveable([_FakeModel()])

    def test_rejects_non_primitive_dict_key(self):
        with self.assertRaises(UnsupportedValue):
            to_saveable({(1, 2): "value"})

    def test_estimate_size(self):
        value = {"a": torch.zeros(10, dtype=torch.float32), "b": [torch.zeros(5, dtype=torch.float32)]}
        self.assertEqual(estimate_size(value), 15 * 4)

    def test_format_bytes(self):
        self.assertEqual(format_bytes(0), "0 B")
        self.assertEqual(format_bytes(1024), "1.00 KB")
        self.assertEqual(format_bytes(1536), "1.50 KB")


class TestStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = os.path.join(self._tmp.name, "cache")
        self.store = SSDCacheStore(self.cache_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip(self):
        key = "a" * 64
        outputs = [torch.arange(12, dtype=torch.float32).reshape(3, 4)]
        written = self.store.save(key, outputs, class_type="KSampler", node_id="7")
        self.assertEqual(written, 12 * 4)
        self.assertTrue(self.store.has(key))

        loaded = self.store.load(key)
        self.assertIsInstance(loaded, list)
        self.assertTrue(torch.equal(loaded[0], outputs[0]))

    def test_load_missing_returns_none(self):
        self.assertIsNone(self.store.load("b" * 64))

    def test_unsupported_leaves_no_files(self):
        with self.assertRaises(UnsupportedValue):
            self.store.save("c" * 64, [_FakeModel()])
        self.assertFalse(self.store.has("c" * 64))
        leftovers = [n for _r, _d, files in os.walk(self.store.cache_dir) for n in files]
        self.assertEqual(leftovers, [])

    def test_atomic_write_keeps_old_entry_on_failure(self):
        key = "d" * 64
        self.store.save(key, [torch.zeros(2, 2)])
        original = self.store.load(key)

        with mock.patch("ssd_cache.store.torch.save", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                self.store.save(key, [torch.ones(2, 2)])

        self.assertTrue(torch.equal(self.store.load(key)[0], original[0]))
        leftovers = [
            n
            for _r, _d, files in os.walk(self.store.cache_dir)
            for n in files
            if n.startswith(".tmp-")
        ]
        self.assertEqual(leftovers, [])

    def test_disk_usage_and_clear(self):
        self.store.save("e" * 64, [torch.zeros(8, dtype=torch.float32)])
        self.store.save("f" * 64, [torch.zeros(8, dtype=torch.float32)])

        size, count = self.store.disk_usage(max_age=0.0)
        self.assertEqual(count, 2)
        self.assertGreater(size, 0)

        removed = self.store.clear()
        self.assertGreaterEqual(removed, 2)
        self.assertEqual(self.store.disk_usage(max_age=0.0), (0, 0))

    def test_set_dir_creates_nested_directory(self):
        target = os.path.join(self._tmp.name, "nested", "deep", "cache")
        self.store.set_dir(target)
        self.assertEqual(self.store.cache_dir, os.path.abspath(target))
        self.assertTrue(os.path.isdir(target))


if __name__ == "__main__":
    unittest.main(verbosity=2)
