"""CreateBlankImage 节点的单元测试。"""

from __future__ import annotations

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

from ssd_cache.blank_image import (  # noqa: E402
    BLANK_GRAY,
    CreateBlankImage,
    compute_blank_size,
    parse_hex_color,
    resolve_color,
)


class TestColor(unittest.TestCase):
    def test_hex_full(self):
        self.assertEqual(parse_hex_color("#ff0000"), (255, 0, 0))
        self.assertEqual(parse_hex_color("00ff00"), (0, 255, 0))

    def test_hex_short(self):
        self.assertEqual(parse_hex_color("#f00"), (255, 0, 0))
        self.assertEqual(parse_hex_color("#0f0"), (0, 255, 0))

    def test_hex_invalid_falls_back_to_gray(self):
        self.assertEqual(parse_hex_color("nope"), (127, 127, 127))
        self.assertEqual(parse_hex_color(""), (127, 127, 127))

    def test_rgb_mode_clamps(self):
        self.assertEqual(resolve_color("RGB", "#000000", 300, -5, 128), (255, 0, 128))

    def test_hex_mode_ignores_rgb(self):
        self.assertEqual(resolve_color("十六进制 (Hex)", "#123456", 0, 0, 0), (18, 52, 86))


class TestComputeSize(unittest.TestCase):
    def test_square_1mp(self):
        self.assertEqual(compute_blank_size("1:1 (Square)", 1.0, 8), (1000, 1000))

    def test_portrait_3_4_2mp(self):
        width, height = compute_blank_size("3:4 (Portrait Standard)", 2.0, 16)
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)
        self.assertAlmostEqual(width / height, 3 / 4, delta=0.02)
        self.assertAlmostEqual(width * height / 1e6, 2.0, delta=0.1)

    def test_landscape_16_9_1mp(self):
        width, height = compute_blank_size("16:9 (Landscape Widescreen)", 1.0, 8)
        self.assertAlmostEqual(width / height, 16 / 9, delta=0.02)
        self.assertAlmostEqual(width * height / 1e6, 1.0, delta=0.1)

    def test_multiple_alignment(self):
        for multiple in (1, 8, 16, 64):
            width, height = compute_blank_size("16:9 (Landscape Widescreen)", 1.0, multiple)
            self.assertEqual(width % multiple, 0)
            self.assertEqual(height % multiple, 0)

    def test_small_target_never_below_multiple(self):
        width, height = compute_blank_size("1:1 (Square)", 0.01, 64)
        self.assertGreaterEqual(width, 64)
        self.assertGreaterEqual(height, 64)

    def test_unknown_ratio_falls_back_to_square(self):
        width, height = compute_blank_size("not-a-ratio", 1.0, 8)
        self.assertEqual(width, height)


class TestCreateBlankImage(unittest.TestCase):
    def test_custom_mode_uses_given_size(self):
        image, = CreateBlankImage().create(False, 320, 240, "1:1 (Square)", 1.0, 8)
        self.assertEqual(tuple(image.shape), (1, 240, 320, 3))
        self.assertAlmostEqual(float(image.mean()), BLANK_GRAY, places=5)
        self.assertAlmostEqual(float(image.max()), BLANK_GRAY, places=5)
        self.assertAlmostEqual(float(image.min()), BLANK_GRAY, places=5)

    def test_auto_mode_uses_computed_size(self):
        image, = CreateBlankImage().create(True, 1, 1, "1:1 (Square)", 1.0, 8)
        self.assertEqual(tuple(image.shape), (1, 1000, 1000, 3))

    def test_auto_mode_portrait(self):
        image, = CreateBlankImage().create(True, 1, 1, "3:4 (Portrait Standard)", 2.0, 16)
        _batch, height, width, channels = image.shape
        self.assertEqual(channels, 3)
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)
        self.assertGreater(height, width)

    def test_output_is_float_rgb(self):
        image, = CreateBlankImage().create(False, 64, 64, "1:1 (Square)", 1.0, 8)
        self.assertEqual(image.dtype, __import__("torch").float32)
        self.assertEqual(tuple(image.shape), (1, 64, 64, 3))

    def test_custom_color_rgb(self):
        image, = CreateBlankImage().create(
            False, 8, 8, "1:1 (Square)", 1.0, 8,
            "RGB", "#000000", 255, 128, 0,
        )
        self.assertAlmostEqual(float(image[0, 0, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(image[0, 0, 0, 1]), 128 / 255.0, places=5)
        self.assertAlmostEqual(float(image[0, 0, 0, 2]), 0.0, places=5)

    def test_custom_color_hex(self):
        image, = CreateBlankImage().create(
            False, 8, 8, "1:1 (Square)", 1.0, 8, "十六进制 (Hex)", "#00ff00", 0, 0, 0,
        )
        self.assertEqual(tuple(image[0, 0, 0]), (0.0, 1.0, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
