"""「创建空白图片」节点的实现。

独立于 ComfyUI 的 ``nodes`` 模块，便于单元测试，也避免与官方模块重名。
"""

from __future__ import annotations

import math

import torch

#: 常用宽高比（名称 -> (宽, 高)）
_ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Landscape Standard)": (4, 3),
    "2:3 (Portrait Classic)": (2, 3),
    "3:2 (Landscape Classic)": (3, 2),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Landscape Widescreen)": (16, 9),
    "5:8 (Portrait Tall)": (5, 8),
    "8:5 (Landscape Wide)": (8, 5),
    "1:2 (Tall)": (1, 2),
    "2:1 (Panorama)": (2, 1),
    "21:9 (Cinema)": (21, 9),
}

#: 中灰 #7f7f7f 对应的浮点值
BLANK_GRAY = 127 / 255.0

_DEFAULT_RGB = (127, 127, 127)

COLOR_MODES = ["十六进制 (Hex)", "RGB"]


def parse_hex_color(text: str) -> tuple:
    """把 ``#RRGGBB`` / ``#RGB`` / ``RRGGBB`` 解析成 ``(r, g, b)``，非法时回退中灰。"""
    raw = (text or "").strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        return _DEFAULT_RGB
    try:
        return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    except ValueError:
        return _DEFAULT_RGB


def resolve_color(mode: str, hex_color: str, red: int, green: int, blue: int) -> tuple:
    """按模式返回 ``(r, g, b)``（0-255）。"""
    if mode == "RGB":
        return (
            max(0, min(255, int(red))),
            max(0, min(255, int(green))),
            max(0, min(255, int(blue))),
        )
    return parse_hex_color(hex_color)


def compute_blank_size(aspect_ratio: str, megapixels: float, multiple: int) -> tuple:
    """按「宽高比 + 百万像素 + 边长倍数」算出 ``(width, height)``。"""
    a, b = _ASPECT_RATIOS.get(aspect_ratio, (1, 1))
    step = max(1, int(multiple))
    target = max(0.01, float(megapixels)) * 1_000_000.0
    width = math.sqrt(target * a / b)
    height = math.sqrt(target * b / a)
    width = max(step, int(round(width / step)) * step)
    height = max(step, int(round(height / step)) * step)
    return width, height


class CreateBlankImage:
    """生成指定分辨率的中灰（#7f7f7f）空白图片。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "自动",
                        "label_off": "自定义",
                        "tooltip": "关闭：直接用下面的宽/高；打开：按宽高比 / 百万像素 / 倍数自动计算",
                    },
                ),
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "aspect_ratio": (
                    list(_ASPECT_RATIOS.keys()),
                    {"default": "1:1 (Square)"},
                ),
                "megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 64.0, "step": 0.1},
                ),
                "multiple": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "color_mode": (
                    COLOR_MODES,
                    {"default": COLOR_MODES[0], "tooltip": "用十六进制还是 RGB 指定颜色"},
                ),
                "hex_color": (
                    "STRING",
                    {"default": "#7f7f7f", "multiline": False, "tooltip": "十六进制颜色，如 #7f7f7f"},
                ),
                "red": ("INT", {"default": 127, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 127, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 127, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create"
    CATEGORY = "SSD-Cache"
    DESCRIPTION = "生成指定分辨率与颜色的空白图片；支持自定义宽高，或按宽高比 / 百万像素 / 倍数自动计算。"

    def create(
        self,
        auto_mode,
        width,
        height,
        aspect_ratio,
        megapixels,
        multiple,
        color_mode="十六进制 (Hex)",
        hex_color="#7f7f7f",
        red=127,
        green=127,
        blue=127,
    ):
        if auto_mode:
            width, height = compute_blank_size(aspect_ratio, megapixels, multiple)
        width = max(1, int(width))
        height = max(1, int(height))

        r, g, b = resolve_color(color_mode, hex_color, red, green, blue)
        color = torch.tensor([r, g, b], dtype=torch.float32) / 255.0
        image = color.view(1, 1, 1, 3).expand(1, height, width, 3).contiguous()
        return (image,)
