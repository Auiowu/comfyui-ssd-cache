"""pytest 配置：把 ComfyUI 根目录加入 sys.path，便于导入 comfy_execution。"""

from __future__ import annotations

import os
import sys

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

_COMFY_ROOT = os.environ.get(
    "COMFYUI_ROOT", r"D:\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
)
if os.path.isdir(_COMFY_ROOT) and _COMFY_ROOT not in sys.path:
    sys.path.insert(0, _COMFY_ROOT)
