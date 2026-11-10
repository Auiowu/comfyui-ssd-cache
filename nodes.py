"""ComfyUI 节点定义。

三个节点：

* ``SSDCacheEnable``  —— 放在 UNet 加载节点之后，透传 MODEL，同时启用 SSD 缓存。
* ``SSDCacheStats``   —— 可选，放在工作流末端显示缓存命中率。
* ``SSDCacheClear``   —— 可选，手动清理缓存目录（默认不执行，必须勾选 confirm）。
"""

from __future__ import annotations

import logging
import time

from .ssd_cache.config import DEFAULT_CACHE_DIR
from .ssd_cache.provider import get_provider
from .ssd_cache.store import format_bytes

log = logging.getLogger("SSD-Cache")


class SSDCacheEnable:
    """启用 SSD 缓存：把节点输出镜像到磁盘，重启后可直接复用。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "*",
                    {
                        "tooltip": "接任意节点的输出，原样透传；可以放在工作流任意位置",
                    },
                ),
                "cache_dir": (
                    "STRING",
                    {
                        "default": DEFAULT_CACHE_DIR,
                        "multiline": False,
                        "tooltip": "SSD 缓存目录，不存在会自动创建",
                    },
                ),
                "enable": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"}),
                "max_entry_mb": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 102400,
                        "step": 64,
                        "tooltip": "单条缓存的大小上限（MB），0 表示不限制",
                    },
                ),
                "memory_cache": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "启用",
                        "label_off": "禁用",
                        "tooltip": "关闭后节点结果不留在内存/显存，只写 SSD（省内存但每次都要读盘）",
                    },
                ),
                "memory_cache_mb": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 102400,
                        "step": 256,
                        "tooltip": "内存/显存缓存上限（MB）。0 表示不限制；超出后按最久未使用淘汰，结果仍可从 SSD 取回",
                    },
                ),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "apply"
    CATEGORY = "SSD-Cache"
    DESCRIPTION = "接任意数据并原样透传，用于启用本工作流的 SSD 缓存；可放在工作流任意位置。"

    def apply(self, value, cache_dir, enable, max_entry_mb, memory_cache=True, memory_cache_mb=0):
        provider = get_provider()
        ok = provider.configure(
            cache_dir, enable, max_entry_mb, memory_cache=memory_cache, memory_cache_mb=memory_cache_mb
        )
        if not ok:
            log.error("SSD-Cache: 启用失败，缓存目录不可用: %s", cache_dir)
        return (value,)


class SSDCacheStats:
    """显示当前工作流的 SSD 缓存命中率。

    把它接在工作流末端任意输出上（例如 VAE Decode 的 IMAGE），它会在所有上游
    节点执行完之后才运行，显示的就是本轮最终命中率。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "anything": (
                    "*",
                    {
                        "tooltip": "接任意一个末端节点的输出，用于保证本节点最后执行；可以不接",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("stats",)
    FUNCTION = "report"
    OUTPUT_NODE = True
    CATEGORY = "SSD-Cache"
    DESCRIPTION = "显示 SSD 缓存命中率；接在末端时显示最终结果。"

    def report(self, anything=None):
        text = get_provider().format_stats()
        return {"ui": {"text": [text]}, "result": (text,)}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 必须每次执行；返回时间戳而不是 NaN，避免把 NaN 传染给下游节点的缓存键。
        return time.time()


class SSDCacheClear:
    """手动清理 SSD 缓存。默认只统计不删除，必须把 confirm 勾上才会真正清理。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_dir": (
                    "STRING",
                    {"default": DEFAULT_CACHE_DIR, "multiline": False},
                ),
                "confirm": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "确认删除",
                        "label_off": "仅统计",
                        "tooltip": "勾选后执行本节点会删除该目录下所有缓存",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "clear"
    OUTPUT_NODE = True
    CATEGORY = "SSD-Cache"
    DESCRIPTION = "手动清理 SSD 缓存目录；confirm 未勾选时只统计大小。"

    def clear(self, cache_dir, confirm):
        from .ssd_cache.store import SSDCacheStore

        try:
            store = SSDCacheStore(cache_dir)
        except OSError as exc:
            text = f"无法访问缓存目录 {cache_dir}: {exc}"
            return {"ui": {"text": [text]}, "result": (text,)}

        size, count = store.disk_usage(max_age=0.0)
        if not confirm:
            text = (
                f"未执行清理（confirm 未勾选）\n"
                f"目录: {store.cache_dir}\n"
                f"当前占用: {format_bytes(size)} · {count} 个条目"
            )
            return {"ui": {"text": [text]}, "result": (text,)}

        removed = store.clear()
        text = f"已清理 {removed} 个文件，释放约 {format_bytes(size)}\n目录: {store.cache_dir}"
        log.info("SSD-Cache: %s", text.replace("\n", " "))
        return {"ui": {"text": [text]}, "result": (text,)}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()


NODE_CLASS_MAPPINGS = {
    "SSDCacheEnable": SSDCacheEnable,
    "SSDCacheStats": SSDCacheStats,
    "SSDCacheClear": SSDCacheClear,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SSDCacheEnable": "SSD Cache Enable（UNet 后）",
    "SSDCacheStats": "SSD Cache Stats（命中率）",
    "SSDCacheClear": "SSD Cache Clear（手动清理）",
}
