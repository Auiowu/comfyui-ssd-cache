"""Comfyui-SSD-Cache

把 ComfyUI 节点的运行结果镜像到 SSD，弥补内存缓存的不足：
同一工作流重复运行时直接复用磁盘结果，进程崩溃重启后也不必重算已完成的部分。

安装：把本目录链接或复制到 ``ComfyUI/custom_nodes/Comfyui-SSD-Cache``。
"""

from __future__ import annotations

import logging

log = logging.getLogger("SSD-Cache")

# 注册 CacheProvider（必须在 ComfyUI 执行工作流之前完成）
_provider = None
try:
    from .ssd_cache.provider import get_provider

    _provider = get_provider()
except Exception as exc:  # noqa: BLE001 - 不允许插件加载失败拖垮 ComfyUI
    log.exception("SSD-Cache: 初始化失败: %s", exc)

# 接管 ComfyUI 的缓存层（只 patch 运行时对象，不改官方源码）：
# 必须在 PromptExecutor 创建之前完成，custom node 加载阶段正好满足。
if _provider is not None:
    try:
        from .ssd_cache.memory_cache import install_cache_override, install_prompt_scope_patch

        install_cache_override(_provider)
        # 让「是否启用」按工作流作用域判定：没放 Enable 节点的工作流完全不缓存
        install_prompt_scope_patch(_provider)
    except Exception as exc:  # noqa: BLE001
        log.warning("SSD-Cache: 缓存层接管失败，沿用 ComfyUI 原实现: %s", exc)

# HTTP API（PromptServer 未就绪时静默跳过）
try:
    from .ssd_cache.routes import register_routes

    register_routes()
except Exception as exc:  # noqa: BLE001
    log.warning("SSD-Cache: HTTP API 注册失败: %s", exc)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
