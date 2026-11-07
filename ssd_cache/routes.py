"""HTTP API：方便脚本或前端查看统计、手动清理。

* ``GET  /ssd_cache/stats``        返回 JSON 统计
* ``POST /ssd_cache/clear``        清空缓存（需要 ``?confirm=1``）
* ``POST /ssd_cache/configure``    切换目录 / 开关（可选）
"""

from __future__ import annotations

import logging

from aiohttp import web

log = logging.getLogger("SSD-Cache")

_registered = False


def _provider():
    from .provider import get_provider

    return get_provider()


def register_routes() -> bool:
    """把路由挂到 ComfyUI 的 aiohttp 应用上。"""
    global _registered
    if _registered:
        return True

    try:
        from server import PromptServer

        server = getattr(PromptServer, "instance", None)
        if server is None or getattr(server, "routes", None) is None:
            log.warning("SSD-Cache: PromptServer 尚未就绪，跳过 HTTP 路由注册")
            return False
        routes = server.routes
    except Exception as exc:  # noqa: BLE001
        log.warning("SSD-Cache: 无法注册 HTTP 路由: %s", exc)
        return False

    @routes.get("/ssd_cache/stats")
    async def _stats(_request):
        return web.json_response(_provider().stats_snapshot())

    @routes.post("/ssd_cache/clear")
    async def _clear(request):
        confirm = request.query.get("confirm", "")
        if confirm not in ("1", "true", "yes"):
            return web.json_response(
                {"ok": False, "error": "需要 ?confirm=1 才会真正清理"}, status=400
            )
        provider = _provider()
        removed = provider.store.clear()
        provider.store.invalidate_size_cache()
        return web.json_response({"ok": True, "removed": removed})

    @routes.post("/ssd_cache/configure")
    async def _configure(request):
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = {}
        provider = _provider()
        cache_dir = payload.get("cache_dir") or provider.store.cache_dir
        enabled = bool(payload.get("enabled", provider.enabled))
        max_entry_mb = int(payload.get("max_entry_mb", provider.config.max_entry_mb) or 0)
        ok = provider.configure(cache_dir, enabled, max_entry_mb)
        return web.json_response(
            {
                "ok": ok,
                "enabled": provider.enabled,
                "cache_dir": provider.store.cache_dir,
                "max_entry_mb": provider.config.max_entry_mb,
            }
        )

    _registered = True
    log.info("SSD-Cache: HTTP API 已注册")
    return True
