"""SSD-Cache 的持久化配置。

配置文件 config.json 保存在插件根目录，用于让 provider 在 ComfyUI 重启后
（此时 Enable 节点还没执行）就能恢复上一次的缓存目录与开关状态。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

log = logging.getLogger("SSD-Cache")

#: 默认缓存目录（D 盘根目录）
DEFAULT_CACHE_DIR = r"D:\ssd-cache"

_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(_PLUGIN_DIR, "config.json")


@dataclass
class SSDCacheConfig:
    cache_dir: str = DEFAULT_CACHE_DIR
    enabled: bool = False
    max_entry_mb: int = 0

    def to_dict(self) -> dict:
        return {
            "cache_dir": self.cache_dir,
            "enabled": self.enabled,
            "max_entry_mb": self.max_entry_mb,
        }

    @classmethod
    def from_dict(cls, data: object) -> "SSDCacheConfig":
        cfg = cls()
        if not isinstance(data, dict):
            return cfg

        cache_dir = data.get("cache_dir")
        if isinstance(cache_dir, str) and cache_dir.strip():
            cfg.cache_dir = cache_dir.strip()

        enabled = data.get("enabled")
        if isinstance(enabled, bool):
            cfg.enabled = enabled

        max_entry_mb = data.get("max_entry_mb")
        if isinstance(max_entry_mb, (int, float)) and max_entry_mb >= 0:
            cfg.max_entry_mb = int(max_entry_mb)

        return cfg


def load_config() -> SSDCacheConfig:
    """读取配置；文件不存在或损坏时回退到默认值。"""
    if not os.path.isfile(CONFIG_PATH):
        return SSDCacheConfig()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            return SSDCacheConfig.from_dict(json.load(fh))
    except Exception as exc:  # noqa: BLE001 - 配置损坏不应阻止 ComfyUI 启动
        log.warning("SSD-Cache: 无法读取配置文件 %s: %s", CONFIG_PATH, exc)
        return SSDCacheConfig()


def save_config(cfg: SSDCacheConfig) -> None:
    """原子写入配置，避免进程崩溃时留下半截 JSON。"""
    tmp_path = CONFIG_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(cfg.to_dict(), fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, CONFIG_PATH)
    except Exception as exc:  # noqa: BLE001
        log.warning("SSD-Cache: 无法写入配置文件 %s: %s", CONFIG_PATH, exc)
