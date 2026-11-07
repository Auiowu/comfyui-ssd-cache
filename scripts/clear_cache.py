"""独立清理脚本：不启动 ComfyUI 也能清空 SSD 缓存。

用法::

    python scripts/clear_cache.py                 # 清理配置里的目录
    python scripts/clear_cache.py D:\\other-dir    # 清理指定目录
    python scripts/clear_cache.py --dry-run       # 只统计不删除
"""

from __future__ import annotations

import argparse
import os
import sys


def _default_cache_dir() -> str:
    plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(plugin_dir, "config.json")
    if os.path.isfile(config_path):
        try:
            import json

            with open(config_path, "r", encoding="utf-8") as fh:
                return json.load(fh).get("cache_dir") or r"D:\ssd-cache"
        except Exception:  # noqa: BLE001
            pass
    return r"D:\ssd-cache"


def main() -> int:
    parser = argparse.ArgumentParser(description="清理 Comfyui-SSD-Cache 的磁盘缓存")
    parser.add_argument("cache_dir", nargs="?", default=None, help="缓存目录")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不删除")
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir or _default_cache_dir())
    if not os.path.isdir(cache_dir):
        print(f"目录不存在: {cache_dir}")
        return 1

    total = 0
    count = 0
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            if name.endswith((".pt", ".json")) or name.startswith(".tmp-"):
                path = os.path.join(root, name)
                try:
                    total += os.path.getsize(path)
                    count += 1
                except OSError:
                    pass

    human = f"{total / (1024 ** 3):.2f} GB" if total >= 1024 ** 3 else f"{total / (1024 ** 2):.2f} MB"
    print(f"目录: {cache_dir}")
    print(f"条目: {count} 个文件，合计 {human}")

    if args.dry_run:
        print("dry-run：未删除任何文件")
        return 0

    removed = 0
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            if name.endswith((".pt", ".json")) or name.startswith(".tmp-"):
                try:
                    os.unlink(os.path.join(root, name))
                    removed += 1
                except OSError as exc:
                    print(f"删除失败 {name}: {exc}", file=sys.stderr)

    print(f"已删除 {removed} 个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
