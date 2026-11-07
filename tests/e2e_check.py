"""端到端检查脚本：通过 HTTP API 验证节点注册与 SSD 缓存命中。

用法::

    python tests/e2e_check.py [--port 8189] [--rounds 2]

它会在指定端口的 ComfyUI 上：
1. 确认三个节点已注册；
2. 启用 SSD 缓存；
3. 提交一个极简工作流（EmptyLatentImage -> SSDCacheStats）；
4. 再提交一次，检查第二次是否从 SSD 命中。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# 本机可能配置了系统代理，localhost 请求必须绕过代理
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.ProxyHandler({}))
)


def _request(base: str, path: str, payload=None, method=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
        return json.loads(body) if body else {}


def _wait_prompt(base: str, prompt_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = _request(base, f"/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(0.3)
    raise TimeoutError(f"prompt {prompt_id} 未在 {timeout}s 内完成")


WORKFLOW = {
    "1": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 64, "height": 64, "batch_size": 1},
    },
    "2": {
        "class_type": "SSDCacheStats",
        "inputs": {"anything": ["1", 0]},
    },
}


def _detect_prefix(base: str) -> str:
    """ComfyUI 0.44 起所有路由带 /api 前缀；老版本没有。自动探测。"""
    for prefix in ("/api", ""):
        try:
            _request(base, prefix + "/object_info")
            return prefix
        except urllib.error.HTTPError:
            continue
    raise RuntimeError("无法访问 /object_info，ComfyUI 是否在运行？")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8189)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="测试用缓存目录（默认沿用当前配置，会污染真实缓存时可指定临时目录）",
    )
    parser.add_argument(
        "--expect",
        choices=("auto", "write", "hit"),
        default="auto",
        help="auto=首轮写入次轮命中；write=本轮应写入；hit=本轮应命中（重启后验证用）",
    )
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    prefix = _detect_prefix(base)
    base += prefix
    print(f"[0] API 前缀 = {prefix or '(无)'}")

    print(f"[1] 检查节点注册（{base}）")
    info = _request(base, "/object_info")
    for name in ("SSDCacheEnable", "SSDCacheStats", "SSDCacheClear"):
        if name not in info:
            print(f"    FAIL: 未找到节点 {name}")
            return 1
        print(f"    OK: {name}")

    print("[2] 初始统计")
    stats = _request(base, "/ssd_cache/stats")
    print(f"    enabled={stats['enabled']} dir={stats['cache_dir']} entries={stats['disk']['entries']}")

    print("[3] 启用 SSD 缓存")
    target_dir = args.cache_dir or stats["cache_dir"]
    result = _request(
        base,
        "/ssd_cache/configure",
        {"cache_dir": target_dir, "enabled": True, "max_entry_mb": 0},
    )
    print(f"    {result}")
    if not result.get("ok"):
        return 1

    for round_index in range(1, args.rounds + 1):
        print(f"[4.{round_index}] 提交工作流（第 {round_index} 次）")
        submitted = _request(base, "/prompt", {"prompt": WORKFLOW, "client_id": "ssd-cache-e2e"})
        prompt_id = submitted["prompt_id"]
        history_entry = _wait_prompt(base, prompt_id)
        status = history_entry.get("status", {})
        print(f"    prompt_id={prompt_id[:8]} status={status.get('status_str')}")

        stats = _request(base, "/ssd_cache/stats")
        session = stats["session"]
        written = session["stores"] + session.get("stores_pending", 0)
        print(
            f"    节点: 跳过 {session.get('cached_nodes', 0)} "
            f"· 实际计算 {session.get('executed_nodes', 0)}"
        )
        print(
            f"    SSD: 命中 {session['hits']}/{session['lookups']} "
            f"写入 {written}（排队 {session.get('stores_pending', 0)}）"
            f" 不可缓存 {session['skipped_unsupported']} "
            f"磁盘 {stats['disk']['bytes_human']} / {stats['disk']['entries']} 条"
        )

        if args.expect == "write" and round_index == 1 and written < 1:
            print("    FAIL: 这一轮没有写入任何缓存")
            return 1
        if args.expect == "hit" and round_index == 1 and session["hits"] < 1:
            print("    FAIL: 这一轮没有命中 SSD 缓存")
            return 1
        if args.expect == "auto":
            if round_index == 1 and written < 1:
                print("    FAIL: 第一轮没有写入任何缓存")
                return 1
            # 同一进程的第二轮通常由内存缓存直接命中，lookups 会是 0；
            # 只有真的查了 SSD（lookups>0）才要求命中。
            if round_index >= 2 and session["lookups"] > 0 and session["hits"] < 1:
                print("    FAIL: 第二轮查了 SSD 却没有命中")
                return 1

    print("[5] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
