# Comfyui-SSD-Cache

把 ComfyUI 节点的运行结果缓存到 **SSD**，弥补内存缓存不够用、以及工作流跑一半崩掉后
必须从头重算的问题。

同一工作流重复运行、或 ComfyUI 崩溃重启后，已完成节点的结果直接从磁盘恢复，
不需要重新采样。

---

## 工作原理

ComfyUI 0.44 起内置了官方的外部缓存插件接口（`comfy_execution.cache_provider.CacheProvider`）：

- 每个节点执行前，ComfyUI 先用「节点类型 + 全部输入 + 全部上游签名」算出一个确定性的
  `cache_key_hash`（SHA256），先去内存缓存找；
- 内存未命中时，会依次询问已注册的 `CacheProvider`；
- 节点执行完写入内存缓存后，也会异步通知 provider。

本插件注册了一个 `CacheProvider`，把可序列化的节点输出写到 SSD：

- **缓存键**：ComfyUI 自己的 `cache_key_hash`，跨进程 / 跨重启稳定；
- **只缓存张量、字典、列表和标量**（LATENT、IMAGE、MASK、CONDITIONING、SIGMAS…）；
- **MODEL / CLIP / VAE 等业务对象自动跳过**，绝不会把模型权重拖进缓存；
- **输出节点（SaveImage / PreviewImage）不缓存**，避免恢复时跳过写文件、生成预览等副作用；
- 写入采用「临时文件 + 原子替换」，进程中途崩溃不会留下可被误读的半截文件；
- 读取使用 `torch.load(weights_only=True)`，即使缓存目录被投毒也无法执行任意代码；
- 写盘跑在**独立的进程级线程池**里（ComfyUI 每个 prompt 会用 `asyncio.run()` 新建并关闭
  事件循环，放在循环里的异步任务会被取消，因此不能依赖它）。

缓存目录结构（按 hash 前 4 位分两级，避免单目录文件过多）：

```
D:\ssd-cache\
  ab\
    cd\
      abcdef0123...（64 位 hash）.pt      # 张量数据
      abcdef0123....json                 # 元数据（节点类型、大小、时间）
```

> **缓存不会自动清理。** 这正符合「跑一半崩掉、重启后接着用」的需求；
> 需要清理时请用下面的手动方式。

---

## 安装

推荐用目录联接（junction），这样源码目录和 ComfyUI 加载的是同一份文件：

```bat
mklink /J "D:\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI\custom_nodes\Comfyui-SSD-Cache" "E:\Comfyui-SSD-Cache"
```

或者直接把整个目录复制到 `ComfyUI/custom_nodes/Comfyui-SSD-Cache`。

装好后**重启 ComfyUI**。启动日志里应出现：

```
SSD-Cache provider 已注册（初始状态: 禁用，目录: D:\ssd-cache）
SSD-Cache: HTTP API 已注册
```

---

## 节点

### 1. SSD Cache Enable（UNet 后）

放在 **UNet / Checkpoint 加载节点之后**，把 MODEL 透传出去。

| 参数 | 说明 |
| --- | --- |
| `cache_dir` | 缓存目录，默认 `D:\ssd-cache`，不存在会自动创建 |
| `enable` | 是否启用，默认开 |
| `max_entry_mb` | 单条缓存大小上限（MB），`0` = 不限 |

```
Load Checkpoint ──MODEL──▶ SSD Cache Enable ──MODEL──▶ KSampler ──▶ …
```

启用状态和目录会写入插件目录下的 `config.json`，因此 ComfyUI 重启后、在这个节点
执行之前，命中查询就已经生效（ComfyUI 在 prompt 开始时会批量预取缓存）。

### 2. SSD Cache Stats（命中率）

可选。**接在工作流末端任意一个节点的输出上**（例如 VAE Decode 的 `IMAGE`），
它会等上游全部执行完才运行，显示的是本轮最终命中率：

```
SSD-Cache: 已启用
本轮命中率: 10/17 (58.8%) = SSD 恢复 4 + 内存/显存 6 · 实际计算 7
累计命中率: 30/52 (57.7%) · SSD 恢复 12 · 实际计算 22
本轮 SSD: 4/10 命中 (40.0%) · 写入 3（1 排队）· 不可缓存 1
累计 SSD: 12/20 命中 (60.0%) · 写入 9
磁盘: 1.23 GB · 42 条 | 目录: D:\ssd-cache
```

几个容易误解的地方：

- **「本轮命中率」把内存/显存里已有的缓存也算进去了**。ComfyUI 自己命中内存的节点不会
  来询问插件，但通过 `execution_cached` 事件仍然能统计到，所以：
  `命中率 = (SSD 恢复 + 内存/显存已有) / 本轮节点总数`。
- **下面的「本轮 SSD」是只算 SSD 层的**：它只统计「内存未命中后去问 SSD」的节点，
  所以数字会更小，用来衡量磁盘缓存本身的命中情况。
- **验证 SSD 命中必须重启 ComfyUI**。同一进程内第二次运行基本都是内存命中（`SSD 0/0`），
  重启后才会看到 `SSD 恢复 N`。
- 若看到「写入 3（1 排队）」，表示还有 1 个条目已提交、正在后台线程写盘；
  写完后数字会自动归到「写入」。
- 出现 `⚠ 键不稳定（同一节点每次键不同）` 时，说明该节点或其上游每次运行的输入签名都变了，
  缓存必然无法命中。常见原因是随机种子（`RandomNoise` / `KSampler` 的
  `control_after_generate` 设成了随机）或时间戳类节点。

它同时输出一个 `STRING`，可以接到「Show Text」之类的节点。

### 3. SSD Cache Clear（手动清理）

可选。默认 `confirm = false`，只报告占用大小；勾选 `confirm` 后执行才会真正删除。

---

## 手动清理

三种方式任选：

1. **节点**：`SSD Cache Clear`，勾上 `confirm` 后运行一次；
2. **脚本**（不用启动 ComfyUI）：

   ```bat
   D:\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI\.venv\Scripts\python.exe E:\Comfyui-SSD-Cache\scripts\clear_cache.py
   D:\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI\.venv\Scripts\python.exe E:\Comfyui-SSD-Cache\scripts\clear_cache.py --dry-run
   ```

3. **HTTP API**（ComfyUI 运行时；0.44 起路由带 `/api` 前缀）：

   ```
   GET  http://127.0.0.1:8188/api/ssd_cache/stats
   POST http://127.0.0.1:8188/api/ssd_cache/clear?confirm=1
   POST http://127.0.0.1:8188/api/ssd_cache/configure
   ```

4. 直接删除 `D:\ssd-cache` 整个目录也可以（下次运行会自动重建）。

---

## 配置

插件目录下的 `config.json`（由 Enable 节点自动维护）：

```json
{
  "cache_dir": "D:\\ssd-cache",
  "enabled": true,
  "max_entry_mb": 0
}
```

- `enabled` 持久化后，即使工作流里暂时没放 Enable 节点，缓存也会继续生效；
- 想彻底关掉，把 Enable 节点的 `enable` 取消勾选并运行一次，或手动改这里为 `false` 后重启。

---

## 注意事项与限制

- **改了节点代码或升级了自定义节点，缓存不会自动失效。** 缓存键只反映「输入参数 + 图结构」，
  不反映代码实现。升级插件后建议手动清理一次。
- **只有可序列化的输出会被缓存。** MODEL / CLIP / VAE / GUIDER / SAMPLER 等对象节点会自动跳过，
  它们本来也很快（加载模型除外）。
- **输出节点不缓存**（SaveImage / PreviewImage），每次都会重新写文件，这是刻意为之。
- **缓存键依赖上游节点的 `IS_CHANGED`。** 如果某个上游节点每次运行都返回随机值，
  那么它下游的所有节点都不会命中——这是 ComfyUI 缓存机制的固有行为，与本插件无关。
  Stats 节点会列出「键不稳定」的节点，帮助定位是哪一个在变。
- **随机种子是最常见的「永远 0 命中」原因。** `RandomNoise`、`KSampler` 的
  `control_after_generate` 默认可能是随机，每跑一次 seed 就变一次，整条下游链的键都会变。
  把 seed 固定（右键节点改 `control_after_generate` 为 fixed，或手填固定值）后再验证。
- **磁盘会持续增长。** 这是设计如此（不自动清理）；建议偶尔用 `SSD Cache Clear` 或脚本清理。
- **强制结束进程的瞬间**可能丢失最后 1～2 个正在后台写盘的节点（原子写保证不会产生坏文件，
  下次运行重算它们即可）。
- 单个工作流的缓存目录可以不同（Enable 节点的 `cache_dir`），但同一时刻只有一个全局 provider，
  **多个 Enable 节点时后执行的那个生效**，建议一个工作流只放一个。

---

## 开发与测试

```bat
D:\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI\.venv\Scripts\python.exe E:\Comfyui-SSD-Cache\tests\test_store.py
D:\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI\.venv\Scripts\python.exe E:\Comfyui-SSD-Cache\tests\test_provider.py
```

测试只使用标准库 `unittest`，不会往 ComfyUI 环境里装任何依赖。

目录结构：

```
Comfyui-SSD-Cache/
  __init__.py            # ComfyUI 入口：注册 provider、HTTP 路由、节点映射
  nodes.py               # 三个节点
  ssd_cache/
    config.py            # config.json 读写
    store.py             # 磁盘存储：序列化、原子写、统计、清理
    provider.py          # CacheProvider 实现 + 命中统计
    routes.py            # HTTP API
  scripts/clear_cache.py # 独立清理脚本
  tests/                 # unittest 测试
```
