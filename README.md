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
  事件循环，放在循环里的异步任务会被取消，因此不能依赖它）；
- 插件在加载时会**接管 ComfyUI 的缓存层**（运行时 patch `execution.CacheSet`，不修改任何
  官方源文件）：节点结果先进内存/显存，超过你设定的上限后按 LRU 淘汰；无论是否进内存，
  都会异步写入 SSD。读取顺序固定为 **内存/显存 → SSD → 真正计算**；
- **按工作流生效**：每次运行开始时先检查图里有没有 `SSD Cache Enable` 节点，
  没有就整条工作流使用 **ComfyUI 官方默认缓存**（不读写 SSD）；有才切到 SSD 缓存层。

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

### 1. SSD Cache Enable（可放任意位置）

**接任意节点的输出、原样透传**，可以放在工作流任意位置——不限于 UNet 之后。

| 参数 | 说明 |
| --- | --- |
| `value` | 接任意类型的数据（`*`），原样透传。可接 MODEL、LATENT、IMAGE、CONDITIONING… |
| `cache_dir` | 缓存目录，默认 `D:\ssd-cache`，不存在会自动创建 |
| `enable` | 是否启用，默认开 |
| `max_entry_mb` | 单条缓存大小上限（MB），`0` = 不限 |
| `memory_cache` | 是否允许节点结果留在内存/显存。关掉后结果只写 SSD，内存占用为 0（每次运行都读盘） |
| `memory_cache_mb` | 内存/显存缓存上限（MB），`0` = 不限。超出后按最久未使用淘汰，被淘汰的结果仍可从 SSD 取回 |

> `memory_cache` 控制的是插件接管后的缓存层。ComfyUI 原版的内存缓存行为被替换成：
> **先内存 → 再 SSD → 都没有才真正计算**，所以关掉内存缓存不会导致重复计算，
> 只是每次命中都要从磁盘读一次。

```
Load Checkpoint ──MODEL──▶ SSD Cache Enable ──MODEL──▶ KSampler ──▶ …
EmptyLatentImage ─LATENT─▶ SSD Cache Enable ─LATENT─▶ KSampler ──▶ …
```

> **没放这个节点的工作流，使用 ComfyUI 官方默认缓存机制**（`--cache-type` 决定的
> RAMPressure / LRU / Classic 原样工作），不会读写 SSD。
> 放了才整条工作流切到 SSD 缓存层。

启用状态和目录会写入插件目录下的 `config.json`；每次运行开始时插件会先读取本工作流里
Enable 节点的参数再决定缓存层，所以预取阶段就能按正确的开关和目录工作。

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
内存/显存: 512.00 MB / 2.00 GB · 18 条
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

### 4. 创建空白图片

生成指定分辨率的中灰（`#7f7f7f`）空白图片，输出 `IMAGE`。

| 参数 | 说明 |
| --- | --- |
| `auto_mode` | **关闭（自定义）**：使用下面的 `width` / `height`<br>**打开（自动）**：按 `aspect_ratio` + `megapixels` + `multiple` 自动计算 |
| `width` / `height` | 自定义模式的宽高 |
| `aspect_ratio` | 自动模式的宽高比：1:1、3:4、4:3、2:3、3:2、9:16、16:9、5:8、8:5、1:2、2:1、21:9 |
| `megapixels` | 自动模式的目标像素数（百万），如 0.4、1、2、4 |
| `multiple` | 自动模式下宽高对齐到的边长倍数 |
| `color_mode` | `十六进制 (Hex)` 或 `RGB`，决定用哪组颜色参数 |
| `hex_color` | 十六进制颜色，如 `#ff0000`；也支持 `#RGB` 简写，非法值回退中灰 |
| `red` / `green` / `blue` | RGB 分量（0-255），`color_mode = RGB` 时生效 |

默认是 `#7f7f7f` 中灰。节点上还带一个**原生颜色选择器**，选色后会自动写入 `hex_color`，
也可以直接在文本框里手改十六进制。

计算方式：按宽高比解出满足 `w * h = megapixels * 1e6` 的宽高，再四舍五入到 `multiple` 的整数倍。

例如 `3:4 (Portrait Standard)` + `2.0` MP + 倍数 `16` → **1232 × 1632**（约 2.01 MP）。

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
  "max_entry_mb": 0,
  "memory_cache": true,
  "memory_cache_mb": 2048
}
```

- **是否启用由每个工作流的图里是否包含 Enable 节点决定**：没放 `SSD Cache Enable` 的工作流
  使用 **ComfyUI 官方默认缓存机制**，不查询 SSD 也不写盘，即使这里 `enabled` 是 `true`。
  （`enabled` 只在节点输入是链接、无法静态求值时作为回退值。）
- 想彻底关掉，把 Enable 节点的 `enable` 取消勾选并运行一次，或手动改这里为 `false` 后重启。
- `memory_cache_mb` 是**插件接管缓存层后的唯一内存上限**：设置后 ComfyUI 自带的
  `--cache-ram` / RAM 压力淘汰不再影响节点缓存（`--cache-ram` 只影响模型权重相关逻辑）。

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
- **关闭 `memory_cache` 后每次命中都要读盘。** 单个大张量（几百 MB）会有可感知的 IO 延迟，
  内存够用时不建议关闭；内存紧张时建议开启并设置一个 `memory_cache_mb` 上限，而不是完全关闭。
- 同一工作流里有多个 Enable 节点时，作用域判定取第一个，运行时配置以最后执行的那个为准；
  建议一个工作流只放一个。

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
