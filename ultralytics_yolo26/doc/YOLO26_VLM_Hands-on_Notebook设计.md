# YOLO26 + VLM Hands-on Notebook 设计

> 目标文件：`ultralytics_yolo26x_hands_on.ipynb`
>
> 定位：参与者主导的参数决策、A/B 实验与联动思考题
>
> 建议占用：60 分钟 Workshop 中的 20 分钟

> 状态：已实现并在 W7900D 上从随机目录 clean-kernel 执行，8/8 code cells，0 error

## 1. 目标与边界

Hands-on 不重复讲 PT 导出、I/O Binding 或 direct encode，也不让参与者现场实现 tracker。它回答：

> 在 YOLO 高频检测与 VLM 低频理解之间，如何选择调用频率、ROI 数量、资源位置和背压策略？

参与者必须先做决策，再运行 A/B，最后解释结果：

```text
我选择了什么
为什么这样选
它改善了哪个指标
牺牲了什么
下一步如何从定时轮询升级为事件联动
```

## 2. 前置条件

- Step-by-step 已完成，或参与者已有对应概念。
- 使用 release ONNX 和 prepared MIGraphX cache，不重新导出或 cold compile。
- llama.cpp companion 已启动并通过真实 multimodal completion。
- 输入固定为同一条短视频，保证结果可比较。
- 可运行基线为 JPEG/Base64/HTTP；不把 HIP-IPC 当作当前 release 能力。

## 3. 决策层级

### 3.1 可直接运行的决策

| 决策 | 选项 | 推荐基线 | 观察指标 |
|---|---|---|---|
| VLM 间隔 | 15 / 30 / 60 frames | 30 | refresh、FPS、busy skip |
| 处理长度 | 120 / 240 / all frames | 120 | 样本量与课堂时间 |
| GPU 布局 | single / split | single | contention 与资源成本 |
| 是否运行 VLM | off / async ROI | async ROI | YOLO-only 与并行差异 |

### 3.2 参数化后才能运行的决策

| 决策 | 选项 | 当前缺口 |
|---|---|---|
| ROI 数量 | 1 / 3 | 已通过 `--vlm-top-k` 支持 |
| Busy policy | skip / latest-only | 当前只实现 skip |
| Transport | JPEG HTTP / HIP IPC | release companion 只保证标准 HTTP |

Notebook 必须将选项标记为 `available now` 或 `design only`，不能让参与者选择不会生效的参数。

## 4. 默认决策卡

```python
DECISION = {
    "vlm_interval_frames": 30,
    "vlm_top_k": 3,
    "busy_policy": "skip",
    "transport": "jpeg_http",
    "gpu_placement": "single_gpu",
    "max_frames": 120,
}
```

固定事实：

- 25 FPS 下，30 帧对应 1.2 秒视频时间；
- 第 1 帧和以后每 30 帧且存在 detection 时产生触发机会；
- 一批最多 3 个 ROI；
- 3 个 ROI 由客户端并发 HTTP 请求；
- llama.cpp 使用 `--parallel 3` 提供 3 个 slot；
- 应用侧同一时间只允许一批 VLM work；
- 上一批未完成时，当前实现跳过新触发。

`--parallel 3` 是服务容量，不是调用周期，也不表示每次一定存在 3 个 ROI。

## 5. Notebook Cell 设计

目标为 18 个 cell，其中 8 个 code cell。

| Cell | 类型 | 内容 | 参与者动作 |
|---:|---|---|---|
| 1 | Markdown | 任务、约束和提交物 | 阅读目标 |
| 2 | Markdown | 环境 gate | 理解固定环境 |
| 3 | Python | YOLO、VLM、模型、GPU 检查 | 确认全部 PASS |
| 4 | Markdown | 输入视频和检测对象 | 预测哪些 ROI 会被选择 |
| 5 | Python | 播放输入与 detection frame | 记录观察 |
| 6 | Markdown | YOLO-only baseline | 先预测 FPS |
| 7 | Python | 运行或复用 YOLO-only | 记录 baseline |
| 8 | Markdown | VLM-only baseline | 选择一个 ROI 和 prompt |
| 9 | Python | 单独运行一次 top-1 ROI | 记录真实 latency/output |
| 10 | Markdown | 决策点 | 选择 interval、长度、GPU 布局 |
| 11 | Python | 填写并校验 `DECISION` | 生成实验命令 |
| 12 | Markdown | 并行实验 | 预测 contention 与 skip |
| 13 | Python | 运行 YOLO + async VLM | 生成视频与日志 |
| 14 | Markdown | A/B 结果 | 解释而非只找最大 FPS |
| 15 | Python | 展示速度账本 | 对比 YOLO/VLM 指标 |
| 16 | Markdown | 联动思考题 | 设计 Trigger/Evidence/Return/Failure |
| 17 | Python | 生成小组答卷 JSON/Markdown | 保存方案 |
| 18 | Markdown | 参考答案层级与 takeaways | 比较 fixed/detection/event trigger |

## 6. Hands-on 实验

### 6.1 实验 A：YOLO-only

```text
输入：相同视频、相同帧数
VLM：关闭
输出：FPS、detection latency、frame accounting
```

参与者先回答：输入视频 FPS 是多少、throughput 是否等于播放速度、哪个指标能证明没有丢帧。

### 6.2 实验 B：VLM-only

从已有 YOLO detections 中选一个 ROI：

```text
ROI -> JPEG -> Base64 -> HTTP -> Qwen3-VL -> text
```

记录 ROI 尺寸、JPEG bytes、client end-to-end latency、输出长度和事实约束。

### 6.3 实验 C：YOLO + async VLM

固定默认：

```text
interval=30
top_k=3
busy_policy=skip
llama slots=3
single GPU
```

参与者观察：

- YOLO 是否等待 VLM；
- VLM active 时 detector latency 是否上升；
- 理论 trigger opportunities 与实际 submitted batches 是否一致；
- 一批实际有几个 ROI；
- 视频帧数是否完整。

### 6.4 实验 D：只改一个变量

```text
Group A: interval 15
Group B: interval 60
Group C: top_k 1（若 CLI 已实现）
Group D: split GPU（若环境允许）
```

禁止同时改变多个参数，否则无法解释因果关系。

## 7. 必要的可观测性

Hands-on 不能使用估算调用次数。运行日志至少输出：

```json
{
  "frames": 240,
  "trigger_opportunities": 0,
  "trigger_with_detections": 0,
  "submitted_batches": 0,
  "skipped_busy": 0,
  "submitted_rois": 0,
  "completed_rois": 0,
  "failed_rois": 0,
  "batch_latency_p50_ms": 0.0,
  "batch_latency_p95_ms": 0.0,
  "yolo_fps": 0.0,
  "detection_p95_ms": 0.0
}
```

当前 `frame_count // interval + 1` 只能估算触发机会，不能代表实际 VLM batch。正式生成 Hands-on Notebook 前，应先增加这些 counters；这是 instrumentation，不改变 skip 语义。

## 8. A/B 结果表

| 指标 | YOLO-only | 默认并行 | 改动后 | 解释 |
|---|---:|---:|---:|---|
| frames | | | | 必须相同 |
| wall FPS | | | | 同卡 contention |
| detection P50/P95 | | | | VLM active 对 MIGraphX 的影响 |
| trigger opportunities | 0 | | | 由 interval 和检测共同决定 |
| submitted batches | 0 | | | busy skip 后的真实数量 |
| completed ROIs | 0 | | | 与 batch/top-K 的关系 |
| VLM batch P50/P95 | - | | | 不使用 async submit time |
| skipped busy | 0 | | | 背压代价 |

参与者必须写一句决策：

```text
在我们的实时性目标下，我选择 ______，因为 ______；代价是 ______。
```

## 9. 联动思考题

统一题目：

> 当前系统每 30 帧选择置信度最高的 3 个 ROI。如何避免重复、过时或无价值的 VLM 请求？

每组提交六项：

```text
Trigger：何时调用？
Evidence：给哪些帧、ROI 和 metadata？
Output：返回哪些受限字段？
Backpressure：VLM 忙时怎么处理？
Failure：超时或不确定时怎么处理？
Metrics：如何证明策略更好？
```

参考层级：

| 方案 | Trigger | 优点 | 问题 |
|---|---|---|---|
| A 固定间隔 | 每 N 帧 | 简单、可复现 | 与内容无关 |
| B Detection trigger | 出现目标类别 | 更相关 | 重复调用、无时间连续性 |
| C Tracking trigger | 新 Track、区域进入、停留 | 能去重 | 需要 tracker 状态 |
| D Event trigger | 确定性事件 + pre/post evidence | 价值最高 | 实现和验证成本最高 |

课堂不要求实现 C/D，只要求说明升级需要哪些状态和失败策略。

## 10. 课堂节奏

建议 20 分钟：

| 时间 | 内容 |
|---:|---|
| 0-3 min | 阅读任务、预测 top-3 ROI |
| 3-7 min | YOLO-only 与 VLM-only baseline |
| 7-10 min | 选择 interval/placement，写下预测 |
| 10-14 min | 运行或复用 async VLM 结果 |
| 14-17 min | 阅读 A/B 速度账本 |
| 17-20 min | 完成联动设计答卷 |

## 11. 产物

```text
output/hands_on/
├── decision.json
├── yolo_only_metrics.json
├── parallel_metrics.json
├── comparison.csv
├── design_answer.json
└── run.log
```

`decision.json` 保存参与者选择；`design_answer.json` 保存联动思考题，不执行尚未实现的策略。

## 12. 验收标准

- Notebook 是合法 JSON，所有 cell 有正确 `metadata.language`。
- clean kernel 执行，0 error。
- 默认配置无需编辑源码即可运行。
- 只能选择实际支持的 executable options。
- A/B 每次只改变一个变量。
- 指标使用真实 submitted/completed/skipped 计数。
- 不把 `--parallel 3` 解释为调用周期或三倍性能。
- 不把 async submit latency 解释为 VLM inference latency。
- 不把固定 30 帧触发描述为已实现的业务联动。
- 每位参与者最终产出一份可检查的设计答卷。

## 13. Takeaways

1. **异步消除等待，不消除 GPU contention。**
2. **并发 slot 是容量，不是调用策略。**
3. **间隔越短，更新更及时，但 busy skip 和资源竞争更多。**
4. **top-K 越大，上下文可能更多，但请求与显存成本也更高。**
5. **固定间隔解决怎么并行；事件驱动解决什么时候值得调用。**
