# YOLO26 Step-by-step 教学逻辑

> 目标文件：`ultralytics_yolo26x_step_by_step.ipynb`
> 定位：视觉优先、因果链优先的推荐讲师版本

## 核心问题

> Warm inference 已经很快，为什么完整视频 pipeline 仍可能慢？

当前版本不把 H2D/D2H 当作独立 CUDA 知识点，而是沿一帧数据定位系统边界：

```text
正确模型 -> 部署引擎 -> 数据位置 -> 删除往返
-> parity -> 系统 A/B -> VLM 侧路 -> production proof
```

## 十步逻辑

| 步骤 | 问题 | 证据 | 下一步 |
|---:|---|---|---|
| 1 | 实验条件是否固定？ | 模型、GPU、Provider、cache、输入帧 | 建立 PT 基线 |
| 2 | 部署必须保留什么？ | PT 检测图、静态 ONNX shape | 区分 cold/warm |
| 3 | 哪些成本只发生一次？ | first predict 与 warm P50/P95 | 追踪模型外的一帧 |
| 4 | 一帧在哪里跨设备？ | payload、H2D/D2H、25 FPS 流量 | 删除重复整帧往返 |
| 5 | 如何保持 GPU 常驻？ | GPU tensor、stable pointer、detections | 检查结果是否改变 |
| 6 | 优化是否仍正确？ | box/class parity、IoU | 测量完整 resident stages |
| 7 | GPU 时间花在哪里？ | decode/preprocess/infer/NMS | 做系统边界 A/B |
| 8 | 删除 host round-trip 有效吗？ | 相同 120 帧 CPU vs GPU-direct | 添加低频 VLM |
| 9 | VLM 如何不阻塞每帧路径？ | storyboard、caption、独立 latency | production 验收 |
| 10 | 完整工作流是否成立？ | 视频、manifest、393/393、copy audit | 总结边界 |

## H2D/D2H 讲法

第 4 步先展示两条真实路径：

```text
CPU round-trip: GPU decode -> full-frame D2H -> CPU overlay -> upload -> GPU encode
GPU direct:     GPU decode -> preprocess -> infer/NMS -> overlay/NV12 -> encode
```

单张 1080p RGB 帧约为 `5.93 MiB`：

- 单向、25 FPS：约 `148.3 MiB/s`；
- D2H + 再上传：约 `296.6 MiB/s`；
- `1x300x6 FP32` 紧凑输出：约 `0.172 MiB/s`。

结论不是“单次 DMA 很慢”，而是：

> 重复整帧往返会引入同步、CPU overlay 和再次上传，破坏连续的 GPU pipeline。

## 系统 A/B 口径

第 8 步固定同一视频、模型、120 帧、rocDecode 和 VA-API，只改变边界：

| 路径 | FPS | 整帧 D2H |
|---|---:|---:|
| CPU round-trip | 48.209 | 2.774 ms/frame |
| GPU direct | 97.959 | 0 ms/frame |

本次实测 GPU direct 为 CPU round-trip 的 `2.03x`。该结果只说明本机、本视频和当前负载下的边界收益，不作为跨环境固定承诺。

## 文案约束

- 每段 Markdown 只保留“问题 / 证据 / 下一步”。
- 细节放在代码、图表和 `output/step_by_step/logs/`。
- 不用 resident inference 推导完整视频 FPS。
- transfer microbenchmark 只用于解释 payload，不单独证明系统收益。
- 所有性能结论必须经过 parity 和 frame accounting。
