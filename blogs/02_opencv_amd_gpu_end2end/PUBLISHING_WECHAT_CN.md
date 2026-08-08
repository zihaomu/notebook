# OpenCV 公众号发布说明（最终版）

## 推荐主标题

**OpenCV、MIGraphX、VLM：我们提出了一套跨框架零主机拷贝推理架构**

标题强调本文的独立贡献是架构与数据契约，不把标准 Q8_0 notebook 没有执行的定制 llama.cpp IPC consumer 写成完整实测。

## 推荐摘要

我们独立提出了一套以 OpenCV HIP 为设备图像中枢的跨框架零主机拷贝推理架构：rocDecode 通过 DLPack 交出 GPU tensor，OpenCV 以 non-owning `GpuMat` 处理外部显存并直接写入 MIGraphX 的 GPU blob；VLM 分支在 GPU 上完成 ROI 预处理并导出 64 字节 HIP IPC handle。最终 step-by-step notebook 已执行验证检测链、GPU NMS 与 IPC 客户端导出；定制 llama.cpp IPC consumer 标记为未执行，语义基线仍走标准 JPEG 接口。完整视频基线在 W7900D 上处理 393 帧达到 50.8 FPS。

## 唯一配套项目

```text
/home/zihaomu/bigssd/notebook/opencv_amd_end2end
```

最终复现入口：

- `opencv_amd_end2end_step_by_step.ipynb`：指针、GPU NMS、HIP-IPC 客户端和 copy audit
- `opencv_amd_end2end.ipynb`：393 帧完整视频 workflow
- `output/pipeline/manifest.json`：最终环境、模型和产物清单
- `output/pipeline/01_yolo_base.log`：50.8 FPS 保存日志
- `output/pipeline/02_llamacpp_q8_timeline.json`：四段字幕与延迟
- `output/pipeline/03_final_llamacpp_q8.mp4`：最终 H.264 视频

博客目录不重复保存源码、日志、JSON、notebook 或 MP4。

## 封面与配图

- 横版封面：`assets/wechat_cover_cn.jpg`，900×383
- 方形分享图：`assets/wechat_thumbnail_cn.jpg`，500×500
- 零主机拷贝架构与验证边界：`assets/zero_host_copy_architecture_cn.png`，1800×1370
- 输入片段：`assets/input_sidewalk.gif`，由最终项目 `data/sidewalk.mp4` 生成
- 输出片段：`assets/output_sidewalk_vlm.gif`，由最终项目 `03_final_llamacpp_q8.mp4` 生成
- VLM storyboard：`assets/vlm_storyboard_segment_03.jpg`
- MIGraphX/NMS overlay：`assets/detection_overlay.png`

推荐正文顺序：封面 → 最终视频 → 输入/输出 GIF → 架构主图 → copy audit → storyboard → 预处理与检测图。

## 最终视频

上传以下项目文件，不在博客目录维护副本：

```text
/home/zihaomu/bigssd/notebook/opencv_amd_end2end/output/pipeline/03_final_llamacpp_q8.mp4
```

规格：H.264、1920×1220、25 FPS、393 帧、15.72 秒。

## 最终可引用数据

| 指标 | 最终项目保存结果 |
|---|---:|
| 基础视觉链路 | **50.8 FPS** |
| 总耗时 | 7.74 s / 393 帧 |
| OpenCV HIP 预处理 | 1.69 ms/frame |
| MIGraphX FP16 检测 | 9.00 ms/frame |
| 后处理 | 0.56 ms/frame |
| 场景片段 | 4 段，每段 4 秒左右 |
| VLM 片段延迟 | 1.67–1.90 s |
| IPC handle | 64 bytes |
| 检测候选/回传 | 300 candidates → 9 survivors |
| 预处理 validation 最大误差 | 5.96×10⁻⁸ |
| 解码 validation MAE | 0.5658/255 |

不要再引用旧博客包中的 78–80 FPS、51.1 FPS、32.0 FPS、HIP-IPC ingestion 表或并发 ROI 跑分；这些数据不在最终配套项目的文本证据中。

## 原创性与证据边界

可以准确表述为：

> 我们独立提出并实现了以 OpenCV HIP 为设备图像中枢的 OpenCV–MIGraphX–VLM 跨框架零主机拷贝推理架构，并用最终 notebook 对设备指针链、GPU NMS 和 HIP-IPC 客户端出口进行了逐阶段验证。

必须同时保留以下边界：

1. DLPack、HIP IPC、OpenCV 和 MIGraphX 是已有基础技术；原创点是总体数据通路、五项显存契约、OpenCV 接口补齐和定制服务端方案。
2. 检测链 A/B/C 已在最终 notebook 执行。
3. HIP-IPC 客户端 D 已执行，导出 8,110,080-byte packed buffer 和 64-byte handle。
4. 定制 llama.cpp IPC consumer 在 copy audit 中标记为 `Not exercised here`。
5. 标准 Q8_0 语义基线走 CPU ROI → JPEG → HTTP → llama.cpp，不是零主机拷贝证据。
6. CPU overlay 与 VA-API `hwupload` 搬运完整帧，不属于零主机拷贝主张。
7. `zero-host-copy` 允许必要的 GPU→GPU reshape/continuous/IPC pack，也允许 NMS 后少量坐标回传。

## 发布前检查

- 最终项目两本 notebook 仍可打开，保存输出中没有 error。
- `manifest.json` 状态为 `PASS`，视觉链路为 50.8 FPS。
- 最终视频为 393 帧、1920×1220、15.72 秒。
- 封面写“推理路径零主机拷贝”，不写“完整视频全程绝对零拷贝”。
- 架构图明确标注定制 IPC consumer 本次 notebook 未执行。
- 云端活动权益、额度和实例可用性按发布当天页面规则复核。

## 云端入口

- [AMD 开发者注册链接](https://developer.amd.com.cn/login?source=mEi9fAoWW)
- [Radeon Cloud Gallery](https://radeon.anruicloud.com/)

本文项目依赖定制 OpenCV、rocDecode、MIGraphX 和 llama.cpp 环境，不应把任意基础云实例理解为已预装同一套实验。
