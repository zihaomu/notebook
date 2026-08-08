# 第二篇：OpenCV–MIGraphX–VLM 零主机拷贝推理架构

## 最终文件

- [中文公众号最终稿](BLOG_CN.md)
- [公众号标题、摘要、配图与发布边界](PUBLISHING_WECHAT_CN.md)
- `assets/`：最终稿实际使用的发布素材

本目录只保存最终文章和发布素材。源码、notebook、模型、日志、JSON、SRT 和视频统一由配套项目维护。

## 文章总结

本文提出一套以 OpenCV HIP 为设备图像中枢的跨框架数据通路：

```text
rocDecode → DLPack GPU tensor → non-owning OpenCV GpuMat
          ├→ OpenCV HIP preprocess → MIGraphX device pointer → GPU NMS
          └→ OpenCV GPU ROI preprocess → 64-byte HIP IPC handle
```

方案通过设备、布局、所有权、同步和生命周期五项契约，避免让大块图像和完整模型 tensor 以 Host RAM 作为框架交换格式。

最终 step-by-step notebook 实际验证了检测链和 HIP-IPC 客户端导出；定制 llama.cpp IPC consumer 未在标准 notebook 中执行。标准 Q8_0 语义基线走 JPEG，CPU overlay 与 VA-API `hwupload` 也是已知 Host 边界。

## 最终可复现结果

| 指标 | 结果 |
|---|---:|
| W7900D 视觉链路 | 50.8 FPS |
| 393 帧总耗时 | 7.74 s |
| OpenCV HIP 预处理 | 1.69 ms/frame |
| MIGraphX 检测 | 9.00 ms/frame |
| 后处理 | 0.56 ms/frame |
| 最终视频 | 1920×1220，25 FPS，15.72 s |
| VLM 片段延迟 | 1.67–1.90 s |

## 唯一配套项目

```text
/home/zihaomu/bigssd/notebook/opencv_amd_end2end
├── opencv_amd_end2end_step_by_step.ipynb
├── opencv_amd_end2end.ipynb
├── src/
├── scripts/
├── models/
├── data/
└── output/pipeline/
```

两本 notebook 均已保存 W7900D 成功执行输出且无错误。完整重跑：

```bash
cd /home/zihaomu/bigssd/notebook/opencv_amd_end2end
bash scripts/start_notebook_container.sh
# 或在项目容器中：
python3 scripts/run_pipeline.py --force
```

默认 notebook 复用并验证 `output/pipeline/`；设置 `RUN_PIPELINE=1` 可重新生成。

## 发布素材

保留的素材均由最终项目产物或最终文章边界生成：封面、分享图、输入/输出 GIF、零主机拷贝架构图、storyboard、预处理图和检测图。最终 MP4 直接从配套项目上传，不在博客目录复制。
