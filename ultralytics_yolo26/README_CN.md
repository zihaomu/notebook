# AMD Radeon 上的 Ultralytics YOLO26x 生产视频 Workshop

本 workshop 将 Ultralytics YOLO26x 从熟悉的 `predict()` 带入生产式连续视频流水线。内容从 Ultralytics Python API 和 ONNX 导出开始，再由 Ultralytics 持有的 ONNX Runtime MIGraphX 后端完成 Radeon FP16 推理，并连接 rocDecode、OpenCV HIP 前处理、GPU NMS、硬件编码和 Qwen3-VL 场景理解。

**讲师：** Zihao Mu，Member of Technical Staff，Product Application Engineering，AMD

**受众：** 训练、微调、导出或部署 YOLO 的 Ultralytics 用户，以及希望从 notebook 单图推理迁移到连续视频流水线的计算机视觉工程师。

**难度：** 中级，需熟悉 Python、基础 YOLO 推理和 Jupyter notebook。

**准备：** 只需一台能运行现代浏览器的笔记本电脑；所有实验都在预配置 Radeon Cloud 环境中运行。

## 两本 Notebook

1. [`ultralytics_yolo26x_step_by_step.ipynb`](ultralytics_yolo26x_step_by_step.ipynb)
   - 从 `YOLO.predict()` 建立正确性基线。
   - 展示真实 `export(format="onnx")` 路径。
   - 验证 MIGraphX FP16、GPU I/O Binding、稳定设备指针、OpenCV HIP 前处理和 GPU NMS。
   - 将 GPU 检测性能与 overlay/编码传输分开统计。
2. [`ultralytics_yolo26x_end_to_end.ipynb`](ultralytics_yolo26x_end_to_end.ipynb)
   - 运行或按身份校验复用完整 393 帧 workflow。
   - 展示分阶段性能、Qwen3-VL 时间线、字幕、最终视频和可复现 manifest。

两本 notebook 由 [`scripts/build_notebooks.py`](scripts/build_notebooks.py) 确定性生成，已在 workshop 镜像中从 clean kernel 执行并保存输出。

## 架构与职责

```text
YOLO26x checkpoint --Ultralytics export--> 静态 ONNX [1,3,640,640] -> [1,300,6]
                                                |
视频 -> rocDecode -> OpenCV HIP 前处理 -> Ultralytics ONNX/MIGraphX
                                                |
                                      ORT GPU I/O Binding
                                                |
                                      OpenCV GPU NMS
                                                |
                             紧凑检测结果 + GPU RGB 帧
                                                |
                              有界队列 + 独立 HIP 编码流
                                                |
                  DRM PRIME surface 内 RGB -> NV12 + box/text overlay
                                                |
                                    h264_vaapi -> MP4
```

Ultralytics 负责模型加载、metadata、provider 选择、FP16 编译、cache 选择和推理；OpenCV 负责前处理和 GPU NMS。native bridge 将 FFmpeg 单独创建的 linear NV12 VA-API 编码 surface 通过 DRM PRIME 映射到 HIP；它不是 rocDecode surface 的直接透传。

no-VLM 主视觉 pass 的整帧 buffer 全部留在 GPU。有界 worker queue 将第 N 帧的 HIP RGB-to-NV12/overlay 和 VA-API submit 与第 N+1 帧推理重叠；只有 GPU NMS 后的紧凑存活检测结果进入 host。Qwen3-VL 分析和字幕渲染 pass 则有意保留 JPEG/HTTP 与 host-frame 边界。

## Ultralytics MIGraphX 后端

镜像固定安装：

```text
zihaomu/ultralytics@34e213ca3ece4c18962f5bb922ec74da0c474d24
onnxruntime-migraphx==1.24.2
```

并应用 [`docker/patches/ultralytics-migraphx-iobinding.patch`](docker/patches/ultralytics-migraphx-iobinding.patch)：

- 为 `MIGraphXExecutionProvider` 启用静态 shape GPU I/O Binding；
- 直接绑定 ROCm PyTorch 输入/输出指针；
- 在视频循环中复用输出 allocation；
- 对 YOLO26 end-to-end 导出跳过多余的 torchvision NMS warmup。

这是固定的 workshop backend，不表示这些改动已经进入 Ultralytics 上游正式版本。

## 模型与 Cache

在 full release 中，第一个 Notebook 单元会离线校验镜像内置的以下资产：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `yolo26x.pt` | 118,667,365 | `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92` |
| `yolo26x.onnx` | 223,287,479 | `88568299de91d4967f239a062c9f1619f695ebd05de73cd66b8f589591aaeb0a` |
| `Qwen3-VL-8B-Instruct-Q8_0.gguf` | 8,709,520,224 | `cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7` |
| `mmproj-F16.gguf` | 1,159,030,336 | `d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4` |

仅在源码开发或非 full 镜像模式下，缺失文件才会进入下载 fallback；下载器支持 `.part`、自动有界重试、HTTP Range 续传、固定大小/SHA 校验和原子替换。baked 模式发现文件缺失或损坏时会直接失败，绝不转为联网下载。URL 和覆盖变量见 [`models/README.md`](models/README.md)。

ORT 生成的 `.mxr` 位于 `models/ort-migraphx-cache/<identity>/`。identity 绑定 ONNX SHA、GPU、ROCm/PyTorch HIP、MIGraphX、ORT、Ultralytics commit 和 workshop patch SHA，不能假设可跨环境复用。

## 实测结果

环境：AMD Radeon PRO W7900D (`gfx1100`) / ROCm 7.2 / 官方 release ONNX。

| 范围 | 结果 |
|---|---:|
| 同输入 native MIGraphX inference | mean 7.109 ms |
| 同输入 Ultralytics/ORT I/O Binding | mean 7.374-7.800 ms；稳态 memory-copy 记录为 0 |
| GPU 常驻检测 benchmark | 实测 50.5-87.7 FPS；对主机负载敏感 |
| 旧 host overlay + raw BGR/VA-API workflow | host load 181.36 时为 37.1 FPS |
| 当前异步 HIP overlay + direct VA-API workflow | **74.7 FPS**，host load 218.12 |
| 独立 393 帧 direct-path 复跑 | **71.9 FPS**，host load 209.15 |
| Direct queue / 整帧 D2H | 393 submitted = 393 encoded / 0.00 ms |
| Qwen3-VL | 4 个时间段 |
| 最终输出 | 393 帧，1920x1220，15.72 秒 |

当前视觉 pass 的 detection 为 10.44 ms/frame，异步 GPU overlay/encode worker 为 7.85 ms/frame，而主线程 queue feed 仅 0.15 ms/frame。worker 与推理重叠，因此不能将阶段均值直接相加计算端到端 FPS。YOLO 与最终 MP4 均严格包含 393 个 packet、393 个 container sample 和 393 个可解码帧。

历史 `50.8 FPS` 不是同一次 A/B：它于 8 月 8 日使用另一份 ONNX（`97d516...`）和当时负载更低的主机生成。9 月 2 日主机 load average 约 188-209 时，在同一 GPU/render node 背靠背复测，历史 native 流程为 `27.4 FPS`，当前 Ultralytics/ORT 流程为 `29.8 FPS`。rocprof 在 native 与 ORT 的稳态 inference marker 内均记录到 0 条 memory-copy operation。完整诊断见 [`doc/performance_diagnosis_CN.md`](doc/performance_diagnosis_CN.md)。

direct encoder 完成后也尝试了全循环 `rocprofv3 --memory-copy-trace`。CSV 与 JSON 两次都先正常完成被测帧，然后 ROCm 7.2.1 在结果导出阶段触发 `ring_buffer mmap failed with errno 22`；JSON 被截断，因此这些文件不计作 copy-trace 证据。有效的 zero-copy trace 结论仍严格限定在 50 次 `MEASURE_YOLO` inference marker；主 YOLO direct path 的零整帧 D2H 另由选中代码路径、运行时计量、queue accounting、严格帧数和像素审计共同约束。

原始证据见 [`output/benchmarks/gpu_stages.json`](output/benchmarks/gpu_stages.json)、[`output/benchmarks/performance_diagnosis.json`](output/benchmarks/performance_diagnosis.json)、[`output/benchmarks/direct_encode_repeat_393.log`](output/benchmarks/direct_encode_repeat_393.log)、明确记录工具限制的 [`CSV`](output/benchmarks/rocprof_direct_copy_csv_failure.log) / [`JSON`](output/benchmarks/rocprof_direct_copy_json_failure.log) profiler failure log，以及 [`output/pipeline/manifest.json`](output/pipeline/manifest.json)。

## 构建镜像

基于用户指定镜像：

```text
crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/
muzihao2/work:opencv_end2end_2026_08_12
```

必须从 tracked-clean 工作树构建，并显式提供 release ID：

```bash
RELEASE_ID=ultralytics-yolo26-YYYY-MM-DD-rN \
  bash scripts/build_notebook_image.sh
```

生成：

```text
zihao/ultralytics-yolo26-workshop:rocm7.2.1-full
```

当前双镜像 release 的固定标签与 immutable OCI digest 统一记录在 [`release/current.env`](release/current.env)。pipeline 的人类可读标签是：

```text
crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work:ultralytics-yolo26-workshop-full_2026_09_04
```

完整部署由两个镜像组成：pipeline/Jupyter 镜像保存代码和全部模型文件，digest-pinned companion 镜像提供 `llama-server` 二进制。launcher 会在创建运行资源前拉取并校验两个镜像。

full 专用镜像在 `/workspace` 内包含完整 workshop，并在 `/opt/ultralytics-yolo26/models` 内固定保存四个模型文件、正式 `gfx1100` MIGraphX `.mxr`、patched Ultralytics、ORT MIGraphX，以及两个 build-time native 工件：pybind HIP/DRM PRIME/VA-API encoder 和 standalone surface capability probe。OpenCV HIP 与 rocDecode 从固定 base image 继承。

pipeline 直接读取镜像内不可变模型目录。由于 llama.cpp 是独立容器，启动脚本首次将镜像内模型按 SHA 校验后复制到 named volume `ultralytics_yolo26_models`，再以 `/models:ro` 挂给 llama.cpp；不再需要网络下载或宿主模型 bind mount。模型在镜像层只存一份，`/workspace` 不保留第二份。

OCI label 记录源码 commit、base image、Ultralytics patch、bridge source、bundle、四个模型和 MIGraphX cache 的 SHA-256。`scripts/test_notebook_image.sh` 在无网络、无源码/模型挂载、只读根文件系统下执行，证明编译环境和全部模型确实自包含。

## 启动

先登录 `release/current.env` 引用的 registry，再根据宿主机分配选择物理 GPU 和对应 VA-API render node。launcher 会自动拉取两个 immutable image ref；显式镜像覆盖会标记为开发模式。

```bash
PIPELINE_GPU=0 \
LLAMA_GPU=0 \
VAAPI_DEVICE=/dev/dri/renderD128 \
bash scripts/start_notebook_container.sh
```

浏览器打开：

```text
http://127.0.0.1:8892/?token=ultralytics-yolo26
```

停止：

```bash
bash scripts/stop_notebook_container.sh
```

## 验证

启动或测试容器前先校验双镜像 release lock：

```bash
python3 scripts/validate_release_lock.py --check-local-images
```

```bash
bash scripts/test_notebook_image.sh
python3 scripts/validate_runtime.py --require-models
python3 tests/test_predict_production_parity.py \
  --model models/yolo26x.onnx --video data/sidewalk.mp4
python3 tests/benchmark_gpu_stages.py \
  --video data/sidewalk.mp4 --frames 120 --warmup 10
python3 scripts/run_pipeline.py --validate-only
```

Python 命令需在 workshop 镜像或 notebook 容器内执行。
