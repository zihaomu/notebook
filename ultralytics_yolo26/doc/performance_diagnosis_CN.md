# Ultralytics YOLO26x Workshop 性能差异与 GPU 驻留诊断

日期：2026-09-03

## 结论

`50.8 FPS -> 34 FPS` 不是因为 Ultralytics/ORT 将 YOLO 输入或输出搬回 CPU。

- rocprofv3 对 native MIGraphX 和 Ultralytics/ORT 各追踪 50 次稳态推理，`MEASURE_YOLO` 区间内均为 **0 条 memory-copy record**，没有 H2D 或 D2H。
- 同一个官方 ONNX、同一个 GPU input pointer、同一个预分配 output pointer 下，native 与 ORT 输出逐元素相同，最大绝对误差为 0。
- 稳定环境中的同步推理约为 native 7.1 ms、ORT I/O Binding 7.3-7.8 ms。Ultralytics 各层 wrapper 只带来很小的 Python 开销；主要额外成本是 ORT 调度，约 0.2-0.7 ms。
- 真正的大吞吐边界位于检测之后：整帧 D2H、CPU overlay、raw BGR pipe、VA-API `hwupload`。
- 当前主机为严重共享负载环境：128 个逻辑 CPU，测试时 load average 约 176-209。writer 单独处理 393 个 1080p BGR frame 只有约 31-42 FPS。
- 同一时刻、同一 GPU2/renderD130 背靠背复测：历史 native 流程 27.4 FPS，当前 Ultralytics/ORT 流程 29.8 FPS。因此历史 50.8 FPS 与当前 37.1 FPS（load 181.36）不是同环境可比数据。
- 诊断后的正式修复已完成：主 YOLO pass 使用 GPU overlay + DRM PRIME/VA-API direct encode，并通过独立 HIP stream 与下一帧推理重叠。正式 393 帧 workflow 在 load 218.12 下达到 **74.7 FPS**；独立复跑在 load 209.15 下达到 **71.9 FPS**。
- 两次 direct 跑均为 393 submitted = 393 encoded = 393 packets = 393 可解码帧，完整帧 D2H 为 0.00 ms/frame。Qwen3-VL 字幕 pass 仍保留 host-frame writer，这是有意保留的独立边界。

## 历史基线也不是同一个 ONNX

历史 50.8 FPS 使用：

```text
SHA-256: 97d5165312f2d1957a24d7ae27863a98860ac88656d40fdac8f10de77999bc35
Ultralytics: 8.4.115
simplify: false
ONNX nodes: 672
```

Workshop 官方 release ONNX 使用：

```text
SHA-256: 88568299de91d4967f239a062c9f1619f695ebd05de73cd66b8f589591aaeb0a
Ultralytics metadata version: 8.4.41
simplify: true
ONNX nodes: 606
```

但同卡 native microbenchmark 中两份图均约 7.1-7.2 ms，所以模型图差异不是当前完整视频降速的主因。

## 数据驻留边界

### 保持在 GPU 上

```text
rocDecode surface
  -> DLPack torch tensor
  -> OpenCV HIP warp/convert
  -> fixed BCHW torch tensor
  -> ORT MIGraphX I/O Binding
  -> fixed [1,300,6] GPU output
  -> OpenCV GPU NMS
```

`src/preprocess.py` 中的 `blob.copy_()` 是 device-to-device 操作。ORT 输入/输出均绑定 `torch.Tensor.data_ptr()`；rocprof 证明稳态 inference 区间没有 memory-copy activity。

### 检测后的当前边界

主 YOLO/no-VLM pass：

1. GPU NMS 后仅将存活 boxes/scores/classes 下载到 CPU；这是每帧几十到几百字节级的紧凑 metadata，不是整帧传输。
2. GPU RGB frame 由有界队列持有；producer event 保证推理流写入完成，worker 在独立 HIP stream 上等待该 event。
3. native bridge 将 FFmpeg 创建的 linear NV12 VA-API encoder surface 经 DRM PRIME 导入 HIP，在 surface 内完成 RGB-to-NV12、box、label 和状态文字 overlay。
4. 同一 VAAPI surface 直接提交给 `h264_vaapi`，不经过 `rgb_gpu.cpu().numpy()`、CPU OpenCV overlay 或 raw BGR pipe。

这不是 decode-surface passthrough：rocDecode RGB tensor 与 FFmpeg-owned encoder surface 是不同对象，中间发生 GPU-to-GPU 的像素格式转换。场景分析/字幕 pass 仍使用 JPEG/HTTP 与 host-frame VA-API writer，因为它需要 CPU 文字排版与扩展画布。

历史 OpenCV-first/raw-pipe 路径的四个 host 边界保留为 fallback，并可通过 `--gpu-direct-encode off` 复现；它们不是 Ultralytics/ORT 新增。

## 边界 A/B 实测

同一 GPU2、同一官方 ONNX、同一 OpenCV HIP/GPU NMS，在当前高负载主机上：

| 路径 | FPS |
|---|---:|
| decode + preprocess + ORT inference + GPU NMS | 65.4 |
| 上述路径 + full-frame D2H + CPU overlay | 59.7 |
| 上述路径 + raw BGR -> ffmpeg -> VA-API | 31.1 |

由此可见，编码输入边界是当前主机上的主要限制。

## 已修复的额外复制

异步 writer 初版在主线程执行：

```python
np.ascontiguousarray(frame_bgr).tobytes()
```

这会额外复制约 6.2 MB/frame。历史 writer 也有同样复制；当前 workshop 已改为队列持有 NumPy frame，worker 使用：

```python
memoryview(frame).cast("B")
```

需要说明：历史 writer 同样使用 `.tobytes()`，所以这不是 50.8 FPS 与当前数据的唯一差异；它只是可以消除的一项公共 host-host copy。

## ORT 调度优化

MIGraphX EP session 已改为：

```text
ORT_SEQUENTIAL
intra_op_num_threads=1
inter_op_num_threads=1
allow_spinning=0
```

目的是降低高 CPU 负载下 ORT 线程池调度抖动。该配置不会改变 GPU I/O Binding 或模型数值；补丁 SHA 会自动进入 cache identity，从而生成独立 MIGraphX cache。

## 可复现命令

GPU 边界 benchmark：

```bash
python3 tests/benchmark_video_boundaries.py \
  --mode gpu --video data/sidewalk.mp4

python3 tests/benchmark_video_boundaries.py \
  --mode overlay --video data/sidewalk.mp4

python3 tests/benchmark_video_boundaries.py \
  --mode encode --video data/sidewalk.mp4 \
  --output output/benchmarks/boundary.mp4
```

rocprof 零复制验证：

```bash
rocprofv3 --memory-copy-trace --marker-trace -- \
  python3 tests/profile_inference_residency.py \
  --model models/yolo26x.onnx --iterations 50
```

只统计 `MEASURE_YOLO` marker 的开始与结束时间内的 memory-copy records。


### Direct 全循环 profiler 限制

加入 direct encoder 后，分别以 CSV 和 JSON 输出对 30 帧与 8 帧完整循环执行 `rocprofv3 --memory-copy-trace`。两次被测 pipeline 都正常完成并输出完整视频，但 ROCm 7.2.1 的 profiler 在结果导出阶段均触发：

```text
ring_buffer: munmap failed: Invalid argument
ring_buffer.cpp: mmap failed with errno 22
```

CSV 为空，JSON 虽写入 28,426,240 bytes 但在字符串中间截断，均不能作为完整 trace 解析。因此本文不从这两次失败导出中推导“全循环 0 copy”。有效的 profiler 结论仍仅限上面的 `MEASURE_YOLO` inference marker。主 YOLO direct path 的“0.00 ms 整帧 D2H”由强制 `vaapi-direct` 分支、源码中不执行 `rgb_gpu.cpu()`、运行时 metric、393/393 queue drain、ffprobe 严格帧计数和 overlay 像素审计共同约束。机器可读状态见 `output/benchmarks/performance_diagnosis.json` 的 `profiler_export_limitation`。

## 完整视频 50+ FPS 已恢复

实施结果：

1. `native/hip_vaapi_bridge.cpp` 提供 HIP RGB-to-NV12、box/text/status overlay，以及 DRM PRIME VAAPI surface 提交。
2. `GpuDirectVaapiWriter` 使用深度 3 的有界队列、producer event 和独立 HIP stream，将 frame N 的 overlay/encode 与 frame N+1 的推理重叠。
3. 正式 393 帧运行：74.7 FPS，host load 218.12；detection 10.44 ms/frame，后台 GPU overlay + direct encode worker 7.85 ms/frame，主线程 feed 0.15 ms/frame。
4. 独立 393 帧复跑：71.9 FPS，host load 209.15；证明结果不是一次性低负载偶然值。
5. `ffprobe` 严格确认 H.264 High/yuv420p、15.72 秒、393 packets、393 samples、393 decoded frames。GPU overlay 的 5 帧像素抽样也通过。
6. FFmpeg 4.4 初版输出曾出现 30 packets 但仅 29 decoded frames；根因是 packet duration 缺失导致轨道时长少一帧。bridge 现在显式设置一帧 duration，回归测试要求 submitted/packets/samples/decoded 完全一致。
7. 正式 workflow 强制 `--gpu-direct-encode on`、`frame_d2h_ms == 0.0`、队列完整 drain，并设置 `>=50 FPS` 硬门槛。

阶段均值不能直接相加计算端到端 FPS，因为 inference 与 direct encode worker 在不同 stream/thread 上重叠。历史 37.1 FPS/raw-pipe 和 27.4/29.8 FPS A/B 仍用于解释根因，不代表当前默认路径。
