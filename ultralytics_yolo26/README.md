# Ultralytics YOLO26x Production Video Workshop on AMD Radeon

Take an Ultralytics YOLO26x checkpoint beyond `predict()` and into a production-style video pipeline on AMD Radeon. The workshop starts with the familiar Ultralytics Python API, exports YOLO26x to ONNX, runs it through an Ultralytics-owned ONNX Runtime MIGraphX backend, and connects it to hardware video decode, OpenCV HIP preprocessing, GPU NMS, hardware encode, and Qwen3-VL scene understanding.

**Speaker:** Zihao Mu, Member of Technical Staff, Product Application Engineering, AMD

**Audience:** Ultralytics users who train, fine-tune, export, or deploy YOLO models, and vision engineers moving from notebook inference to continuous video pipelines.

**Experience:** Intermediate. Attendees should be comfortable with Python, basic YOLO inference, and Jupyter notebooks.

**Preparation:** A laptop with a modern browser. All experiments run in a preconfigured Radeon Cloud environment.

## Notebooks

1. [`ultralytics_yolo26x_step_by_step.ipynb`](ultralytics_yolo26x_step_by_step.ipynb)
   - Establishes a `YOLO.predict()` baseline.
   - Demonstrates the real `export(format="onnx")` path.
   - Proves MIGraphX FP16, GPU I/O Binding, stable device pointers, OpenCV HIP preprocessing, and GPU NMS.
   - Separates GPU detection latency from overlay and encoded-video transport.
2. [`ultralytics_yolo26x_end_to_end.ipynb`](ultralytics_yolo26x_end_to_end.ipynb)
   - Runs or identity-validates the complete 393-frame video workflow.
   - Displays GPU-stage performance, Qwen3-VL temporal scenes, subtitles, the final video, and the reproducibility manifest.

Both notebooks are generated from [`scripts/build_notebooks.py`](scripts/build_notebooks.py), executed in the workshop image, and saved with outputs.

## Architecture

```text
YOLO26x checkpoint --Ultralytics export--> static ONNX [1,3,640,640] -> [1,300,6]
                                                |
video -> rocDecode -> OpenCV HIP preprocess -> Ultralytics ONNX/MIGraphX
                                                |
                                    ORT GPU I/O Binding
                                                |
                                    OpenCV GPU NMS
                                                |
                         compact detections + resident RGB frame
                                                |
                       bounded queue + dedicated HIP encode stream
                                                |
                 RGB -> NV12 + box/text overlay in a DRM PRIME surface
                                                |
                                  h264_vaapi -> MP4
```

Ultralytics owns model loading, metadata, provider selection, FP16 compilation, cache selection, and inference. OpenCV owns preprocessing and GPU NMS. The native bridge maps separate FFmpeg-owned linear NV12 VA-API encoder surfaces through DRM PRIME into HIP; it does not pass rocDecode surfaces directly to the encoder.

The no-VLM vision pass keeps every full-frame buffer on the GPU. A bounded worker queue overlaps HIP RGB-to-NV12/overlay and VA-API submission for frame N with inference for frame N+1. Only compact surviving detections cross to the host. The Qwen3-VL analysis and subtitle-rendering pass intentionally retains JPEG/HTTP and host-frame boundaries.

## Ultralytics MIGraphX Backend

The image installs the workshop fork at commit:

```text
zihaomu/ultralytics@34e213ca3ece4c18962f5bb922ec74da0c474d24
```

It also installs `onnxruntime-migraphx==1.24.2` and applies [`docker/patches/ultralytics-migraphx-iobinding.patch`](docker/patches/ultralytics-migraphx-iobinding.patch). The patch:

- enables static-shape GPU I/O Binding for `MIGraphXExecutionProvider`;
- binds ROCm PyTorch input/output pointers directly;
- keeps output allocations stable across the video loop;
- skips redundant torchvision NMS warmup for end-to-end YOLO26 exports.

This fork is a pinned workshop backend, not a claim that the changes are already part of an upstream Ultralytics release.

## Models and Cache

The first notebook cell prepares and strictly validates:

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `yolo26x.pt` | 118,667,365 | `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92` |
| `yolo26x.onnx` | 223,287,479 | `88568299de91d4967f239a062c9f1619f695ebd05de73cd66b8f589591aaeb0a` |
| `Qwen3-VL-8B-Instruct-Q8_0.gguf` | 8,709,520,224 | `cb8616bf6ed228982d9e47d7b72b42195342efa26044b0ee1873e61d9e78d3d7` |
| `mmproj-F16.gguf` | 1,159,030,336 | `d406d03ebabefdef86a2c86bf0c1b65f9e046f7a81c218f25de4931b46a07fc4` |

Downloads use `.part` files, automatic bounded retries, HTTP Range resume, fixed size/SHA validation, and atomic replacement. See [`models/README.md`](models/README.md) for URLs and overrides.

ORT-generated `.mxr` files live under `models/ort-migraphx-cache/<identity>/`. The identity includes the ONNX SHA, GPU architecture/name, ROCm/PyTorch HIP, MIGraphX, ONNX Runtime, Ultralytics commit, and workshop patch SHA. A cache is not assumed portable across these identities.

## Measured Results

Validated on an AMD Radeon PRO W7900D (`gfx1100`), ROCm 7.2, with the official release ONNX:

| Scope | Result |
|---|---:|
| Same-input native MIGraphX inference | 7.109 ms mean |
| Same-input Ultralytics/ORT I/O Binding | 7.374-7.800 ms mean; zero steady-state memory-copy records |
| GPU-resident detection benchmark | 50.5-87.7 FPS observed; strongly host-load dependent |
| Legacy host overlay + raw BGR/VA-API workflow | 37.1 FPS at host load 181.36 |
| Current async HIP overlay + direct VA-API workflow | **74.7 FPS** at host load 218.12 |
| Independent 393-frame direct-path repeat | **71.9 FPS** at host load 209.15 |
| Direct queue / full-frame D2H | 393 submitted = 393 encoded / 0.00 ms |
| Qwen3-VL temporal analysis | 4 segments |
| Final output | 393 frames, 1920x1220, 15.72 s |

The current vision pass reports 10.44 ms/frame detection, 7.85 ms/frame in the asynchronous GPU overlay/encode worker, and only 0.15 ms/frame of main-thread queue feed. The worker overlaps with inference, so stage means are not summed to derive end-to-end FPS. Both MP4 outputs contain 393 packets, 393 container samples, and 393 decodable frames.

The historical `50.8 FPS` result is not a same-run backend comparison: it was captured on August 8 with a different ONNX (`97d516...`) and a less-loaded host. On September 2, with host load average around 188-209, a back-to-back test on the same GPU/render node measured the historical native pipeline at `27.4 FPS` and the current Ultralytics/ORT pipeline at `29.8 FPS`. rocprof recorded zero memory-copy operations inside both native and ORT steady-state inference ranges. See [`doc/performance_diagnosis_CN.md`](doc/performance_diagnosis_CN.md) for the complete A/B and copy audit.

A whole-loop `rocprofv3 --memory-copy-trace` was also attempted after the direct encoder was added. Both CSV and JSON runs completed the application frames, then ROCm 7.2.1 failed during result export with `ring_buffer mmap failed with errno 22`; the JSON was truncated. Those files are not counted as copy-trace evidence. The valid zero-copy trace claim remains scoped to the 50-call `MEASURE_YOLO` inference markers; the no-full-frame-D2H direct-path claim is additionally enforced by the selected code path, runtime metric, queue accounting, frame accounting, and pixel audit.

Raw workshop evidence is in [`output/benchmarks/gpu_stages.json`](output/benchmarks/gpu_stages.json), [`output/benchmarks/performance_diagnosis.json`](output/benchmarks/performance_diagnosis.json), [`output/benchmarks/direct_encode_repeat_393.log`](output/benchmarks/direct_encode_repeat_393.log), the two explicit [`CSV`](output/benchmarks/rocprof_direct_copy_csv_failure.log) / [`JSON`](output/benchmarks/rocprof_direct_copy_json_failure.log) profiler failure logs, and [`output/pipeline/manifest.json`](output/pipeline/manifest.json).

## Build the Workshop Image

The default base is the user-provided validated image:

```text
crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/
muzihao2/work:opencv_end2end_2026_08_12
```

Build the thin derived image:

```bash
bash scripts/build_notebook_image.sh
```

Default output image:

```text
zihao/ultralytics-yolo26-workshop:rocm7.2.1
```

The build script pins and validates the fork commit, prefetches the ORT wheel, and compiles the HIP/DRM PRIME/VA-API bridge into `/opt/venv`. The image records both the Ultralytics patch SHA and bridge source SHA as OCI labels. This avoids relying on Git/PyPI connectivity inside `docker build` and prevents a bind mount from hiding a missing native module.

## Start Locally

Choose the physical GPU and matching VA-API render node for the host:

```bash
PIPELINE_GPU=0 \
LLAMA_GPU=0 \
VAAPI_DEVICE=/dev/dri/renderD128 \
bash scripts/start_notebook_container.sh
```

Open:

```text
http://127.0.0.1:8892/?token=ultralytics-yolo26
```

Stop the services with:

```bash
bash scripts/stop_notebook_container.sh
```

## Validation

```bash
# Image, provider, rocDecode, GPU I/O Binding, and one-frame direct encode
bash scripts/test_notebook_image.sh

# Model/runtime gate
python3 scripts/validate_runtime.py --require-models

# High-level predict vs production path
python3 tests/test_predict_production_parity.py \
  --model models/yolo26x.onnx --video data/sidewalk.mp4

# GPU-resident stage benchmark
python3 tests/benchmark_gpu_stages.py \
  --video data/sidewalk.mp4 --frames 120 --warmup 10

# GPU overlay + direct VA-API frame-accounting regression
python3 tests/test_gpu_overlay_encode.py \
  --video data/sidewalk.mp4 --output output/direct-test.mp4 --frames 30

# Validate saved 393-frame workflow identity and >=50 FPS gate
python3 scripts/run_pipeline.py --validate-only
```

Run these Python commands inside the workshop image or notebook container.
