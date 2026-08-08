# OpenCV AMD End-to-End Notebooks

这是一个独立的 OpenCV 5 + AMD Radeon ROCm/HIP 端到端 notebook 文件夹。

## 两本 Notebook

1. `opencv_amd_end2end_step_by_step.ipynb`
   - 最前面检查 Python、OpenCV HIP、ROCm GPU、MIGraphX、磁盘和相对模型目录
   - Qwen 缺失时通过 `hf-mirror.com` 自动下载，并在 notebook 显示实时进度
   - 展示 rocDecode、OpenCV HIP、MIGraphX YOLO26x、GPU NMS 和 Q8_0 VLM
2. `opencv_amd_end2end.ipynb`
   - 开头执行同样的环境与模型准备
   - 完整 393 帧视频 workflow
   - Q8_0 场景 timeline、SRT 与最终英文字幕视频
   - 默认复用已验证产物；设置 `RUN_PIPELINE=1` 可重新生成

两本 notebook 都保存 W7900D 成功执行的输出。

## 模型准备

这次源码提交暂不上传大模型二进制文件。运行 notebook 前，请将以下两个
YOLO26x 文件放入相对 `models/` 目录：

- `yolo26x.onnx`
- `yolo26x_compiled.mxr`（W7900D/gfx1100 的 MIGraphX FP16 产物）

`models/README.md` 记录了预期文件大小和 SHA-256；后续可通过 Git LFS 单独发布。

以下两个大文件不进入 Git：

- `Qwen3-VL-8B-Instruct-Q8_0.gguf`（8.11 GiB）
- `mmproj-F16.gguf`（1.08 GiB）

任意一本 notebook 的第一个代码单元都会检查 `./models`。如果 Qwen 文件缺失，则从
`unsloth/Qwen3-VL-8B-Instruct-GGUF` 经 `https://hf-mirror.com` 下载，并用
固定的 Jupyter HTML 进度条显示实时速度、进度和 ETA。下载支持 `.part` 断点续传，完成后检查文件大小、
GGUF 文件头和 SHA-256，再原子改名。

命令行等价操作：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/model_setup.py
```

## 本地启动

无需额外传入模型目录，直接运行：

```bash
bash scripts/start_notebook_container.sh
```

浏览器打开：

```text
http://127.0.0.1:8891/?token=opencv-amd-end2end
```

GGUF 尚未下载时，Jupyter 会先启动，llama.cpp 在后台等待。执行 notebook 第一个单元后，
模型会下载到相对 `models/`；两个文件完成后 llama.cpp 自动开始加载。

停止服务：

```bash
bash scripts/stop_notebook_container.sh
```
