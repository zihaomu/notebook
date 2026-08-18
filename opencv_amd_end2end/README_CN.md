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

## Notebook 运行镜像

镜像内明确区分两个根目录：

```text
/workspace/                        # 当前目录，唯一的可写宿主挂载
├── opencv_amd_end2end*.ipynb
├── data/  models/  output/
└── docker/  scripts/  src/

/opencv_workspace                  # 镜像内置工具/源码，不是运行时挂载
├── THIRD_PARTY_VERSIONS
├── third_party/opencv             # e038708，5.x-hip
├── third_party/opencv_contrib     # 467cbc6，5.x-hip-zerocopy
└── opencv_end2end/third_party -> ../third_party

/opt/opencv5                       # 已安装的 OpenCV 5 HIP
/opt/rocm                          # ROCm、MIGraphX、rocDecode Python binding
```

构建镜像：

```bash
bash scripts/build_notebook_image.sh
bash scripts/test_notebook_image.sh
```

默认生成 `zihao/opencv-amd-end2end:rocm7.2.1`。构建脚本复用已验证的
`zihao/opencv-llamacpp-q8:rocm7.2.1` 运行时层，并把旧工程中的两个干净源码仓
作为 BuildKit context 写入镜像 `/opencv_workspace/third_party`。`.git` 历史不会进入
镜像；基础镜像 ID 和两个 commit 会写入 image label 以及
`/opencv_workspace/THIRD_PARTY_VERSIONS`。

旧目录 `../../opencv_workspace/opencv_end2end` **只在构建镜像时提供源码**，运行
notebook 容器时不再挂载。需要发布到其他仓库时，可设置 `PIPELINE_IMAGE`；需要替换
基础镜像或源码位置时，可设置 `BASE_IMAGE`、`LEGACY_WORKSPACE`、`OPENCV_SOURCE`
和 `OPENCV_CONTRIB_SOURCE`。

llama.cpp Q8_0 仍使用独立的 `zihao/llamacpp-q8:b9766-rocm` 服务容器，与 notebook
容器加入同一 Docker network。它只读挂载当前包的 `models/`，因此 notebook 下载完
GGUF 后即可自动加载。

## 本地启动

无需额外传入模型目录，也无需挂载旧 workspace。脚本会在镜像缺失时自动构建，并将
当前目录唯一挂载到容器 `/workspace`：

```bash
bash scripts/start_notebook_container.sh
```

云端机器本身的访问地址：

```text
http://127.0.0.1:8891/?token=opencv-amd-end2end
```

通过 VS Code Remote SSH 登录时，在 **端口（Ports）** 面板转发远端 `8891`。如果
本机 `8891` 已被占用，VS Code 会自动选择其他本地端口；当前会话实际映射为
`8891 -> 8892`，因此本机浏览器使用：

```text
http://127.0.0.1:8892/lab?token=opencv-amd-end2end
```

`/home/zihaomu/.vscode/settings.json` 已将该端口标记为
`OpenCV AMD JupyterLab`，并设置重连后恢复转发、自动打开本机浏览器。以后以 VS Code
**端口**面板显示的本地端口为准，不要假设它始终是 `8892`。

GGUF 尚未下载时，Jupyter 会先启动，llama.cpp 在后台等待。执行 notebook 第一个单元后，
模型会下载到相对 `models/`；两个文件完成后 llama.cpp 自动开始加载。

确认实际挂载和镜像内第三方版本：

```bash
docker inspect opencv_amd_end2end_notebook --format '{{json .Mounts}}'
docker exec opencv_amd_end2end_notebook cat /opencv_workspace/THIRD_PARTY_VERSIONS
```

挂载列表应只有当前包到 `/workspace`；`/opencv_workspace` 应来自镜像层。停止服务：

```bash
bash scripts/stop_notebook_container.sh
```
