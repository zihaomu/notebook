# 第一篇：OpenCV 5 在 AMD Radeon 上的 HIP 加速

## 最终文件

- [中文公众号最终稿](BLOG_CN.md)
- [English final article](BLOG_EN.md)
- [公众号标题、摘要、配图和发布边界](PUBLISHING_WECHAT_CN.md)
- `assets/`：最终稿实际使用的发布素材

本目录只保存最终发布版本。实验代码、输入图片、二进制程序和 notebook 输出不在博客
目录重复维护。

## 文章总结

OpenCV 社区正在推进 OpenCV 5 的 ROCm/HIP 支持。当前实现保留用户熟悉的
`cv::cuda::GpuMat`、`cv::cuda::Stream` 和 GPU filter API，底层则由 AMD ROCm/HIP
在 Radeon GPU 上编译和执行。

Radeon PRO W7900D 上的固定图像处理 workload 为：

```text
GaussianBlur 31×31 → Sobel X → Sobel Y → magnitude
```

4K 实测结论：

| 口径 | 结果 |
|---|---:|
| CPU 算子链 | 150.97 ms |
| GPU 纯计算 | 2.22 ms，68.1× |
| GPU 含上传、预处理、计算和下载 | 23.00 ms，6.6× |
| GaussianBlur 最大像素误差 | 1/255 |
| PSNR | 61.91 dB |

68.1× 只代表输入已经在显存中的 GPU 算子吞吐；6.6× 才包含每轮 host/device 数据
搬运。文章的核心工程结论是：GPU pipeline 应尽量让输入与中间结果常驻显存，而不是
围绕单个算子反复 upload/download。

本文使用 OpenCV 5 HIP 开发分支，并非 stock OpenCV release 默认提供的 Radeon
binary。性能结果绑定固定硬件、构建和 workload，不能外推到任意 OpenCV 算子或 GPU。

## 唯一配套项目

```text
/home/zihaomu/bigssd/notebook/opencv_amd_gpu
├── opencv_amd_gpu_demo.ipynb
├── bench_cv_gpu.cpp
├── gaussian_correctness.cpp
├── data/Bengal_tiger_small.jpg
└── correctness_output/
```

`opencv_amd_gpu_demo.ipynb` 是最终可执行入口，已经保存性能和正确性输出。博客中的
68.1×、6.6×、61.91 dB 等数据均可在该 notebook 中核对。

## Radeon Cloud 复现

- [OpenCV 社区专属 AMD 开发者注册链接](https://developer.amd.com.cn/login?source=mEi9fAoWW)
