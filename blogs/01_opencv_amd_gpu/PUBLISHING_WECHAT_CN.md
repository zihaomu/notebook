# OpenCV 公众号发布说明（最终版）

## 推荐主标题

**OpenCV 用户，多一个 GPU 选择：cv::cuda 跑上 Radeon，还能免费复现**

最终版同时强调工程结论与低门槛复现：没有本地 Radeon，也可以通过 OpenCV 社区专属链接注册 AMD 开发者、领取本次活动提供的免费 48GB 显存 Radeon 云算力，并直接运行同一本 notebook。

## 备选标题

1. **没有 Radeon 也能跑：AMD 提供免费 48GB 云算力，复现 OpenCV GPU 加速**
2. **OpenCV 的 cv::cuda 跑上 Radeon：4K 最高 68.1×，还可免费上云复现**
3. **一键启动 48GB Radeon 云 GPU，亲自验证 OpenCV 5 的 HIP 加速**

标题 2 信息最完整，但必须同时保留“68.1× 为 GPU 纯计算、6.6× 为含传输”的边界。标题 1 更适合活动推广；推荐主标题更符合 OpenCV 公众号的社区语气。

## 推荐摘要

OpenCV 社区正在推进 ROCm/HIP 支持：保留开发者熟悉的 `cv::cuda` API，让 GPU 模块在 AMD Radeon 上执行。Radeon PRO W7900D 实测中，4K 算子链纯计算提速 68.1 倍，计入上传和下载后仍有 6.6 倍，GaussianBlur 最大误差仅 1/255。没有本地 Radeon 也可以参与：通过 OpenCV 社区专属链接注册 AMD 开发者后，即可领取活动提供的免费 48GB 显存 Radeon 云算力，在 Gallery 找到 `opencv_on_amd` 并启动实验。

## 云端复现入口

1. OpenCV 社区专属 AMD 开发者注册链接：https://developer.amd.com.cn/login?source=mEi9fAoWW
2. Radeon Cloud 短入口：https://radeon.anruicloud.com/
3. 当前 Gallery 页面：https://developer.amd.com.cn/radeon/
4. `opencv_on_amd` 公开预览：https://developer.amd.com.cn/radeon/templates/2298/preview

推荐把**专属注册链接**设置为公众号“阅读原文”链接，因为用户需要先注册 AMD 开发者并领取活动权益。正文中同时保留 `radeon.anruicloud.com`，供完成注册的用户进入 Gallery。若制作二维码，优先指向专属注册链接。

经实际页面核验：

1. 专属链接可访问 AMD 开发者登录/注册页，页面提供“注册账号”。
2. Gallery 中存在 `opencv_on_amd` 卡片。
3. 卡片指向 `github.com/zihaomu/notebook · opencv_amd_gpu/opencv_amd_gpu_demo.ipynb`。
4. `Preview` 可公开查看完整 notebook 和保存输出。
5. 未登录点击 `Launch` 会提示 `Please login first`。
6. 登录后 Launch 前端请求 1 张 GPU；实例就绪后出现 `Open Notebook`。

## 封面与分享图

最终版推荐：

- 横版封面：`assets/wechat_cover_cn.jpg`，900×383
- 方形分享图：`assets/wechat_thumbnail_cn.jpg`，500×500
- 云端四步引导图：`assets/wechat_radeon_cloud_steps_cn.png`，1080×1770

最终封面把“免费复现”和“48GB 显存”放在主视觉中，性能结果由正文数据卡承接。

## 正文配图顺序

1. `assets/wechat_cover_cn.jpg`：文章开头，可按公众号模板决定是否重复展示。
2. `assets/wechat_radeon_cloud_steps_cn.png`：导语后，展示专属注册、领取权益、Gallery、Launch 和 Run All。
3. `assets/wechat_results_cn.png`：核心性能与正确性数据卡。
4. `assets/wechat_hip_stack_cn.png`：解释 `cv::cuda` 与 HIP/ROCm 的关系。
5. `assets/performance_benchmark.png`：完整三分辨率结果。
6. `assets/correctness_comparison.jpg`：CPU/GPU 正确性验证。

云端引导图是依据实际页面卡片和交互生成的高分辨率示意图，不应标注为网页截图。图中已经注明实际界面与活动规则以页面为准。

## 编辑时不要删除的活动边界

1. 用户需要先通过 **OpenCV 社区专属链接注册 AMD 开发者**，再按活动页面提示领取权益。
2. 免费 48GB 显存云算力是**本次活动提供的权益**，不要改成长期、永久或无条件免费。
3. 注册、领取方式、额度、排队情况、实例时长和资源可用性以注册页与 Radeon Cloud 页面实时规则为准。
4. `Preview` 可以公开查看；实际运行需要完成注册/登录后 `Launch`。
5. 云端模板对应本文使用的同一本 notebook，但云实例负载变化可能导致性能数字与文章记录略有波动。

## 编辑时不要删除的技术边界

1. **68.1× 是 4K GPU 纯计算结果**，输入已经在显存中。
2. **6.6× 是 4K 含 upload、GPU 预处理、计算和 download 的结果**。
3. CPU 计时没有包含其最初的灰度/FP32 转换，因此含传输比较对 GPU 偏保守，但不是完全对称的端到端比较。
4. 结果对应固定 workload、机器和构建，不代表任意 OpenCV 算子、CPU 或 Radeon 型号。
5. 使用的是 OpenCV 5 HIP 开发分支；相关 PR 截至 2026-08-08 仍处于 Open/review，不是 stock OpenCV release 默认提供的 Radeon binary。
6. 少量 NVIDIA 专属入口没有 ROCm 对等实现，会返回 `StsNotImplemented`。

## 公众号排版建议

- 首屏顺序：导语 → 专属注册/领取免费 48GB 云算力 → 四步引导图 → 4K 双口径结果。
- 把专属注册链接做成高亮按钮或“阅读原文”；同时将短域名 `radeon.anruicloud.com` 单独成行，供注册后进入 Gallery。
- 正文表格之外保留 `wechat_results_cn.png`，保证移动端数据可读。
- 代码块保留三段：熟悉的 `cv::cuda` API、最短 CMake、设备发现检查。
- 末尾设置三段 CTA：先用专属链接注册并领取权益；再到云端 Run All；最后到 PR 提交不同硬件/算法的测试反馈。

## 署名建议

技术工作来源至少应体现：

- Jeff Daily：OpenCV 4.x ROCm/HIP 原始移植
- zhangnju：OpenCV 5.x 移植、兼容性修复及测试/应用验证
- AMD Radeon Cloud：本次活动的免费 48GB 显存云算力与可执行 notebook 环境
- OpenCV / AMD ROCm 社区：review、测试与持续集成工作

最终署名方式请根据公众号编辑规范和实际作者列表确认。

## 发布前人工确认

由于免费权益可能随活动运营变化，发布当天请同时打开专属注册链接和 `radeon.anruicloud.com` 做一次人工确认：

- 专属注册链接 `https://developer.amd.com.cn/login?source=mEi9fAoWW` 可访问，且仍保留注册入口
- 注册后可按页面提示领取活动权益
- `opencv_on_amd` 卡片仍在 Gallery 中可见
- Preview 正常
- 登录后 Launch 可用
- 页面显示的免费额度与 48GB 实例权益仍与稿件一致

若权益发生变化，请更新最终稿的导语、云端章节、封面和发布摘要；配套项目代码与实验结果保持独立。
