# Ultralytics YOLO26 双镜像发布与 CI 就绪修正计划

> 工作目录：`/home/zihaomu/bigssd/notebook/ultralytics_yolo26`
>
> 文档状态：Phase 0-4 已实施并通过本地静态/GPU smoke；新镜像重建与 ACR 发布待精确标签批准
>
> 目标平台：Linux/amd64、AMD Radeon PRO W7900D (`gfx1100`)
>
> 制定日期：2026-09-07

## 1. 目标与完成定义

本计划解决以下五个交付问题：

1. 将 pipeline/Jupyter 镜像和 llama.cpp companion 镜像定义为一个不可拆分、可追溯的发布单元。
2. launcher 在新机器上默认从 ACR 拉取固定 digest，而不是依赖本地短标签或隐式现场构建。
3. 镜像只能从干净 Git commit 构建，OCI revision、源码 bundle 和发布记录必须能相互校验。
4. 修正文档中“full 镜像无需下载模型”与“Notebook 会下载模型”的冲突，并明确双容器边界。
5. 增加无 GPU 静态 smoke gate 和 W7900D GPU release gate。

完成后，“ready”必须同时满足：

- 新机器只需检出仓库、登录所需 registry 并运行 launcher；不需要宿主模型目录。
- launcher 在创建 volume、network 或 container 前成功解析并拉取两个 digest-pinned 镜像。
- pipeline 和 llama.cpp 容器均携带同一个 release ID，并在镜像或模型集不匹配时自动重建。
- full 路径不访问模型下载地址；模型只从 pipeline 镜像原子初始化到共享 named volume。
- 发布镜像的 `org.opencontainers.image.revision` 对应一个干净且可检出的 Git commit。
- CI 能阻止短标签、`:latest`、脏源码构建、Notebook 错误输出和 GPU 核心路径回归进入发布。

## 2. 当前已验证基线

### 2.1 Pipeline/Jupyter 镜像

当前发布标签：

```text
crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/
muzihao2/work:ultralytics-yolo26-workshop-full_2026_09_04
```

当前 OCI index digest：

```text
sha256:6a1a701553902c2041ac43ef89e0362bad7269973c3c9a474e75b22f3dab6b2e
```

Linux/amd64 manifest digest：

```text
sha256:3dced057e8a82f3fbc58231b3460b56cbbc9f607171b7cd21e618fc64b636299
```

镜像内已经包含：

- 两本已执行 Notebook 和完整 workshop 源码。
- `yolo26x.pt`、`yolo26x.onnx`。
- `Qwen3-VL-8B-Instruct-Q8_0.gguf`、`mmproj-F16.gguf`。
- W7900D/gfx1100 对应的 ORT MIGraphX `.mxr` cache。
- Ultralytics 8.4.75、ORT MIGraphX 1.24.2、OpenCV HIP、rocDecode 和 HIP/VA-API bridge。

### 2.2 llama.cpp companion 镜像

当前可拉取标签：

```text
crpi-ygzb1jbfyj9pjrm6.cn-shenzhen.personal.cr.aliyuncs.com/
image_hana/unsloth_llama.cpp:latest
```

当前 OCI index digest：

```text
sha256:b43835f2f07a8ad01f2d8532c3968053b8debfed23e8e7ab85d97931e23c47f2
```

Linux/amd64 manifest digest：

```text
sha256:f995df5028d249d1c97e3f47f68d6172b32ce92d150c21e8fa9c9362106a45a1
```

注意：`latest` 只是可读标签。launcher 和 release lock 必须使用
`@sha256:b438...`，不能继续使用可变标签。

### 2.3 已通过的实机门

- `--network none --read-only` 的新容器 image validator 通过。
- 四个模型文件、identity 和 MXR cache 的 SHA-256 校验通过。
- rocDecode GPU frame、OpenCV HIP、GPU NMS、MIGraphX FP16 I/O Binding 通过。
- MIGraphX 10 次 warm inference：约 `6.9 ms/run`。
- Ultralytics predict 与 production path 最低 class-matched IoU：`0.9275`。
- 12 帧 HIP overlay + direct VA-API：submitted/encoded/packet/sample 均为 12。
- Qwen3-VL 真实 completion 请求成功，不只是 health check。
- 两本 Notebook 分别为 8/8 和 9/9 代码单元已执行，保存输出中无 error。
- 两个正式视频均为 393 个可解码 H.264 frame，时长 15.72 秒。

这些结果是后续修正必须保持的 baseline，不需要重新设计推理核心。

## 3. 当前问题与根因

### 3.1 双镜像关系没有机器可读的 release lock

`scripts/start_notebook_container.sh` 当前默认使用：

```text
PIPELINE_IMAGE=zihao/ultralytics-yolo26-workshop:rocm7.2.1-full
LLAMA_IMAGE=zihao/llamacpp-q8:b9766-rocm
```

这两个本地短标签无法在新机器上解析，也没有文件声明“哪个 pipeline digest 必须配哪个
llama.cpp digest”。已有 llama container 的复用判断只检查 model volume 和 model set，
没有检查 companion image ID。

### 3.2 launcher 的缺镜像行为不适合发布使用

pipeline 镜像不存在时，launcher 会直接调用 `build_notebook_image.sh`。但构建脚本要求
四个被 Git 忽略的大模型、MXR cache、ORT wheel 和 Ultralytics checkout 已经存在或可
联网取得。全新 clone 因而不是可靠的一键路径。

launcher 也没有在产生 Docker side effects 前预拉并验证 companion 镜像，没有等待
llama.cpp `/health` 和 `/v1/models` 后再宣布整体 ready。

### 3.3 Git revision 与实际 build context 可能不一致

`build_notebook_image.sh` 使用 `git rev-parse HEAD` 写 OCI revision，却没有拒绝 tracked
dirty state。当前工作树中仍有一份 tracked 设计文档处于删除状态：

```text
doc/ultralytics_yolo26_workshop_改造计划.md
```

因此现有镜像虽然功能正确、bundle SHA 可校验，但其 OCI revision 不能完整表达当时的
工作树内容。

### 3.4 文档对模型行为的描述冲突

`README_CN.md` 和 `README.md` 的模型章节仍写“第一个 Notebook 单元会下载模型”，后面的
镜像章节又写 full 镜像不需要下载。实际逻辑是：

- full 镜像路径：`ensure_models()` 只做存在性、大小和 SHA 校验，不下载。
- 源码开发/非 full 镜像 fallback：文件缺失时才使用可恢复下载器。
- Qwen 权重在 pipeline 镜像中，但 `llama-server` 二进制在 companion 镜像中。

### 3.5 没有 CI workflow

仓库当前不存在 `.github/workflows`。现有 validators 很强，但依赖人工运行；普通
`pytest` 还可能被环境中的 shard 插件筛成 0 项，所以 CI 必须调用项目的可执行验证入口，
并把 `0 tests` 视为失败而不是成功。

## 4. 目标发布契约

### 4.1 发布单元

增加机器可读文件：

```text
ultralytics_yolo26/release/current.env
```

建议字段：

```bash
WORKSHOP_RELEASE_ID=ultralytics-yolo26-2026-09-07-r1
WORKSHOP_SOURCE_COMMIT=<clean-build-commit>
WORKSHOP_BUNDLE_SHA256=<bundle-sha256>
WORKSHOP_MODEL_SET_SHA256=a7d82e6cc0cb2d92a96000fb65dc39e954d49e50d2bc028ebdc0dce2d5db82ad
PIPELINE_IMAGE_TAG=<immutable-human-readable-tag>
PIPELINE_IMAGE_REF=<registry/repository>@sha256:<oci-index-digest>
PIPELINE_AMD64_MANIFEST=sha256:<linux-amd64-manifest-digest>
LLAMA_IMAGE_TAG=<immutable-human-readable-tag>
LLAMA_IMAGE_REF=<registry/repository>@sha256:<oci-index-digest>
LLAMA_AMD64_MANIFEST=sha256:<linux-amd64-manifest-digest>
```

规则：

- `*_IMAGE_REF` 是运行时唯一权威值，必须含 `@sha256:`。
- `*_IMAGE_TAG` 只用于人类查找和发布记录，不参与容器复用判断。
- release ID 同时写入两个容器的环境变量和 Docker label。
- 已发布 lock 不允许原地修改；新版本新增归档文件并更新 `current.env`。
- OCI index digest 用于跨 registry 拉取锁定；同时记录 amd64 manifest 便于审计。

### 4.2 新发布标签

现有 `2026_09_04` 标签不得覆盖。修正完成后建议使用新标签，例如：

```text
crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/
muzihao2/work:ultralytics-yolo26-workshop-full_2026_09_07-r1

crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/
muzihao2/work:ultralytics-yolo26-llamacpp-b9766_2026_09_07
```

这两个名称目前只是候选。`muzihao2/work` 是 release-only namespace；在执行 tag/push
前，必须由用户明确批准最终的两个精确标签。

若暂时不把 companion 镜像复制到上海 ACR，则第一版 lock 可以固定现有深圳 ACR digest：

```text
crpi-ygzb1jbfyj9pjrm6.cn-shenzhen.personal.cr.aliyuncs.com/
image_hana/unsloth_llama.cpp@sha256:b43835f2f07a8ad01f2d8532c3968053b8debfed23e8e7ab85d97931e23c47f2
```

但最终建议两个镜像进入同一 registry namespace，减少登录、权限和区域可用性差异。

## 5. 分阶段实施

### Phase 0：冻结当前证据并处理 tracked 删除

涉及文件：

- `doc/ultralytics_yolo26_workshop_改造计划.md`
- `output/pipeline/manifest.json`
- 两本 Notebook

步骤：

1. 明确旧设计文档是保留还是删除。
2. 默认选择恢复该历史设计文档；如果确认已废弃，则单独提交删除并在 commit message 中说明。
3. 不修改当前已验证输出和 Notebook 保存输出。
4. 提交本计划以及后续代码修改，使 tracked worktree 归零。
5. 记录当前 baseline 的 image digest、bundle SHA、model-set SHA 和 GPU 验收结果。

Gate P0：

```bash
git status --porcelain --untracked-files=no -- ultralytics_yolo26
```

输出必须为空。被 `.gitignore` 排除的模型、wheel、build 和 cache 不计为源码污染。

### Phase 1：建立双镜像 release lock

新增：

- `release/current.env`
- `release/README.md`
- `scripts/validate_release_lock.py`

修改：

- `scripts/start_notebook_container.sh`
- `scripts/push_notebook_image.sh`

实现要求：

1. launcher 默认读取 `release/current.env`，环境变量仍可显式覆盖，供本地开发使用。
2. validator 拒绝空 digest、短标签和 `:latest` 运行引用。
3. 在创建 volume/network/container 前，按顺序执行：
   - `docker image inspect "$PIPELINE_IMAGE_REF" || docker pull "$PIPELINE_IMAGE_REF"`
   - `docker image inspect "$LLAMA_IMAGE_REF" || docker pull "$LLAMA_IMAGE_REF"`
   - 校验平台为 `linux/amd64`。
   - 校验 pipeline OCI label 中的 source commit、bundle SHA 和 model-set SHA。
   - 校验 companion 镜像中 `/opt/llama.cpp/build/bin/llama-server` 可执行。
4. 默认路径禁止隐式本地 build。仅当显式设置 `ALLOW_LOCAL_BUILD=1` 时，才调用
   `build_notebook_image.sh`；该模式必须在日志中标记为 development mode。
5. 复用已有 llama container 时，额外比较：
   - 当前 container `.Image` 与解析后的 companion image ID。
   - `WORKSHOP_RELEASE_ID`。
   - `ULTRALYTICS_MODEL_SET_ID`。
   - `/models` volume 名称。
6. 复用 pipeline container 时，除现有检查外再比较 `WORKSHOP_RELEASE_ID`。
7. 两个容器启动后分别等待：
   - Jupyter `/api/status`。
   - llama.cpp `/health`。
   - llama.cpp `/v1/models` 返回目标 GGUF 且具备 multimodal capability。
8. 任一步失败时输出对应容器最近日志并返回非零，不打印“ready”。

Gate P1：在删除本地短标签、使用全新 named volume 和新容器名的情况下，launcher 能仅靠
两个 digest-pinned registry 引用启动；模型初始化期间不访问 Hugging Face 或其他模型源。

### Phase 2：强制干净 Git provenance

修改：

- `scripts/build_notebook_image.sh`
- `scripts/push_notebook_image.sh`
- `docker/Dockerfile`
- `docker/validate-image.py`

构建前硬失败条件：

```bash
git diff --quiet -- ultralytics_yolo26
git diff --cached --quiet -- ultralytics_yolo26
```

实现要求：

1. build script 在任何 tracked 修改、删除或 staged 修改存在时退出。
2. `WORKSHOP_GIT_COMMIT` 只能来自 clean `HEAD`。
3. 构建开始前计算 bundle SHA；构建结束后在镜像内重新计算，二者必须一致。
4. image validator 验证以下 OCI labels 非空且与文件内容一致：
   - `org.opencontainers.image.revision`
   - `io.ultralytics.workshop.bundle.sha256`
   - `io.ultralytics.model-set.sha256`
   - `io.ultralytics.release.id`
   - `io.ultralytics.companion.digest`
5. push script 拒绝：
   - dirty worktree。
   - 未通过 `test_notebook_image.sh` 的本地 image ID。
   - 目标 tag 已存在且 digest 不同。
   - 未显式提供 release tag 的调用。
6. 生成发布记录时保存 `docker buildx imagetools inspect` 的 OCI index digest 和
   linux/amd64 manifest digest。

#### 避免 commit/digest 循环依赖

采用两次 commit：

1. **Build commit**：包含 launcher、validator、文档和 CI 修改；worktree 清洁后构建。
2. 构建、GPU 验收、经用户批准后推送两个新 tag，取得最终 digest。
3. **Release-lock commit**：只写入最终 digest 和 build commit，不重新声称它是镜像源码
   commit。

镜像 OCI revision 指向 build commit；release lock commit 是对该镜像的发布登记。这样两者
职责明确，不需要用包含自身 digest 的 commit 构建自身。

Gate P2：从 release lock 给出的 source commit 建立临时 clean worktree，计算出的 bundle
SHA 与远端镜像 label 完全一致。

### Phase 3：修正文档与 Notebook 表述

修改：

- `README.md`
- `README_CN.md`
- `models/README.md`
- `scripts/build_notebooks.py`
- 由 generator 生成的两本 Notebook

统一表述：

> Full release 模式下，`ensure_models()` 校验镜像内固定模型，不进行下载。下载器只服务于
> 源码开发或非 full fallback，不属于 workshop 的默认执行路径。

必须新增的说明：

- 完整交付是一个双镜像 release，而不是一个包含全部 runtime 的单容器。
- pipeline 镜像保存唯一的模型层；launcher 将其按 SHA 原子初始化到 named volume，供
  companion 容器只读挂载。
- `HF_ENDPOINT` 在 full release 模式下不会用于模型获取，仅保留开发 fallback 兼容性。
- 默认 ACR digest、认证前提、GPU/render node 选择、启动、停止和故障日志命令。
- W7900D/gfx1100 是已验证目标；预编译 MXR 不承诺跨 GPU/ROCm/MIGraphX 版本复用。
- human-readable tag 与实际运行 digest 的区别。

Notebook 要求：

- 第一个模型准备单元输出 `verified baked model`，不能笼统显示 `downloading`。
- 所有已有 cell 保留唯一 `metadata.id` 和 `metadata.language`。
- clean-kernel 重新执行后，代码单元 execution count 完整且 error output 为 0。
- generator 与已执行 Notebook 的职责写清楚：generator 生成 source，发布 Notebook 保存经
  验证输出；CI 不应把“清空输出后的生成结果”与“已执行发布文件”做字节相等比较。

Gate P3：中英文 README、模型文档和两本 Notebook 中，不再存在“full 默认实时下载模型”或
“单镜像完整运行”的陈述。

### Phase 4：增加 CI smoke gate

新增：

```text
.github/workflows/ultralytics-yolo26-smoke.yml
ultralytics_yolo26/scripts/ci_static_smoke.py
ultralytics_yolo26/scripts/ci_gpu_smoke.sh
```

#### Job A：静态门

运行环境：`ubuntu-latest`。

触发：

- `pull_request`，限定 `ultralytics_yolo26/**` 和 workflow 自身路径。
- `push` 到 `main`，使用相同 path filter。
- `workflow_dispatch`。

检查项：

1. `bash -n scripts/*.sh docker/*.sh`。
2. Python 文件 `compileall` 或 AST syntax check，不导入 ROCm 依赖。
3. 两本 Notebook 是合法 JSON；每个已有 cell 均有 `metadata.id` 和
   `metadata.language`；保存输出中不存在 `output_type=error`。
4. release lock schema、digest 格式和 release ID 一致。
5. launcher 默认值来自 release lock，禁止 `zihao/...`、`:latest` 和裸 tag。
6. README 中的 release ID、两个镜像 tag/digest 与 lock 一致。
7. build/push script 包含 clean-worktree guard。
8. 所有 smoke 命令若执行项为 0，必须失败；不依赖外部 pytest shard 配置。

#### Job B：W7900D GPU release gate

运行环境建议：`[self-hosted, Linux, X64, rocm, w7900]`。启用前先确认当前仓库确实注册了
这些 runner labels。

安全策略：

- 不在自托管 GPU runner 上直接执行不受信任 fork 的 PR 代码。
- 默认仅 `workflow_dispatch` 和受保护 `main` push 可运行 GPU job。
- registry 密码仅通过 GitHub Secrets 和 `docker login --password-stdin` 使用。
- 设置 workflow `concurrency`，同一台 W7900D 同时只跑一个 release smoke。

隔离参数：

```text
MODEL_VOLUME=ultralytics_yolo26_ci_${GITHUB_RUN_ID}
NETWORK=ultralytics_yolo26_ci_${GITHUB_RUN_ID}
PIPELINE_CONTAINER=ultralytics_yolo26_ci_notebook_${GITHUB_RUN_ID}
LLAMA_CONTAINER=ultralytics_yolo26_ci_llama_${GITHUB_RUN_ID}
OUTPUT_DIR=<runner-temp>/ultralytics_yolo26
JUPYTER_PORT=<reserved-ci-port>
LLAMA_PORT=<reserved-ci-port>
```

使用 `trap` 保证成功或失败后都删除 CI container、network 和 volume，不影响常驻 workshop。

GPU 检查顺序：

1. 按 release lock 拉取两个 digest。
2. 运行 `scripts/test_notebook_image.sh`：无网络、只读 rootfs、无源码/模型挂载。
3. 用全新 named volume 启动双容器。
4. 校验 Jupyter、llama `/health` 和 `/v1/models`。
5. 发出一次真实 Qwen completion，要求非空内容和正常 finish reason。
6. 运行 predict/production parity。
7. 运行 12 帧 GPU overlay/direct VA-API smoke，严格检查 frame/packet/sample 计数。
8. 运行 `scripts/run_pipeline.py --validate-only`，检查正式 manifest 和 393 帧产物。
9. 上传 JSON、容器日志和 smoke 视频；不上传模型文件。

建议 timeout 为 30 分钟。任何 fallback 到 CPU、任何完整帧 D2H、模型 SHA 不一致、Qwen
未 ready 或编码帧数不一致都必须硬失败。

#### Job C：发布门

发布不应由普通 push 自动触发。使用受保护的 `workflow_dispatch` 或人工步骤：

1. 要求 Job A 和 Job B 对同一个 build commit 成功。
2. 输入并回显待发布的两个精确 tag，但不回显 registry secret。
3. 检查远端 tag 不会被覆盖。
4. 等待用户明确批准两个 release-only ACR tag。
5. 推送后读取远端 digest、写 release lock，并再次执行 digest-pinned GPU smoke。

Gate P4：branch/release 规则要求对应 smoke 状态成功，且 CI 日志中能看到非零测试数量、
两个 immutable image refs 和最终 release ID。

### Phase 5：重建、发布和 fresh-host 验收

执行顺序：

1. 合入 Phase 0-4，得到 clean build commit。
2. 从该 commit 运行静态门和本地 GPU 门。
3. 构建新的 pipeline image；不得重打现有 `2026_09_04` 标签。
4. 对 companion image 做二进制路径、ROCm 架构和真实 Qwen 请求验证。
5. 用户批准两个精确 ACR tag 后，才允许 tag/push。
6. 取得远端 OCI index digest 与 amd64 manifest digest。
7. 提交 immutable release lock。
8. 模拟新机器：删除本地 release 镜像短标签，使用全新 volume/network/container 名称，执行
   launcher。
9. 临时阻断 HF/model 下载域名，证明启动只访问 registry，不访问模型源。
10. 完成双容器健康检查、真实 Qwen 请求、YOLO parity、direct encode 和 manifest 验收。

最终发布报告至少记录：

- release ID。
- build commit 和 release-lock commit。
- 两个 tag、OCI index digest、linux/amd64 manifest digest。
- bundle SHA 和 model-set SHA。
- GPU、ROCm、ORT、MIGraphX、Ultralytics 和 llama.cpp build 信息。
- CI run URL 与所有 gate 结果。

## 6. 文件级改动清单

| 文件 | 计划改动 |
|---|---|
| `release/current.env` | 新增双镜像 immutable refs、release ID、commit 和 SHA |
| `release/README.md` | 说明 release lock 更新规则和禁止覆盖策略 |
| `scripts/start_notebook_container.sh` | 读取 lock、默认 pull ACR、双镜像 preflight、image/release identity 重建、双服务 readiness |
| `scripts/build_notebook_image.sh` | clean-tree guard、release labels、build 后 bundle 校验 |
| `scripts/push_notebook_image.sh` | 显式 tag、远端防覆盖、测试证明和 digest 采集 |
| `scripts/validate_release_lock.py` | 校验 lock schema、digest、labels、平台和文档一致性 |
| `scripts/ci_static_smoke.py` | Python/Notebook/release/docs 静态检查 |
| `scripts/ci_gpu_smoke.sh` | 隔离的双容器 GPU smoke 和清理 trap |
| `docker/validate-image.py` | 增加 release/companion provenance 校验 |
| `docker/Dockerfile` | 增加 release ID 和 companion digest OCI labels |
| `README.md`、`README_CN.md` | 修正下载语义，补双镜像 digest-pinned fresh-host 流程 |
| `models/README.md` | 区分 baked verify 与 development fallback download |
| `scripts/build_notebooks.py` | 更新 Notebook 中的模型准备说明和输出措辞 |
| 两本 `.ipynb` | 由 generator 更新并 clean-kernel 重跑 |
| `.github/workflows/ultralytics-yolo26-smoke.yml` | 新增静态和 W7900D GPU gates |

## 7. 最终验收矩阵

| 场景 | 通过条件 |
|---|---|
| 源码 | tracked worktree clean；build commit 可从远端检出 |
| Release lock | 两个运行引用均含 `@sha256:`；无 `latest` 或本地短标签 |
| Fresh host | 无宿主模型目录、无本地镜像时自动 pull 两个 ACR refs 并启动 |
| 离线模型 | 启动过程中无 HF/模型源请求；模型从 image 初始化到 volume |
| Pipeline image | 无网络、只读 rootfs validator 通过 |
| YOLO | MIGraphX 为第一 provider、FP16、GPU I/O Binding、parity IoU >= 0.90 |
| Encode | rocDecode GPU frame；direct VA-API；submitted=encoded=packet=sample |
| Qwen | `/health`、`/v1/models` 和真实 completion 全部通过 |
| Notebook | 8/8、9/9 code cells 执行，0 error，metadata 完整 |
| 正式产物 | YOLO/final 视频各 393 个可解码帧；manifest status PASS |
| Provenance | image revision、bundle SHA、model-set SHA 与 release lock 一致 |
| CI | 静态门和 W7900D release gate 对同一 build commit 成功 |

## 8. 回滚策略

- 保留当前 `2026_09_04` pipeline digest，不覆盖、不删除。
- 保留当前深圳 ACR companion digest，直到新 release pair 完整通过 fresh-host gate。
- launcher 支持通过环境变量显式传入旧的两个 digest，用于紧急回滚。
- release lock 更新单独提交；回滚时 revert lock commit 即可，不需要重建模型层。
- named volume 以 model-set ID 判定内容；回滚到不同 model set 时必须使用新 volume 或由初始化器
  完整替换，禁止混用文件。
- CI 失败只清理本次 run 带 ID 的资源，不停止当前 `ultralytics_yolo26_notebook` 和
  `ultralytics_yolo26_llamacpp` 常驻服务。

## 9. 推荐实施顺序

1. 处理旧设计文档删除状态，并提交本计划。
2. 实现 release lock validator 和 launcher ACR-first 行为。
3. 增加 clean provenance guards。
4. 修正 README、模型文档和 Notebook 文案，重新执行 Notebook。
5. 增加静态 CI，再接入受保护的 W7900D GPU gate。
6. 在本地完成 fresh-volume 双容器验收。
7. 提交 clean build commit。
8. 请求用户批准两个精确 release-only ACR tag。
9. 重建、推送、记录 digest、提交 release lock。
10. 对远端 digest 执行最终 fresh-host GPU smoke 并发布验收报告。

在第 8 步获得明确批准之前，不对
`crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work`
执行任何新 tag 或 push。

## 10. 2026-09-07 实施记录

已完成：

- 恢复发布 commit 中的旧设计文档，消除既有 tracked deletion。
- 新增 bootstrap `release/current.env` 及同名归档，固定已验证的双镜像 OCI digest。
- launcher 改为 release-lock 驱动、ACR-first、双镜像 preflight、container image/release/model-set 身份校验，并等待 Jupyter 和 llama.cpp 双服务 ready。
- full/baked 模式改为严格离线模型校验；缺失或损坏时禁止下载。源码开发模式保留断点续传 fallback。
- build/push 增加 tracked-clean、显式 release ID、精确标签批准、防覆盖和 provenance 门禁。
- 新镜像定义增加 release ID、source commit 和 companion digest 元数据。
- 中英文 README、模型说明和两本 Notebook 已统一 baked/offline 与双镜像语义。
- 两本 Notebook 已在 W7900D 上 clean-kernel 重跑，分别 8/8、9/9 代码单元成功。
- 新增静态 smoke、隔离 GPU smoke 和 GitHub Actions workflow。
- 本地隔离 GPU smoke 已通过：immutable 双镜像、真实 Qwen completion、YOLO parity、12 帧 direct VA-API 和 393 帧 manifest 均通过。

- 两阶段发布边界已修正：`current.env` 和历史 lock 仅作为宿主 Git 发布登记，不复制进镜像、也不参与 bundle SHA；镜像只保留稳定的 `release/README.md` 规则以及 source/release/companion OCI 身份，消除 pipeline digest 自引用。

待发布步骤：

1. 提交当前源码形成 clean build commit。
2. 注册或指定带 `rocm,w7900` labels 的该仓库 self-hosted runner；当前 GitHub API 返回 runner 数量为 0。
3. 明确批准两个新的 release-only ACR 精确标签。
4. 从 clean build commit 重建 pipeline 镜像，运行 GPU gate 后推送。
5. 将新远端 digest 写入新的 immutable release lock，并执行最终 fresh-host 验收。

在完成上述步骤前，`release/current.env` 保持指向已验证的 2026-09-04 bootstrap release pair。
