---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 25px; justify-content: flex-start; }
  h1 { color: #b30000; }
  h2 { color: #b30000; border-bottom: 2px solid #b30000; padding-bottom: 6px; }
  table { font-size: 21px; }
  code { font-size: 21px; }
  strong { color: #b30000; }
  blockquote { font-size: 21px; color: #555; }
---

<!-- _paginate: false -->
<!-- _class: lead -->
# seekdb × hipVS
## AMD GPU 向量检索加速

**让 seekdb 向量索引可选跑在 AMD GPU(gfx1100 / RDNA3)—— 后端 hipVS(cuVS 移植)**

> 默认构建零影响 · GPU 按索引 opt-in · 诚实口径:单查无收益,**批量**决定性胜出

2026-08 · 分支 `feature/hipvs-cuvs`(实验 / RFC)

<!-- 开场:今天讲我们怎么把 AMD GPU 接进 seekdb 的向量检索,以及一个被数据反复修正的诚实结论——GPU 只在批量场景赢。请全场记住这一句。 -->

---

## 1. 背景:为什么要 GPU 向量检索

- 向量检索(ANN)= RAG / 语义搜索 / 推荐 / 去重 的**基础算子**;做进 SQL 才能存算一体
- seekdb(OceanBase 血统)默认后端 **VSAG(CPU HNSW)**,单查已亚毫秒 —— 基线不弱
- GPU 的机会:海量向量的**批量**相似度计算(相似度 JOIN / 批量打分)是天然并行负载
- AMD 生态缺 cuVS → **hipVS**(ROCm 上的 cuVS 移植)补位
- **本次核心**:把二者接起来,并**如实测量**收益边界与工程代价

<!-- 强调 CPU 基线不弱,所以 GPU 证明自己不容易,这也是后面诚实结论的前提。 -->

---

## 2. 技术栈:五层解耦

| 层 | 组件 | 角色 |
|---|---|---|
| SQL / 存储 | **seekdb** | 向量列、向量索引、APPROXIMATE 查询 |
| 向量抽象 | **obvsag C-API** | 后端选择缝:vsag(CPU)/ cuvs(GPU) |
| GPU 桥接 | **libseekdb_cuvs_bridge.so** | 纯 C shim,隐藏 ROCm/cuVS 头 |
| GPU 算法 | **hipVS**(libcuvs) | CAGRA 图索引 + 暴力 KNN |
| 运行时 | **ROCm 7.2.4** | AMD gfx1100 / RDNA3 |

> 硬件:Radeon PRO W7900。官方 cuVS 仅支持 CDNA(gfx90a/942/950),gfx1100 **源码编译 hipVS**,已全量测试通过。

<!-- 桥接层是纯 C,刻意隐藏 GPU 细节,seekdb 主体不碰 ROCm。 -->

---

## 3. 架构与集成缝

```text
 SQL 引擎 / 存储层
      │  ObPluginVectorIndexAdaptor:incr(delta) + snap 两级,合并
      ▼
 obvsag C-API (ob_vsag_adaptor)   ── 后端选择 ──► vsag(CPU HNSW)
      │                                          └► cuvs(GPU) …↓
      ▼  thin C shim(隐藏 ROCm/cuVS 头)
 libseekdb_cuvs_bridge.so  ──►  hipVS libcuvs_c (CAGRA)  ──►  ROCm / gfx1100
```

- 向量索引**两级**:delta(增量,始终新)+ snapshot(快照),查询各查一份再合并
- **关键洞察**:普通 HNSW **从不调 build_index** → GPU 钩子挂在 **add_index**(缓存向量)+ **knn_search**(惰性建 CAGRA 并服务)

<!-- 这页讲清楚我们改在哪、为什么改在 add_index 和 knn_search 而不是 build_index。 -->

---

## 4. 两种用法(用户面)

**A. 声明式 per-index —— 只加一个 `lib=cuvs`**
```sql
VECTOR INDEX idx(c2) WITH (distance=l2, type=hnsw, lib=cuvs, ...)
```
> 该索引走 GPU,同 server 其它索引仍走 CPU,**无需任何环境变量**。

**B. 批量 ANN 算子 —— N 条探针一次 GPU 调用**
```sql
CALL dbms_vector.batch_knn('index_tbl', 'probe_tbl', 10, 'out_tbl');
```
> 这是 GPU **真正发力**的入口(相似度 JOIN / 批量打分)。

改回 `lib=vsag` 或关掉 GPU → 结果完全一致,可回退可对照。

<!-- 只展示两个一行核心用法,细节在核心文档里。 -->

---

## 5. 关键发现:单查无收益,批量制胜 ⭐

**一个被数据反复修正的诚实结论:**

- 假设 GPU 更快 → 端到端一测:**单查 cuVS 反而慢 2.25×**
- 诊断:慢在**未压缩 delta 的近线性全扫**,不是 ANN 本身
- 压缩出快照后 APPROX **提速 37×**,VSAG 快照 **0.4ms/查**
- 结论:**单查 GPU 追不上亚毫秒 CPU HNSW**(每调用固定开销:显存分配 + PCIe + kernel 启动)
- 翻盘:多条探针**合并成一次 GPU 调用** → 固定开销摊薄 → **116×~263×**

<!-- 全场最重要一页。诚实地讲我们怎么从一个错误假设一步步走到正确结论。 -->

---

## 6. 性能①:单查 vs 批量 —— 一张表讲完

| 路径 | 吞吐 | recall@10 | 说明 |
|---|---|---|---|
| VSAG-CPU 逐条(10k) | 4,598 q/s | 0.926 | 最优单查基线 |
| cuVS-GPU 逐条(10k) | 470 q/s | 0.873 | 更慢(每调用开销) |
| **cuVS-GPU 批量(nq=100)** | **54,777 q/s** | 0.873 | **比逐条 116× / 比最优 CPU 12×** |
| 桥接批量 nq=5000 | 634,930 q/s | — | **263×**,GPU 饱和 |

> 批量红利来源 = **摊薄每次调用的固定开销**;单查永远追不上。

<!-- 一张表把故事讲完:单查没戏,批量指数级拉开。 -->

---

## 7. 性能②:端到端 SQL 演示(10k 索引)

| N(批量) | CPU 墙钟 ms | GPU 墙钟 ms | 加速比 |
|---|---|---|---|
| 100 | 373 | 551 | 0.68× |
| 500 | 1,545 | 544 | 2.84× |
| 1000 | 3,023 | 630 | 4.8× |
| 2000 | 5,923 | 757 | 7.82× |
| 4000 | 10,036 | 832 | **12.07×** |

> CPU 墙钟随 N **线性**增长;GPU 近乎持平。交叉点约 **N≈300**。

<!-- 这是 notebook 真实跑出来的端到端数,不是微基准。 -->

---

## 8. 性能③:批量提速曲线

![w:760](../notebook_workspace/results/seekdb_hipvs_speedup.png)

> 左:墙钟(CPU 线性 vs GPU 持平)· 中:加速比 · 右:每探针成本(GPU 随 N 迅速摊薄)

<!-- 三张子图配合上一页的表,视觉冲击力最强。 -->

---

## 9. 正确性与安全回退(正确性 > 加速)

| 场景 | 处理 | 验证 |
|---|---|---|
| 无过滤 | GPU 服务 | APPROX == EXACT |
| `WHERE` 过滤 | 后置校验,任一被排除即**回退** | 结果正确 |
| `DELETE` 后 | 回退(CAGRA 静态) | 删除行不再出现 |
| 非 L2(cos/IP) | 不服务、全回退 | 结果正确 |
| 新鲜度 | 门控 `built_n==buffer` 才服务 | 四步 APPROX==EXACT |
| 规模召回(10k) | GPU 服务 | recall@10 = **0.89** |

> 天然落地:**delta → VSAG(始终新);snapshot → cuVS**。

<!-- 强调:任何不确定场景一律回退,绝不给错答案。 -->

---

## 10. 发布硬化:可选 · 受支持 · 零默认影响

- **编译开关** `OB_BUILD_CUVS`(+ 桥接路径 + 开发者 tracer)
- `OB_BUILD_CUVS=OFF`(**默认**):桥接不链接、GPU 代码从不运行、**行为与上游 VSAG 完全一致**
- **DDL**:正式接受 `lib=cuvs`(dense L2 HNSW),IVF 拒绝
- **插件路由**:标记 `lib=cuvs` 句柄走 GPU(无环境变量)
- **去 PoC 化**:去硬编码路径、移除全局 env 与调试 tracer、补 CI 契约测试

<!-- 让评审放心:默认不变、不劣化、可关。 -->

---

## 11. 镜像固化 + Notebook 一键演示

- 自包含镜像**已推 ACR**:`…/muzihao2/work:seekdb-hipvs-nb-demo`
- 资产固化到 **`/seekdb_workspace`**,与用户盘 **`/workspace`** 解耦(k8s:PVC + Secret)
- observer 由 **notebook 单元**启动(教学分步可见),Jupyter 由 entrypoint 启动

```bash
docker run -d --device=/dev/kfd --device=/dev/dri/renderD135 \
  -p 18890:8888 -v $PWD/notebook_workspace:/workspace <image>
# 浏览器 http://<host>:18890 (token: seekdb-hipvs) → 打开 notebook 自上而下运行
```

<!-- 拿到镜像即体验,适合对外演示和 k8s 部署。 -->

---

## 12. 现状与路线图

**"完整支持"= 7 条 DoD**:DDL 开关 · 增删改查正确 · 过滤正确 · 重启可用 · 百万级不 OOM · 指标达标有回归 · 可选有文档

| 阶段 | 内容 | 状态 |
|---|---|---|
| M-A 正确性/回退 | 过滤/删除/非L2/规模召回 | ✅(recall 0.89) |
| M-B 持久化/新鲜度 | B2 新鲜度 ✅;B1 重启重建 | 暂缓 |
| M-C 性能/批量 | C3 基准 + **批量算子** | ✅ 交付 |
| M-D 产品化/上游 | CMake/DDL/测试/文档 | ✅ 大部分 |

> 整体完成约 **15–25%**:happy path + 正确性 + 批量算子已打通。

<!-- 诚实给出完成度,后续工作清晰。 -->

---

## 13. 结论

- **接通**:seekdb 向量索引可选跑 AMD GPU,默认零影响、按索引 opt-in、有安全回退
- **收益边界(诚实)**:单查 GPU ≈ CPU **无收益**;**批量 116×~263×**,端到端 SQL **~12×**
- **又快又更准**:默认参数 cuVS recall ≈0.88 ≥ VSAG ≈0.64,批量与单查结果一致
- **定位一句话**:GPU 的价值在**批量向量检索 / 更高召回 / 超大索引**,而非更快的单查

<!-- 收尾回到开场那句话,首尾呼应。 -->

---

<!-- _class: lead -->
## Q & A

**预设问答**
- 单查为何不快? → CPU HNSW 已亚毫秒,GPU 每调用固定开销吃掉优势
- GPU 值在哪? → 批量 / 更高召回 / 超大索引 + 相似度 JOIN
- 默认会变慢或不稳吗? → 不会,`OB_BUILD_CUVS=OFF` 与上游一致
- 正确性怎么保证? → 不确定即回退 VSAG,过滤/删除/非L2/新鲜度均已验证

**谢谢!**

<!-- 留 Q&A 时间,预设问题帮助控场。 -->
