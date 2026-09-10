# YOLO26 + VLM Hands-on：Prompt 实验设计

> 目标文件：`ultralytics_yolo26x_hands_on.ipynb`
> 目标：用户只修改一句自然语言问题，复杂实现由应用负责。

## 核心心智模型

```text
YOLO：每帧看见对象
Coordinator：决定何时触发、提供哪些帧
VLM：根据 Evidence + Prompt 回答问题
```

用户只控制 Prompt。视频、YOLO boxes 和首次对比使用的 evidence hash 保持不变。

## 16 个 Cell

| Cell | 类型 | 内容 | 用户动作 |
|---:|---|---|---|
| 1 | Markdown | 同一视频，不同问题 | 理解目标 |
| 2 | Markdown | YOLO / Coordinator / VLM 两个时钟 | 区分控制边界 |
| 3 | Python | 环境与 baseline 准备，源码隐藏 | 无 |
| 4 | Markdown | 固定 evidence 原则 | 预测可见事实 |
| 5 | Python | YOLO frame、storyboard、trigger timeline、SHA | 查看证据 |
| 6 | Markdown | Prompt Card 示例 | 选择问题方向 |
| 7 | Python | `MY_QUESTION = ...` | **唯一编辑项** |
| 8 | Markdown | 运行前预测 | 预测关注点 |
| 9 | Python | 同 evidence 的默认/custom 答案 | 比较答案 |
| 10 | Markdown | focus、grounding、format | 判断质量 |
| 11 | Python | 同 prompt 的四个 trigger window | 查看 evidence 变化 |
| 12 | Markdown | 同 YOLO、不同 semantic view | 理解组合能力 |
| 13 | Python | baseline/custom 视频 | 观察结果 |
| 14 | Markdown | Prompt / Trigger / Evidence 边界 | 明确能控制什么 |
| 15 | Python | 保存 submission | 提交结论 |
| 16 | Markdown | Takeaways | 总结 |

## 固定 Prompt Card

系统负责固定三项：

```text
Role: careful visual observer
Format: one sentence, no more than 20 words
Grounding: use only visible evidence; say unclear when needed
```

参与者只填写：

```python
MY_QUESTION = "What road-safety risks are visible?"
```

## 两个实验

### 同 Evidence，不同 Prompt

```text
Evidence E + default question -> general scene summary
Evidence E + user question    -> user-selected focus
```

必须保存同一个 evidence SHA，证明答案变化来自 Prompt，不是输入变化。

### 同 Prompt，不同 Trigger

```text
0-4 s evidence   -> answer 1
4-8 s evidence   -> answer 2
8-12 s evidence  -> answer 3
12-15.72 s       -> answer 4
```

Prompt 不变，但 Coordinator 在不同时间选择了不同 Evidence，因此答案变化。

## 已验证结果

默认问题：

> What is happening in this scene?

输出：

> Pedestrians cross street as cars pass by in urban setting.

道路安全问题：

> What road-safety risks are visible?

同一 evidence 输出：

> Pedestrians cross street while vehicles approach, creating potential collision risk.

两个视频均通过 393 frames / 393 packets / 393 decoded frames 校验。

## 用户最终理解

1. YOLO 决定每帧看见哪些对象。
2. Coordinator 决定什么时候问、给 VLM 看哪些帧。
3. Prompt 决定 VLM 从什么角度解释证据。
4. Prompt 不能改变视频，也不能创造不可见事实。
