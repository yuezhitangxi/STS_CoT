# Self-SoftCoT + STS 实验方案

## 目标

在 Self-SoftCoT 的基础上只做一个核心改动：将原来的 Projection Model 替换为 STS（Soft Token Selection）模块，观察用 soft token bank 构造反馈输入后，是否能缓解 projection 输出 embedding norm 与普通 token embedding norm 不匹配的问题，并提升或稳定推理效果。

## 背景观察

当前 Self-SoftCoT 复现中的 norm 统计显示：

- 普通 token embedding norm 约为 `0.90`
- soft token 原始 embedding norm 约为 `0.97`
- Projection Model 输入 hidden state norm 约为 `315`
- Projection Model 输出 norm 约为 `27`
- 最终 prompt embedding 的最大 norm 会被 projection 输出拉到 `27-28`
- 未观察到 NaN/Inf

这说明现有 Projection Model 输出的反馈 embedding 与原始 token embedding 的尺度差距较大。STS 的动机是：不直接用 hidden state 线性投影成反馈 embedding，而是从一个可学习 soft token bank 中选择/加权组合出反馈输入，使反馈向量天然落在更接近 token embedding 的表示空间。

## 方法改动

保留 Self-SoftCoT 的整体训练、数据、GSPO 目标、Qwen-2.5-7B-Instruct 底座、thought token 数量等设置，只替换 Projection Model。

原方法：

```text
h_k -> Linear Projection_i -> projected soft embedding e_k
```

改为 STS：

```text
h_k -> W_q -> q_k
q_k 与 Soft Token Bank S 做 attention
e_k = sum_i alpha_i s_i
```

其中：

```text
q_k = W_q h_k
alpha_i = softmax(q_k^T s_i / tau)
e_k = sum_i alpha_i s_i
```

Soft Token Bank `S` 是 `N` 个可学习向量。对排除特殊 token 后的完整 Qwen 词表 embedding 做 KMeans，直接使用聚类中心初始化 Bank。两个 thought position 共享同一个 Bank，但分别使用 Xavier 初始化的独立 `W_q`。

## 实现范围

只新增一个 STS 模块替代当前 `model.projections[i](vec)`：

- 输入：第 `k` 个 soft thought hidden state `h_k`
- 输出：反馈给 LLM 的 soft embedding `e_k`
- 两个 thought position 共享同一个 Soft Token Bank
- 每个 thought position 保留独立的 `W_q`，并使用 Xavier uniform 初始化
- KMeans 中心按 bank size 和 seed 缓存，评测 checkpoint 时不重复聚类

不改动：

- 数据集
- prompt 格式
- thought token 数量主设定
- GSPO 损失
- reward 计算
- 评测脚本
- Qwen-2.5-7B-Instruct 底座

## 超参数搜索

第一轮只搜索 STS 相关超参。

| 超参 | 候选值 | 说明 |
|---|---:|---|
| soft token bank size `N` | `16, 32, 64, 128` | 图片方案中使用 `64`，先围绕 64 搜索 |
| query dim `d_q` | `hidden_size, 1024, 2048` | 若 `d_q < hidden_size`，同时学习 `W_q` 和 bank projection |
| temperature `tau` | `0.5, 1.0, 2.0` | attention logits 使用 `q^T s / tau` |
| bank 初始化 | `kmeans` | 对完整非特殊词表 embedding 聚类，使用聚类中心 |
| bank norm scale | `0.5, 1.0, 2.0` | 初始化后整体缩放，用于控制反馈尺度 |
| top-k selection | `none, 8, 16` | 可选稀疏化；第一轮可先不用 |
| entropy regularization | `0, 1e-4, 1e-3` | 防止 attention 过早塌缩到少数 soft tokens |

建议第一轮搜索不要全组合，采用分阶段：

1. 固定 `d_q=hidden_size`、`tau=1.0`、共享 Bank、KMeans 初始化，搜索 `N = 16, 32, 64, 128`。
2. 选最好的 `N` 后，搜索 `tau = 0.5, 1.0, 2.0`。
3. 根据 attention entropy、Bank effective rank 和 cosine similarity 判断是否发生塌缩。
4. 最后微调 `bank norm scale` 和 `entropy regularization`。

## 对照实验

最小对照如下：

| 实验 | Projection 形式 | 目的 |
|---|---|---|
| baseline | 原 Self-SoftCoT Linear Projection | 复现实验基线 |
| STS-main | STS，`N=64`，共享 KMeans Bank | 验证核心改动 |
| STS-bank-size | STS，搜索 `N` | 看 token bank 容量影响 |
| STS-temperature | STS，搜索 `tau` | 看选择分布锐度影响 |
| STS-scale | STS，搜索 bank norm scale | 看反馈 norm 是否影响性能 |

## 记录指标

除了原论文指标，还记录以下中间指标：

- 普通 token embedding norm
- soft token bank norm
- STS query norm
- STS 输出 `e_k` norm
- STS 输出与普通 token embedding norm 的比例
- attention entropy
- top-1 soft token 使用频率
- 不同 thought position 的 token bank 使用差异
- Bank effective rank
- Bank token 两两 cosine similarity 的 mean/std/min/max
- NaN/Inf 检测
- GSM8K accuracy
- 单 seed 推理耗时

## 预期现象

如果 STS 有效，应该至少看到：

- STS 输出 norm 更接近普通 token embedding norm，而不是稳定在 `27` 左右。
- final prompt embedding 的 max norm 明显下降。
- attention 分布不是完全均匀，也不是一开始就塌缩到单个 token。
- 在相同训练预算下，GSM8K accuracy 不低于原 Projection baseline，最好更稳定。

## 当前推荐配置

```text
N = 64
d_q = hidden_size
temperature = 1.0
position-wise query = independent
soft token bank = shared
bank_init = kmeans
bank_norm_scale = 1.0
W_q init = xavier_uniform
attention_scaling = qS^T / tau
top_k = none
entropy_reg = 1e-4
```

第一版先用该配置跑通训练和单 seed 评测，再决定是否扩大搜索。
