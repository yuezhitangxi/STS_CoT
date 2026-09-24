# STS 实验结果汇总

更新时间：2026-09-24（`N=64, tau=1, /sqrt(d)` 对照实验完成前）

## 1. 实验口径

除特别说明外，主结果使用以下统一配置：

- Backbone：Qwen-2.5-7B-Instruct，参数冻结；
- 数据集：GSM8K，训练集 7000 题，测试集 1319 题；
- soft thought 数量：2；
- 训练 seed：42；评测 seed：41；
- `group_size=5`，`episodes_per_round=16`，`update_epochs=3`；
- `lr=1e-5`，`beta_kl=0.01`；
- 最多 150 train steps、1 个训练数据 epoch；
- 表中准确率均为单 seed 结果，不能视为统计稳定的最终排名。

## 2. 完整测试集结果

| 方法 | N | tau | Attention scaling | Bank / Wq | Correct | Accuracy | 相对 Linear | 训练耗时 | 评测耗时 |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| Self-SoftCoT Linear baseline | - | - | - | Linear Projection | 1151/1319 | **87.26%** | - | - | - |
| Legacy STS | 32 | 1 | `/sqrt(d)` | 两个独立随机词表 Bank；`Wq=I` | 1121/1319 | **84.99%** | -2.27 pt | 15.03 h | 2.70 h |
| Legacy STS | 64 | 1 | `/sqrt(d)` | 两个独立随机词表 Bank；`Wq=I` | 1107/1319 | **83.93%** | -3.33 pt | 15.65 h | 2.51 h |
| STS v2 | 64 | 1 | 无 `/sqrt(d)` | 共享 KMeans Bank；独立 Xavier Wq | 1069/1319 | **81.05%** | -6.21 pt | 15.07 h | 1.38 h |
| STS v2 | 32 | 1 | 无 `/sqrt(d)` | 共享 KMeans Bank；独立 Xavier Wq | 973/1319 | **73.77%** | -13.49 pt | 18.85 h | 2.99 h |

当前所有已完成 STS 实验均未超过 Linear baseline。Legacy STS 优于 STS v2 的两个无缩放配置；STS v2 中 `N=64` 又明显优于 `N=32`。

## 3. 表示与选择行为

下表对两个 thought position 取平均。最大 attention entropy 分别为 `ln(32)=3.4657` 和 `ln(64)=4.1589`。

| 方法 | N | Query norm | Bank token norm | STS output norm | Attention entropy | Top-1 weight | Bank effective rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy STS | 32 | 335.9745 | 0.7865 | 0.1740 | 3.4181 | 0.0664 | 未记录 |
| Legacy STS | 64 | 366.8516 | 0.7902 | 0.1376 | 4.0883 | 0.0473 | 未记录 |
| STS v2 | 32 | 311.6221 | 0.7488 | 0.7863 | 0.0174 | 0.9977 | 29.66 / 32 |
| STS v2 | 64 | 311.5295 | 0.7480 | 0.8524 | 0.1018 | 0.9844 | 59.02 / 64 |

各配置在全部 1319 个测试样本上的 Top-1 Bank token 如下：

| 方法 | N | Thought position 0 | Thought position 1 |
|---|---:|---:|---:|
| Legacy STS | 32 | `bank[15]` | `bank[26]` |
| Legacy STS | 64 | `bank[22]` | `bank[12]` |
| STS v2 | 32 | `bank[30]` | `bank[16]` |
| STS v2 | 64 | `bank[62]` | `bank[55]` |

Legacy STS 的分布接近均匀，导致多个 Bank token 相互抵消，STS output norm 只有约 `0.14-0.17`。STS v2 去除 `/sqrt(d)` 后走向相反极端：Top-1 权重达到 `98%-100%`，几乎退化为每个位置固定选择一个 Bank token。

STS v2 的 Bank effective rank 仍接近满秩，且平均非对角 cosine similarity 很低（N=32：0.0223；N=64：0.0209），说明当前主要问题是 attention collapse，而不是 Bank 本身整体坍缩。

## 4. N=64 v2 补充 Norm

以下数据直接由最终 checkpoint 和 1319 条评测记录计算，std 为总体标准差：

- Query norm（两个位置合并，2638 个观测）：`311.5295 +/- 1.2506`；
- Position 0 query norm：`311.8377 +/- 1.2805`；
- Position 1 query norm：`311.2213 +/- 1.1395`；
- Bank norm：mean `0.7480`，std `0.2303`，min `0.0815`，max `0.9742`；
- `bank[62]` norm：`0.8997`；
- `bank[55]` norm：`0.8313`；
- 若 attention 完全均匀，当前 Bank 的输出 norm 仅为 `0.1279`。

在 `d=3584` 时，`sqrt(d)=59.87`。因此无缩放和完整 `/sqrt(d)` 之间跨度很大：前者已经导致尖锐选择，后者有可能重新接近均匀分布。

## 5. Scale-Match 辅助实验

这部分在 Linear Projection checkpoint 上考察反馈 norm，使用 seed 42 和测试子集，不可与 1319 题主结果直接比较。

### 150 题开发实验

| Variant | Correct / Total | Accuracy | 相对 vanilla | Injected norm | 平均输出 tokens |
|---|---:|---:|---:|---|---:|
| vanilla | 140/150 | **93.33%** | - | 27.738, 27.013 | 205.2 |
| scale_match x0.5 | 130/150 | 86.67% | -6.67 pt | 0.467, 0.467 | 209.1 |
| scale_match x1 | 130/150 | 86.67% | -6.67 pt | 0.933, 0.933 | 211.4 |
| scale_match x2 | 137/150 | 91.33% | -2.00 pt | 1.867, 1.867 | 214.0 |
| scale_match x4 | 130/150 | 86.67% | -6.67 pt | 3.733, 3.733 | 210.5 |

### 16 题 Pilot

| Variant | Correct / Total | Accuracy | 相对 vanilla | Injected norm |
|---|---:|---:|---:|---|
| vanilla | 15/16 | 93.75% | - | 27.721, 26.996 |
| scale_match x0.5 | 14/16 | 87.50% | -6.25 pt | 0.467, 0.467 |
| scale_match x1 | 12/16 | 75.00% | -18.75 pt | 0.933, 0.933 |
| scale_match x2 | 15/16 | 93.75% | 0.00 pt | 1.867, 1.866 |
| scale_match x4 | 13/16 | 81.25% | -12.50 pt | 3.734, 3.733 |

这些辅助结果表明，单纯把反馈 norm 强制匹配到普通 token embedding 附近并不会自动提高准确率。

## 6. 未形成最终结果的配置

| 配置 | 状态 | 说明 |
|---|---|---|
| Legacy STS，N=128，tau=1 | 用户停止 | 无完整训练与评测结果 |
| Legacy STS，N=64，tau=0.5 | 用户停止 | 无完整训练与评测结果 |
| Legacy STS，N=64，tau=2 | 未运行 | 停止后续 sweep 时尚未启动 |
| STS v2，N=64，tau=1，恢复 `/sqrt(d)` | 运行中 | 严格单变量对照；复用同一 N=64 KMeans cache |

## 7. 当前结论

1. KMeans 初始化、共享 Bank 和 Xavier Wq 在无 `/sqrt(d)` 时没有带来准确率提升。
2. 无 `/sqrt(d)` 的主要失败模式是 attention collapse，而不是 Bank rank collapse。
3. Legacy `/sqrt(d)` 配置的主要问题则是 attention 过于平滑和输出 norm 过小。
4. 正在运行的 N=64 `/sqrt(d)` 严格对照将检验：在保留 KMeans、共享 Bank 和 Xavier Wq 的前提下，恢复缩放能否从尖锐选择一侧拉回合理区间。
5. 若完整 `/sqrt(d)` 再次过于均匀，下一步应搜索中间 logit scale，而不是继续只改变 Bank size。

## 8. 服务器结果位置

- Legacy STS：`buffer/results/sts_sweep/sts_qwen25_gsm8k_20260922_215806/`
- STS v2（无缩放）：`buffer/results/sts_sweep/sts_v2_n32_n64_tau1_20260923_165431/`
- STS v2（`/sqrt(d)` 对照）：`buffer/results/sts_sweep/sts_v2_n64_tau1_sqrtd_20260924_110615/`
- Scale-match：`buffer/results/scale_match/`

待 `/sqrt(d)` 对照完成后，应在本文件中追加其准确率、attention entropy、Top-1 weight、STS output norm 和 Bank effective rank，再更新结论。
