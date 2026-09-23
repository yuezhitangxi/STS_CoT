# Initial STS Sweep Summary

## Setup

- Backbone: Qwen-2.5-7B-Instruct
- Dataset: GSM8K
- Training seed: 42
- Evaluation seed: 41
- Thought tokens: 2
- Data budget: one GSM8K training epoch
- Bank initialization: random non-special vocabulary embeddings
- Query initialization: identity matrix
- Bank norm scale: 1.0

## Accuracy

| Method | Bank size | Temperature | Correct | Accuracy | Delta vs. linear baseline |
|---|---:|---:|---:|---:|---:|
| Linear Projection baseline | - | - | 1151 / 1319 | 87.26% | - |
| STS | 32 | 1.0 | 1121 / 1319 | 84.99% | -2.27 pt |
| STS | 64 | 1.0 | 1107 / 1319 | 83.93% | -3.33 pt |

These are single-seed results and should not be treated as a statistically
stable ranking.

## Representation Diagnostics

The following values were collected from the trained checkpoints during
evaluation.

| Bank size | Input embedding norm | STS output norm | Attention entropy | Maximum entropy | Mean top-1 weight |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.9062 | 0.1741 | 3.4180 | 3.4657 | 0.0665 |
| 64 | 0.9062 | 0.1376 | 4.0882 | 4.1589 | 0.0473 |

Attention remains close to uniform after training. Averaging many largely
unrelated vocabulary embeddings causes cancellation: the STS feedback norm is
roughly 5-7 times smaller than the ordinary input embedding norm. Increasing
the bank from 32 to 64 makes both the norm mismatch and accuracy worse in this
initial implementation.

The remaining sweep configurations were stopped after these two completed runs.
