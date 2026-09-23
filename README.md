# STS-CoT

STS-CoT replaces the linear Projection Model in Self-SoftCoT with Soft Token
Selection (STS). For each soft-thought hidden state, STS forms a query, attends
over a learnable bank initialized from Qwen vocabulary embeddings, and feeds the
weighted token mixture back to the frozen language model.

This repository currently targets **Qwen-2.5-7B-Instruct** and GSM8K. It is a
research prototype; the first sweep is included to document both the method and
its current limitations.

## Method

For hidden state `h_k` and soft token bank `S = {s_i}`:

```text
q_k = W_q h_k
alpha_i = softmax(q_k^T s_i / (sqrt(d) * tau))
e_k = sum_i alpha_i s_i
```

The Qwen backbone is frozen. Only the position-wise query projections and soft
token banks are optimized by the original Self-SoftCoT GSPO objective.

## Files

- `buffer/sts.py`: STS module and vocabulary-based initialization.
- `buffer/unified_llm_model.py`: Qwen wrapper with linear/STS projection modes.
- `buffer/train_gspo_buffer_multitask.py`: GSPO training entry point.
- `buffer/evaluate_unified.py`: single-seed evaluation and metric collection.
- `buffer/run_sts_sweep.sh`: detached-friendly two-GPU sweep driver.
- `buffer/norm_monitor.py`: embedding, hidden-state, query, bank, output, and attention statistics.
- `results/initial_sweep_summary.md`: compact results from the first completed configurations.
- `docs/STS实验方案.md`: experiment proposal in Chinese.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare GSM8K in the layout expected by `buffer/data_loader.py`, or reuse the
data preparation from the upstream Self-SoftCoT project.

## Run The Sweep

```bash
cd buffer
MODEL_ID=/path/to/Qwen2.5-7B-Instruct \
DATA_PATH=/path/to/GSM8K \
CUDA_VISIBLE_DEVICES=0,1 \
nohup bash run_sts_sweep.sh > sts_sweep.log 2>&1 &
```

The default sweep runs five configurations:

```text
(N, tau) = (32, 1.0), (64, 1.0), (128, 1.0), (64, 0.5), (64, 2.0)
```

Each completed training run is evaluated with seed 41. The script writes
per-run status files and produces `summary.csv` and `summary.md` when the full
sweep finishes. Large artifacts, datasets, checkpoints, raw predictions, and
logs are intentionally excluded from Git.

## Initial Finding

The first two configurations underperformed the reproduced linear-projection
baseline. Diagnostics show that attention remained close to uniform, causing
the convex combination of randomly sampled vocabulary embeddings to have a
much smaller norm than ordinary token embeddings. See
`results/initial_sweep_summary.md` for exact measurements.

## Attribution

This project is based on
[Self-SoftCoT-Code](https://github.com/haha34342/Self-SoftCoT-Code), upstream
commit `27448bcdb11ec91061ce8a3701616cf7ec23c6c5`. The STS module, instrumentation,
and sweep scripts are additions for this project.
