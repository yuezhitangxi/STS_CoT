# STS-CoT

STS-CoT replaces the linear Projection Model in Self-SoftCoT with Soft Token
Selection (STS). For each soft-thought hidden state, STS forms a position-specific
query, attends over one shared learnable bank initialized by KMeans over Qwen
vocabulary embeddings, and feeds the weighted token mixture back to the frozen
language model.

This repository currently targets **Qwen-2.5-7B-Instruct** and GSM8K. It is a
research prototype; the first sweep is included to document both the method and
its current limitations.

## Method

For hidden state `h_k` and soft token bank `S = {s_i}`:

```text
q_k = W_q h_k
alpha_i = softmax(q_k^T s_i / tau)
e_k = sum_i alpha_i s_i
```

The Qwen backbone is frozen. Thought positions have independent query projections
initialized with Xavier uniform, but share one soft token bank. The queries and
shared bank are optimized by the original Self-SoftCoT GSPO objective.

## Files

- `buffer/sts.py`: shared-bank STS module and FAISS KMeans initialization.
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

Creating a new bank also requires FAISS. GPU FAISS is recommended for the full
Qwen vocabulary:

```bash
conda install -c pytorch -c nvidia -c conda-forge faiss-gpu=1.13.2
```

Version 1.13.2 is the tested build for the project server's Python 3.10 and
CUDA 12.6 environment.

FAISS is only imported when a KMeans cache must be created. Evaluation from an
STS checkpoint does not rerun KMeans.

The shared-bank STS checkpoint layout is not compatible with checkpoints from
the earlier two-bank implementation. Start a new STS run after this revision;
linear-projection baseline checkpoints are unaffected.

Prepare GSM8K in the layout expected by `buffer/data_loader.py`, or reuse the
data preparation from the upstream Self-SoftCoT project.

## Run The Sweep

```bash
cd buffer
MODEL_ID=/path/to/Qwen2.5-7B-Instruct \
DATA_PATH=/path/to/GSM8K \
nohup bash run_sts_sweep.sh > sts_sweep.log 2>&1 &
```

Use `STS_RUN_FILTER` to run a comma-separated subset and
`STS_BANK_CACHE_DIR` to reuse size-specific KMeans caches across sweeps:

```bash
STS_RUN_FILTER=n32_tau1,n64_tau1 \
STS_BANK_CACHE_DIR=./results/sts_bank_cache \
nohup bash run_sts_sweep.sh > sts_sweep.log 2>&1 &
```

The default sweep runs five configurations:

```text
(N, tau) = (32, 1.0), (64, 1.0), (128, 1.0), (64, 0.5), (64, 2.0)
```

Each bank size is clustered once and cached under the run directory. Each
completed training run is evaluated with seed 41. The script records attention
entropy, top-1 weight, STS output norm, bank effective rank, and off-diagonal
pairwise cosine statistics. It produces `summary.csv` and `summary.md` when the
full sweep finishes. Large artifacts, datasets, checkpoints, raw predictions,
and logs are intentionally excluded from Git.

## Initial Finding

The first two configurations of the earlier random-bank, identity-query,
`sqrt(d)`-scaled implementation underperformed the reproduced linear-projection
baseline. Those diagnostics motivated the current KMeans/shared-bank/Xavier
revision. See `results/initial_sweep_summary.md` for the legacy measurements.

## Attribution

This project is based on
[Self-SoftCoT-Code](https://github.com/haha34342/Self-SoftCoT-Code), upstream
commit `27448bcdb11ec91061ce8a3701616cf7ec23c6c5`. The STS module, instrumentation,
and sweep scripts are additions for this project.
