#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}
DATA_PATH=${DATA_PATH:-./data/GSM8K}
RESULT_BASE=${RESULT_BASE:-./results/sts_sweep}
TRAIN_SEED=${TRAIN_SEED:-42}
EVAL_SEED=${EVAL_SEED:-41}
TEST_K=${TEST_K:-0}
TRAIN_STEPS=${TRAIN_STEPS:-150}
MAX_DATA_EPOCHS=${MAX_DATA_EPOCHS:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
NUM_WORKERS=${NUM_WORKERS:-2}
STS_KMEANS_NITER=${STS_KMEANS_NITER:-20}
STS_KMEANS_DEVICE=${STS_KMEANS_DEVICE:-auto}
STS_BANK_METRICS_INTERVAL=${STS_BANK_METRICS_INTERVAL:-50}
STS_BANK_CACHE_DIR=${STS_BANK_CACHE_DIR:-}
STS_RUN_FILTER=${STS_RUN_FILTER:-}

RUN_NAMES=(n32_tau1 n64_tau1 n128_tau1 n64_tau0p5 n64_tau2)
BANK_SIZES=(32 64 128 64 64)
TEMPERATURES=(1.0 1.0 1.0 0.5 2.0)

should_run() {
  local run_name=$1
  [[ -z "$STS_RUN_FILTER" || ",$STS_RUN_FILTER," == *",$run_name,"* ]]
}

write_status() {
  local status_file=$1
  local state=$2
  local run_name=$3
  local bank_size=$4
  local temperature=$5
  local gpu_id=$6
  local train_seconds=${7:-}
  local eval_seconds=${8:-}
  local train_exit=${9:-}
  local eval_exit=${10:-}
  {
    printf 'state\t%s\n' "$state"
    printf 'run\t%s\n' "$run_name"
    printf 'bank_size\t%s\n' "$bank_size"
    printf 'temperature\t%s\n' "$temperature"
    printf 'gpu\t%s\n' "$gpu_id"
    printf 'train_seconds\t%s\n' "$train_seconds"
    printf 'eval_seconds\t%s\n' "$eval_seconds"
    printf 'train_exit\t%s\n' "$train_exit"
    printf 'eval_exit\t%s\n' "$eval_exit"
    printf 'updated_at\t%s\n' "$(date '+%F %T %Z')"
  } > "$status_file"
}

run_one() {
  local result_root=$1
  local gpu_id=$2
  local run_name=$3
  local bank_size=$4
  local temperature=$5
  local run_dir="$result_root/$run_name"
  local output_dir="$run_dir/train_output"
  local checkpoint="$output_dir/ckpt/final.bin"
  local cache_dir=${STS_BANK_CACHE_DIR:-$result_root/bank_cache}
  local bank_cache="$cache_dir/kmeans_n${bank_size}_seed${TRAIN_SEED}.pt"
  local status_file="$run_dir/status.tsv"
  local train_seconds=0
  local eval_seconds=0
  local train_exit=0
  local eval_exit=0

  mkdir -p "$run_dir" "$(dirname "$bank_cache")"
  if [[ -f "$run_dir/DONE" && -f "$run_dir/eval_results.json" ]]; then
    return 0
  fi

  export CUDA_VISIBLE_DEVICES=$gpu_id
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  export TRANSFORMERS_ATTENTION_IMPLEMENTATION=${TRANSFORMERS_ATTENTION_IMPLEMENTATION:-eager}

  if [[ ! -f "$checkpoint" ]]; then
    write_status "$status_file" RUNNING_TRAIN "$run_name" "$bank_size" "$temperature" "$gpu_id"
    local train_start
    local train_end
    train_start=$(date +%s)
    "$PYTHON_BIN" -u train_gspo_buffer_multitask.py \
      --model_id "$MODEL_ID" \
      --data_path "$DATA_PATH" \
      --output_dir "$output_dir" \
      --task_name gsm8k \
      --path_to_projection_module None \
      --num_thought_tokens 2 \
      --group_size 5 \
      --episodes_per_round 16 \
      --update_epochs 3 \
      --train_steps "$TRAIN_STEPS" \
      --save_every 10 \
      --lr 1e-5 \
      --beta_kl 0.01 \
      --max_data_epochs "$MAX_DATA_EPOCHS" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --seed "$TRAIN_SEED" \
      --projection_type sts \
      --sts_bank_size "$bank_size" \
      --sts_temperature "$temperature" \
      --sts_bank_norm_scale 1.0 \
      --sts_bank_cache "$bank_cache" \
      --sts_kmeans_niter "$STS_KMEANS_NITER" \
      --sts_kmeans_device "$STS_KMEANS_DEVICE" \
      --sts_bank_metrics_interval "$STS_BANK_METRICS_INTERVAL" \
      --feedback_mode vanilla \
      --norm_stats_file "$run_dir/train_norm_stats.jsonl" \
      --norm_stats_max_records 3000 \
      > "$run_dir/train_console.log" 2>&1
    train_exit=$?
    train_end=$(date +%s)
    train_seconds=$((train_end - train_start))
    if [[ $train_exit -ne 0 || ! -f "$checkpoint" ]]; then
      write_status "$status_file" TRAIN_FAILED "$run_name" "$bank_size" "$temperature" "$gpu_id" "$train_seconds" "" "$train_exit" ""
      return 0
    fi
  else
    train_seconds=$(awk -F '\t' '$1 == "train_seconds" {print $2}' "$status_file" 2>/dev/null || true)
  fi

  write_status "$status_file" RUNNING_EVAL "$run_name" "$bank_size" "$temperature" "$gpu_id" "$train_seconds" "" "$train_exit" ""
  local eval_start
  local eval_end
  eval_start=$(date +%s)
  "$PYTHON_BIN" -u evaluate_unified.py \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_PATH" \
    --params_file_name "$checkpoint" \
    --task_name gsm8k \
    --dataset_split test \
    --num_thought_tokens 2 \
    --seed "$EVAL_SEED" \
    --test_k "$TEST_K" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --projection_type sts \
    --sts_bank_size "$bank_size" \
    --sts_temperature "$temperature" \
    --sts_bank_norm_scale 1.0 \
    --sts_bank_cache "$bank_cache" \
    --sts_kmeans_niter "$STS_KMEANS_NITER" \
    --sts_kmeans_device "$STS_KMEANS_DEVICE" \
    --sts_bank_metrics_interval "$STS_BANK_METRICS_INTERVAL" \
    --feedback_mode vanilla \
    --results_file "$run_dir/eval_results.json" \
    --norm_stats_file "$run_dir/eval_norm_stats.jsonl" \
    --norm_stats_max_records 3000 \
    > "$run_dir/eval_console.log" 2>&1
  eval_exit=$?
  eval_end=$(date +%s)
  eval_seconds=$((eval_end - eval_start))
  if [[ $eval_exit -ne 0 || ! -f "$run_dir/eval_results.json" ]]; then
    write_status "$status_file" EVAL_FAILED "$run_name" "$bank_size" "$temperature" "$gpu_id" "$train_seconds" "$eval_seconds" "$train_exit" "$eval_exit"
    return 0
  fi

  write_status "$status_file" DONE "$run_name" "$bank_size" "$temperature" "$gpu_id" "$train_seconds" "$eval_seconds" "$train_exit" "$eval_exit"
  date '+%F %T %Z' > "$run_dir/DONE"
}

worker_main() {
  local worker_index=$1
  local run_id=$2
  local result_root="$RESULT_BASE/$run_id"
  local gpu_id=$worker_index
  local index
  for index in "${!RUN_NAMES[@]}"; do
    if ! should_run "${RUN_NAMES[$index]}"; then
      continue
    fi
    if (( index % NUM_WORKERS != worker_index )); then
      continue
    fi
    run_one "$result_root" "$gpu_id" "${RUN_NAMES[$index]}" "${BANK_SIZES[$index]}" "${TEMPERATURES[$index]}"
  done
}

if [[ ${1:-} == --worker ]]; then
  worker_main "$2" "$3"
  exit 0
fi

RUN_ID=${RUN_ID:-${1:-$(date +%Y%m%d_%H%M%S)}}
RESULT_ROOT="$RESULT_BASE/$RUN_ID"
mkdir -p "$RESULT_ROOT"
{
  printf 'run\tbank_size\ttemperature\n'
  for index in "${!RUN_NAMES[@]}"; do
    if ! should_run "${RUN_NAMES[$index]}"; then
      continue
    fi
    printf '%s\t%s\t%s\n' "${RUN_NAMES[$index]}" "${BANK_SIZES[$index]}" "${TEMPERATURES[$index]}"
  done
} > "$RESULT_ROOT/manifest.tsv"

printf 'RUNNING\t%s\n' "$(date '+%F %T %Z')" > "$RESULT_ROOT/sweep.status"
worker_pids=()
for ((worker=0; worker<NUM_WORKERS; worker++)); do
  "$0" --worker "$worker" "$RUN_ID" >> "$RESULT_ROOT/worker${worker}.log" 2>&1 &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do
  wait "$pid"
done

"$PYTHON_BIN" summarize_sts_sweep.py "$RESULT_ROOT"
if grep -q -E $'^state\t(TRAIN_FAILED|EVAL_FAILED)$' "$RESULT_ROOT"/*/status.tsv 2>/dev/null; then
  printf 'COMPLETED_WITH_FAILURES\t%s\n' "$(date '+%F %T %Z')" > "$RESULT_ROOT/sweep.status"
else
  printf 'DONE\t%s\n' "$(date '+%F %T %Z')" > "$RESULT_ROOT/sweep.status"
fi
