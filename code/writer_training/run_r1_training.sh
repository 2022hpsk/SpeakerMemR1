#!/usr/bin/env bash
# Train the reinforced writer from the included trajectory data.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-models/speakermem-writer-supervised}"
DATA="writer_training/training_trajectories"

run_job() {
  local group_size="$1"
  local tag="speakermem_reinforced_g${group_size}"
  local save_dir="models/speakermem-writer-${tag}"
  local log="$ROOT/logs/${tag}.log"
  local trace="$ROOT/logs/${tag}.trace.jsonl"

  {
    echo "starting group_size=${group_size} on CUDA_VISIBLE_DEVICES=0"
    echo "model=$MODEL"
    echo "data=$DATA"
    echo "save_dir=$save_dir"
    echo "updates=30 learning_rate=1e-6 ppo_epochs=2 kl_coef=0.1 target_kl=0.02"
    echo "max_new_tokens=2048 temperature=0.8 qa_per_network=4 qa_top_k=2"
    echo "hard_failure_penalty=-1 hard_failure_advantage=0 structural_penalty_mode=bounded_local"
  } > "$log"

  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$ROOT/writer_training/train_process_grpo.py" \
    --model "$MODEL" \
    --data "$DATA" \
    --G "$group_size" \
    --steps 30 \
    --lr 1e-6 \
    --max-new-tokens 2048 \
    --temperature 0.8 \
    --ppo-epochs 2 \
    --clip 0.2 \
    --kl-coef 0.1 \
    --min-reward-std 0.02 \
    --max-grad-norm 1.0 \
    --target-kl 0.02 \
    --train-device cuda:0 \
    --ref-device cuda:0 \
    --save-dir "$save_dir" \
    --save-every 5 \
    --qa-per-network 4 \
    --qa-top-k 2 \
    --qa-return-gamma 0.95 \
    --hard-failure-penalty -1 \
    --hard-failure-advantage 0 \
    --qa-model deepseek-v4-flash \
    --trace-jsonl "$trace" \
    >> "$log" 2>&1
}

cd "$ROOT"
mkdir -p logs

if run_job 8; then
  exit 0
fi

g8_log="$ROOT/logs/speakermem_reinforced_g8.log"
if rg -qi "CUDA out of memory|OutOfMemoryError" "$g8_log"; then
  echo "group_size=8 exceeded device memory; switching to group_size=4" \
    >> "$ROOT/logs/speakermem_reinforced_supervisor.log"
  run_job 4
  exit $?
fi

echo "group_size=8 failed without a device-memory error" \
  >> "$ROOT/logs/speakermem_reinforced_supervisor.log"
exit 1
