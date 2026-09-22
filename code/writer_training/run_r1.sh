#!/usr/bin/env bash
# Evaluate the reinforced writer on the held-out SocialMem units.
set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"

units=(grp_c1d2e3f7 grp_b8c9dae0 grp_e3f4a5b6 grp_a1b2c3d4 grp_b0c1d2e3
       grp_e8f9a0b1 grp_c2d3e4f5 grp_dba9214c grp_c1d2e3f4 grp_d2e3f4a5)
failed=0
for unit in "${units[@]}"; do
  echo "START ${unit}"
  PYTHONPATH=.:GroupMemBench:speakermem_pkg/src "$PYTHON" -u speakermem_pkg/examples/socialmembench.py \
    --benchmark socialmem --domain "$unit" --top-k 10 --qa-workers 8 \
    --persist-dir /tmp/speakermem_reinforced_store \
    --results-dir ../results/training/reinforced \
    --writer-vllm-url http://127.0.0.1:8001/v1 \
    --writer-vllm-model speakermem-reinforced
  status=$?
  echo "END ${unit} status=${status}"
  if (( status != 0 )); then
    failed=1
  fi
done

exit "$failed"
