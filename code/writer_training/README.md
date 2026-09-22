# Writer Training and Held-out Evaluation

This directory contains the trajectory schema, reward implementation, training
data, ten training networks, and ten held-out evaluation networks. It does not
contain model weights, optimizer state, GPU logs, or API credentials.

## Data split

- `training_networks/`: networks used to construct writer trajectories.
- `training_trajectories/`: trajectory JSON files used by supervised and
  reinforced training.
- `heldout_networks/`: held-out networks used for the 305-question evaluation.
- `heldout_manifest.json`: held-out selection and integrity metadata.
- `hard_heldout_manifest.json`: the disjoint hard-network selection.
- `training_manifest.json`: the disjoint training-network selection.

The manifests use relative paths within this artifact. The manual trajectories
are self-contained: their source messages and targets are stored in the
trajectory files.

## Offline checks

```bash
cd <supplementary-material-root>
PYTHONPATH=code/writer_training:code/speakermem_pkg/src \
  python -m pytest -q code/speakermem_pkg/tests/test_training.py
python -m py_compile code/writer_training/train_process_grpo.py
```

## Supervised recipe

The supervised writer evaluation is launched with `run_sft.sh`. It uses a local
vLLM endpoint and the configured external LLM for retrieval, answer generation,
and judging.

## Reinforced recipe

The reinforced training runner is `run_r1_training.sh`. It requires a CUDA GPU
and a local model service. Its settings are defined directly in that script.
The included aggregate result is in `results/training/reinforced/`.

## RL/LLM comparison outputs

The complete comparison run requested for the training analysis is stored in
`results/training/rl_llm_compare/`. It contains the selected units and question
IDs, the run and compile logs, the aggregate summary JSON, and the
question-level SocialMem JSONL files. It reports 218/305 correct under that
run's frozen evaluation configuration. These files are outputs, not model
weights or a standalone training checkpoint.
