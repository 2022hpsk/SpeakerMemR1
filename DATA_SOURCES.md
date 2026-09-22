# Benchmark Data Sources

The complete original conversation and question payloads are not included in
this artifact. Download each benchmark from its canonical public location and
place the files at the paths below. The listed release identifiers are the
releases used by the evaluation scripts.

## SocialMemBench

- Dataset: anon4data/socialmembench, Hugging Face train split.
- Configurations: networks, personas, conversations, and qa.
- Release statistics: 43 networks, 430 personas, 7,355 turns, and 1,031 QA items.
- Dataset: https://huggingface.co/datasets/anon4data/socialmembench
- Benchmark code: https://anonymous.4open.science/r/SocialMemBench
- Expected layout:

```text
code/SocialMemBench/
├── meta.json
├── networks.jsonl
├── personas.jsonl
├── conversations.jsonl
└── qa.jsonl
```

Only the small metadata file is included. The four benchmark payload files
are external dependencies.

## GroupMemBench

- Release identifier: commit e2682e01ff490acfe4fac2940159dce60307dfc9.
- Repository: https://github.com/UCSB-NLP-Chang/GroupMemBench
- Four domains: Finance, Technology, Healthcare, and Manufacturing.
- Expected layout:

```text
code/GroupMemBench/
├── data/final/<Domain>/*.json
└── questions/<Domain>/*.jsonl
```

The conversation and question payloads are external dependencies. The
included prompts, utilities, retrieval helpers, requirements, and README are
the evaluation code used after the repository is populated.

## EverMemBench-Dynamic

- Dataset: EverMind-AI/EverMemBench-Dynamic, train data.
- Release statistics: 1,263 dialogue blocks, 170 profiles, and 2,400 QA records.
- Dataset: https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic
- Repository: https://github.com/EverMind-AI/EverMemBench
- Release identifier for the accompanying code: commit
  e10b3d52f0e4cfc5c124ad406b5d95c59c73738b.
- Expected layout:

```text
code/EverMemBench-Dynamic/
├── meta.json
├── dialogues.jsonl
├── profiles.jsonl
└── qars.jsonl
```

Only the metadata file is included.

## LoCoMo

- Release: the public ten-conversation LoCoMo evaluation set.
- Repository: https://github.com/snap-research/locomo
- Paper: https://github.com/snap-research/locomo/tree/main/static/paper/locomo.pdf
- Expected layout:

```text
code/LoCoMo/locomo10.json
```

The original JSON payload is an external dependency. The local LoCoMo README
contains the upstream citation and placement note.

## Training artifact

The following files are included because they are needed to inspect
the writer training and held-out evaluation data:

```text
code/writer_training/training_networks/*.json
code/writer_training/training_trajectories/*.json
code/writer_training/heldout_networks/*.json
```

These files are experiment artifacts and are not a replacement for the
complete SocialMemBench release.

## Full-LLM top-k sensitivity inputs

The sensitivity analysis reuses the ten-network held-out selection and 305
questions from the research repository:

```text
rl_prep/socialmem_real_eval10/
rl_prep/socialmem_real_eval10_manifest.json
.speakermem_store_v4/
```

The memory store is fixed across all nine `{System 1 top-k, System 2 k}`
configurations. It is not redistributed in this supplement; the analysis script
accepts its containing repository through `--project-root`. DeepSeek is used by
the full-LLM pipeline, and GPT-4o-mini is used as the judge under the same
configuration as the main evaluation.


## Reproduction order

1. Obtain each public release listed above.
2. Place the files at the documented paths.
3. Install the dependencies in code/requirements.txt.
4. Run the offline checks in README.md.
5. Run a benchmark evaluation with the required external services.

License and citation information is in THIRD_PARTY_NOTICES.md.
