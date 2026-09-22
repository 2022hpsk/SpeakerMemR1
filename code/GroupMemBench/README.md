# GroupMemBench

> **Artifact note.** This supplementary directory does not redistribute the
> original `data/` or `questions/` payloads. Download the upstream release at
> the URL and commit recorded in `../../DATA_SOURCES.md`, then place the files
> at the paths shown in the layout below.


A benchmark for evaluating group-conversation memory systems on
synthetic enterprise channel logs. The upstream release contains:

- **Dataset.** Four domains (Finance, Technology, Healthcare, Manufacturing)
  of synthetic multi-channel group conversations with role-tagged users,
  decision points, temporal phases, and topic-aware noise.
- **Question sets.** Six question types per domain
  (`multi_hop`, `knowledge_update`, `temporal`, `user_implicit`,
  `term_ambiguity`, `abstention`). Finance and Technology questions are the
  solvability-filtered set; Healthcare and Manufacturing ship the unfiltered
  generated set.
- **Two RAG baselines.** A lexical retriever (BM25) and a dense retriever
  (`text-embedding-3-large`), both feeding the same gpt-5 QA agent + judge.

## Layout

```
GroupMemBench/
├── README.md
├── requirements.txt
├── .env.example                       # copy to .env and fill in API keys
├── run_eval.sh                        # top-level: QA + summarise across all cells
├── llm_utils.py                       # shared OpenAI / Azure OpenAI client
├── prompts/
│   ├── hipporag_agent_system.txt      # gpt-5 QA agent system prompt
│   └── hipporag_judge_system.txt      # gpt-5 judge system prompt
├── baselines/
│   ├── rag_common/eval_lib.py         # shared loaders, retrieval/QA loop
│   ├── bm25/                          # rank-bm25, CPU only, no ingest cost
│   │   ├── eval_benchmark.py
│   │   └── run_eval.sh
│   └── text_embedding_3_large/        # OpenAI embeddings + cosine retrieval
│       ├── eval_benchmark.py
│       └── run_eval.sh
├── task_synthesis/
│   └── summarize_typed_eval.py        # accuracy table per (baseline, qtype)
├── data/final/
│   ├── Finance/synthetic_domain_channels_rolevariants_Finance.json
│   ├── Technology/synthetic_domain_channels_rolevariants_Technology.json
│   ├── Healthcare/synthetic_domain_channels_rolevariants_Healthcare.json
│   └── Manufacturing/synthetic_domain_channels_rolevariants_Manufacturing.json
└── questions/
    ├── Finance/<qtype>.jsonl          # 6 files per domain, eval-ready
    ├── Technology/<qtype>.jsonl
    ├── Healthcare/<qtype>.jsonl
    └── Manufacturing/<qtype>.jsonl
```

## Dataset

Each `synthetic_domain_channels_rolevariants_<Domain>.json` is a JSON object
keyed by channel name; the value is a chronologically ordered list of
messages. Each message carries:

| field            | description                                                  |
|------------------|--------------------------------------------------------------|
| `msg_node`       | unique message id (`Msg_<n>`)                                |
| `content`        | natural-language message body                                |
| `author`         | user id (`User_<n>`)                                         |
| `role`           | role label (e.g. `Compliance Officer`)                       |
| `timestamp`      | ISO 8601                                                     |
| `reply_to`       | parent `msg_node` or `null`                                  |
| `phase_name`     | the decision/work phase the message belongs to               |
| `topic`          | thread topic                                                 |
| `is_noise`       | `true` for distractor messages                               |
| `is_decision_point` | `true` when the message records a decision                |
| ... | tone/style/expertise tags, decision metadata, etc. |

The retrievers in this release index `content` only; metadata is attached at
read-time when the retrieved passage is shown to the QA agent (see
`baselines/rag_common/eval_lib.py:format_retrieved_message`).

## Question sets

Each `questions/<Domain>/<qtype>.jsonl` line is

```json
{"id": "multi_hop_1", "question": "...", "answer": "...", "asking_user_id": "User_7"}
```

Counts (per domain × type):

| qtype             | Finance | Technology | Healthcare | Manufacturing |
|-------------------|---------|------------|------------|---------------|
| multi_hop         | 48      | 41         | 48         | 45            |
| knowledge_update  | 32      | 36         | 17         | 22            |
| temporal          | 45      | 37         | 37         | 43            |
| user_implicit     | 15      | 28         |  2         |  4            |
| term_ambiguity    | 45      | 43         |  7         | 11            |
| abstention        | 29      | 28         | 34         | 48            |

Finance and Technology counts are after solvability filtering; Healthcare and
Manufacturing are the raw generated set.

The six types probe orthogonal failure modes:

- **multi_hop** — answer requires combining 2+ messages.
- **knowledge_update** — a later message overrides an earlier claim;
  answering with the stale value is wrong.
- **temporal** — answer hinges on the `timestamp` ordering.
- **user_implicit** — `asking_user_id` resolves an ambiguous referent
  (e.g. "my deadline").
- **term_ambiguity** — different roles use different surface forms for the
  same concept; retrieval must paper over the variation.
- **abstention** — the answer is *not* in the conversation; correct
  behaviour is to refuse rather than confabulate.

## Running the baselines

1. **Install Python deps** (Python 3.9+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API access.** Copy `.env.example` to `.env` and fill in either
   Azure OpenAI or OpenAI credentials. Both the chat agent and the embedding
   model read from this file.

   ```bash
   cp .env.example .env
   $EDITOR .env
   ```

3. **Run the full sweep** (Finance + Technology × bm25 + text-embedding-3-large
   × 6 qtypes = 24 cells), then summarise:

   ```bash
   bash run_eval.sh
   ```

   Per-cell outputs land at `results/<Domain>/<baseline>__<qtype>.jsonl`,
   per-domain accuracy tables at `results/<Domain>/accuracy.md`.

   Common overrides:

   ```bash
   # Just one domain / one baseline / one qtype
   DOMAINS="Finance" BASELINES="bm25" QTYPES="multi_hop" bash run_eval.sh

   # Only re-summarise (skip QA)
   PHASE=summarize bash run_eval.sh

   # Run cells that have output
   FORCE_QA=1 bash run_eval.sh
   ```

4. **Run a single baseline directly** (bypass the orchestrator):

   ```bash
   CONVERSATION_JSON=data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json \
   QUESTIONS_JSONL=questions/Finance/multi_hop.jsonl \
   OUTPUT_JSONL=results/Finance/bm25__multi_hop.jsonl \
   bash baselines/bm25/run_eval.sh
   ```

   The `text_embedding_3_large` baseline caches its embedding matrix under
   `STORE_DIR` (default: `stores/<baseline>_<domain>_eval_store/`). The cache
   is reused across qtypes for the same domain; rerun with `FORCE_REBUILD=1`
   to rebuild.

## Output schema

Each result line in `results/<Domain>/<baseline>__<qtype>.jsonl` records the
retrieved passages, the agent reasoning + answer, and the judge verdict:

```json
{
  "query": "...",
  "asking_user_id": "User_7",
  "retrieved_docs": ["[user=User_3 / role=...] ...", "..."],
  "agent_reasoning": "...",
  "agent_answer": "2025-07-18",
  "judge_reasoning": "...",
  "judge_answer": "Correct"
}
```

`task_synthesis/summarize_typed_eval.py` aggregates these into a Markdown +
TSV table (judge "Correct" → hit; everything else → miss).

## Models used

The two RAG baselines are intentionally LLM-free at retrieval time so the
benchmark isolates retrieval quality. The QA agent and the judge use the same
model in all cells (default: `gpt-5` via Azure OpenAI; override with
`AGENT_MODEL` / `JUDGE_MODEL`). The dense baseline embeds with
`text-embedding-3-large` at 3072-d (override with `EMBEDDING_MODEL`,
`EMBEDDING_DIMS`).
