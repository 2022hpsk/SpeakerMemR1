#!/usr/bin/env python3
"""Run and aggregate full-LLM SpeakerMem S1/S2 top-k sensitivity tests.

This script only reuses the ten held-out SocialMem networks and their 305
questions. It never loads an RL/SFT Writer: memory construction is reused from
`.speakermem_store_v4`, and QA runs the DeepSeek-backed SpeakerMem pipeline.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # supplementary_materia/ or root/
_DEFAULT_PROJECT_ROOT = HERE if (HERE / "rl_prep").exists() else HERE.parent
PROJECT_ROOT = Path(os.environ.get("SPEAKERMEM_PROJECT_ROOT", str(_DEFAULT_PROJECT_ROOT))).expanduser().resolve()
EVAL10 = PROJECT_ROOT / "rl_prep/socialmem_real_eval10"
MANIFEST = PROJECT_ROOT / "rl_prep/socialmem_real_eval10_manifest.json"
STORE = PROJECT_ROOT / ".speakermem_store_v4"
_DEFAULT_OUT_ROOT = (HERE / "rl_prep/topk_sensitivity_full_llm/results"
                     if (HERE / "rl_prep").exists()
                     else HERE / "results/topk_sensitivity_full_llm")
OUT_ROOT = Path(os.environ.get("SPEAKERMEM_SENSITIVITY_OUTPUT", str(_DEFAULT_OUT_ROOT))).expanduser().resolve()
PYTHON = Path(os.environ.get("SPEAKERMEM_PYTHON", str(PROJECT_ROOT / "venv/bin/python"))).expanduser().resolve()


def selected_units_and_questions() -> tuple[list[str], list[str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    units = [row["network_id"] for row in manifest["selected"]]
    questions: list[str] = []
    for unit in units:
        path = EVAL10 / f"social_{unit}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions.extend(q["id"] for q in payload.get("qa_pairs", []))
    if len(units) != 10 or len(questions) != 305:
        raise RuntimeError(f"Expected 10 networks/305 questions, got {len(units)}/{len(questions)}")
    return units, questions


def write_filters(run_dir: Path, units: list[str], questions: list[str]) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    units_file = run_dir / "units.txt"
    questions_file = run_dir / "questions.txt"
    units_file.write_text("\n".join(units) + "\n", encoding="utf-8")
    questions_file.write_text("\n".join(questions) + "\n", encoding="utf-8")
    return units_file, questions_file


def run_one(s1: int, s2: int, args: argparse.Namespace, units: list[str], questions: list[str]) -> Path:
    tag = f"s1k{s1}_s2k{s2}_sourcek{args.source_k}"
    if args.smoke:
        tag += "_smoke"
    run_dir = OUT_ROOT / tag
    results_dir = run_dir / "results"
    units_for_run = units[:1] if args.smoke else units
    questions_for_run = questions[:2] if args.smoke else questions
    units_file, questions_file = write_filters(run_dir, units_for_run, questions_for_run)
    cmd = [
        str(PYTHON), "-u", "speakermem_pkg/examples/socialmembench.py",
        "--benchmark", "socialmem", "--domain", "all",
        "--domain-list-file", str(units_file),
        "--question-ids-file", str(questions_file),
        "--top-k", str(s1), "--s1-recall-n", str(s1), "--s2-k", str(s2),
        "--s2-source-k", str(args.source_k),
        "--our-answerer", "--qa-workers", str(args.qa_workers),
        "--persist-dir", str(STORE), "--results-dir", str(results_dir),
        "--provider", "deepseek",
    ]
    if args.smoke:
        print("[smoke] one network and two questions", flush=True)
    print("[full-LLM] " + " ".join(cmd), flush=True)
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=HERE, env={**__import__("os").environ,
                            "PYTHONPATH": ".:GroupMemBench:speakermem_pkg/src"},
                              stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"configuration {tag} failed; see {run_dir / 'run.log'}")
    return results_dir


def token_f1(pred: str, gold: str) -> float:
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower().rstrip("."))
    p, g = set(norm(pred).split()), set(norm(gold).split())
    if not p or not g:
        return 0.0
    common = p & g
    if not common:
        return 0.0
    return 2 * (len(common) / len(p)) * (len(common) / len(g)) / ((len(common) / len(p)) + (len(common) / len(g)))


def aggregate(results_dir: Path) -> dict:
    records = []
    for path in sorted(results_dir.glob("socialmem/*/speakermem__*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No result JSONL found under {results_dir}")
    acc = 100 * sum(r.get("verdict") == "correct" for r in records) / len(records)
    f1 = 100 * sum(token_f1(r.get("agent_answer", ""), r.get("gold", "")) for r in records) / len(records)
    return {"questions": len(records), "acc": acc, "token_f1": f1}


def write_svg(rows: list[dict], metric: str, path: Path) -> None:
    """Write a dependency-free SVG line chart."""
    width, height, left, top, right, bottom = 760, 500, 80, 45, 25, 70
    xs = sorted({r["s1_topk"] for r in rows})
    ys = [r[metric] for r in rows]
    lo, hi = min(ys + [0.0]), max(ys + [100.0])
    span = max(hi - lo, 1.0)
    def x(v): return left + (v - min(xs)) * (width-left-right) / max(max(xs)-min(xs), 1)
    def y(v): return top + (hi-v) * (height-top-bottom) / span
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#444"/><line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#444"/>')
    for i, s2 in enumerate(sorted({r["s2_k"] for r in rows})):
        data = sorted((r for r in rows if r["s2_k"] == s2), key=lambda r: r["s1_topk"])
        pts = " ".join(f'{x(r["s1_topk"]):.1f},{y(r[metric]):.1f}' for r in data)
        color = colors[i % len(colors)]
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for r in data:
            lines.append(f'<circle cx="{x(r["s1_topk"]):.1f}" cy="{y(r[metric]):.1f}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{width-right-115}" y="{top+20+i*20}" fill="{color}" font-size="14">S2 k={s2}</text>')
    for xv in xs:
        lines.append(f'<text x="{x(xv)-8:.1f}" y="{height-bottom+22}" font-size="13">{xv}</text>')
    lines.append(f'<text x="{width/2-45}" y="{height-15}" font-size="14">System 1 top-k</text>')
    lines.append(f'<text x="15" y="{height/2}" font-size="14" transform="rotate(-90 15 {height/2})">{metric} (%)</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(rows: list[dict], args: argparse.Namespace) -> None:
    target = OUT_ROOT / "smoke" if args.smoke else OUT_ROOT
    target.mkdir(parents=True, exist_ok=True)
    (target / "sensitivity_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (target / "sensitivity_results.csv").open("w", encoding="utf-8") as f:
        f.write("s1_topk,s2_k,s2_source_k,questions,acc,token_f1\n")
        for r in rows:
            f.write(f"{r['s1_topk']},{r['s2_k']},{r['s2_source_k']},{r['questions']},{r['acc']:.6f},{r['token_f1']:.6f}\n")
    try:
        import matplotlib.pyplot as plt
        for metric, label in (("acc", "Binary accuracy (%)"), ("token_f1", "Token-F1 (%)")):
            fig, ax = plt.subplots(figsize=(7, 5))
            for s2 in sorted({r["s2_k"] for r in rows}):
                data = sorted((r for r in rows if r["s2_k"] == s2), key=lambda r: r["s1_topk"])
                ax.plot([r["s1_topk"] for r in data], [r[metric] for r in data], marker="o", label=f"S2 k={s2}")
            ax.set_xlabel("System 1 top-k")
            ax.set_ylabel(label)
            ax.set_xticks(sorted({r["s1_topk"] for r in rows}))
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(target / f"sensitivity_{metric}.png", dpi=180)
            plt.close(fig)
    except ImportError:
        print("[warn] matplotlib unavailable; using SVG fallback", flush=True)
    for metric in ("acc", "token_f1"):
        write_svg(rows, metric, target / f"sensitivity_{metric}.svg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1-values", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--s2-values", nargs="+", type=int, default=[2, 4, 6])
    ap.add_argument("--source-k", type=int, default=1)
    ap.add_argument("--qa-workers", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="run one configuration on one network and two questions")
    ap.add_argument("--project-root", type=Path, default=None, help="repository root containing rl_prep/ and .speakermem_store_v4")
    ap.add_argument("--output-dir", type=Path, default=None, help="directory for sensitivity outputs")
    args = ap.parse_args()
    global PROJECT_ROOT, EVAL10, MANIFEST, STORE, OUT_ROOT, PYTHON
    if args.project_root is not None:
        PROJECT_ROOT = args.project_root.expanduser().resolve()
    if args.output_dir is not None:
        OUT_ROOT = args.output_dir.expanduser().resolve()
    EVAL10 = PROJECT_ROOT / "rl_prep/socialmem_real_eval10"
    MANIFEST = PROJECT_ROOT / "rl_prep/socialmem_real_eval10_manifest.json"
    STORE = PROJECT_ROOT / ".speakermem_store_v4"
    PYTHON = Path(os.environ.get("SPEAKERMEM_PYTHON", str(PROJECT_ROOT / "venv/bin/python"))).expanduser().resolve()
    if not STORE.exists():
        raise SystemExit(f"Missing full-LLM store: {STORE}")
    if any(x <= 0 for x in [*args.s1_values, *args.s2_values, args.source_k]):
        ap.error("all k values must be positive")
    units, questions = selected_units_and_questions()
    rows = []
    configs = [(args.s1_values[0], args.s2_values[0])] if args.smoke else [(a, b) for a in args.s1_values for b in args.s2_values]
    for s1, s2 in configs:
        result_dir = run_one(s1, s2, args, units, questions)
        metrics = aggregate(result_dir)
        rows.append({"s1_topk": s1, "s2_k": s2, "s2_source_k": args.source_k, **metrics})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    write_outputs(rows, args)
    print(f"Wrote {(OUT_ROOT / 'smoke' if args.smoke else OUT_ROOT) / 'sensitivity_results.json'} and plots", flush=True)


if __name__ == "__main__":
    main()
