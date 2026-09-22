#!/usr/bin/env python3
"""Generate state-action trajectories for supervised and reinforced writer training.

Only the generate subcommand calls the teacher API; validate is fully offline.
The default window is 25, outputs are written to
writer_training/training_trajectories, and benchmark topics are not used as model inputs.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / "writer_training"), str(HERE / "speakermem_pkg" / "src")]
from speakermem.backends.llm import OpenAICompatLLM
from speakermem.prompts import WRITER_SYSTEM
from training_rollout import V4RolloutEnv, parse_actions, prompt_version

SCHEMA = "speakermem-trajectories/1"


def source_files(inputs: Sequence[str]) -> List[Path]:
    files = []
    for item in inputs:
        path = Path(item) if Path(item).is_absolute() else HERE / item
        if path.is_dir(): files.extend(sorted(path.glob("*.json")))
        elif path.is_file(): files.append(path)
        else: raise FileNotFoundError(path)
    return list(dict.fromkeys(files))


def chunks(items: Sequence[dict], size: int) -> Iterable[List[dict]]:
    if size <= 0: yield list(items); return
    for i in range(0, len(items), size): yield list(items[i:i + size])


def network_id(sample, source):
    meta = sample.get("meta") or {}
    return str(meta.get("network_id") or meta.get("group_id") or source.stem)


def load_project_env():
    """Read the project .env file without printing keys."""
    path = HERE / ".env"
    if not path.exists(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def generate_network(sample: Dict[str, Any], source: Path, teacher, *, teacher_model: str,
                     window_size=25, retries=3, max_tokens=16384,
                     env_factory=V4RolloutEnv) -> Dict[str, Any]:
    env, transitions, counts = env_factory(), [], collections.Counter()
    nid = network_id(sample, source)
    try:
        for si, session in enumerate(sample.get("sessions") or []):
            sid = str(session.get("session_id") or session.get("channel") or f"s{si}")
            turns = list(session.get("turns") or session.get("messages") or [])
            for wi, segment in enumerate(chunks(turns, window_size)):
                if not segment: continue
                ts = str(session.get("ts") or segment[0].get("ts") or segment[0].get("timestamp") or "")
                obs = env.observe(segment, session_id=sid, ts=ts)
                accepted, reasons, retry_feedback = None, [], ""
                for attempt in range(retries):
                    user_prompt = obs["messages"][1]["content"] + retry_feedback
                    raw = teacher.chat(WRITER_SYSTEM, user_prompt, json_mode=True,
                                       thinking=False, temperature=0.0, max_tokens=max_tokens,
                                       model=teacher_model)
                    parsed = parse_actions(raw)
                    report = env.validate_actions(parsed["actions"], json_valid=parsed["json_valid"])
                    if report.valid: accepted = parsed; break
                    errors = report.errors or ["invalid JSON"]
                    reasons.append(errors)
                    repair_hint = ""
                    if any("updates one entry twice" in str(error) for error in errors):
                        repair_hint = (
                            "\nSPECIFIC REPAIR: each existing entry_id may appear in at most ONE UPDATE action. "
                            "Merge all revisions for that entry into one UPDATE, or keep only the single most "
                            "recent revision. Do not emit duplicate UPDATE actions for the same entry_id."
                        )
                    if any(".utype=" in str(error) for error in errors):
                        repair_hint += (
                            "\nSPECIFIC REPAIR: utype MUST be exactly one of fact, stance, observation, "
                            "decision, relation. The value behaviour/behavior is forbidden; use observation "
                            "for an observed behavioural pattern."
                        )
                    retry_feedback = (
                        "\n\nVALIDATION RETRY: Your previous JSON was rejected for these exact reasons:\n- "
                        + "\n- ".join(errors)
                        + "\nReturn a corrected JSON object only. Keep the same conversation evidence and "
                          "CURRENT MEMORY IDs; obey every owner/layer constraint."
                        + repair_hint
                    )
                if accepted is None:
                    raise RuntimeError(f"{nid}/{sid}#{wi}: teacher invalid after {retries} attempts: "
                                       f"{reasons[-1] if reasons else []}")
                transition = env.step(segment, accepted["actions"], session_id=sid, ts=ts,
                                      json_valid=accepted["json_valid"])
                if not transition["report"]["valid"]:
                    raise RuntimeError(f"{nid}/{sid}#{wi}: {transition['report']['errors']}")
                transition.update({"segment_index": wi, "source_session_index": si,
                                   "metadata": {"source_topic": session.get("topic", "")}})
                transitions.append(transition)
                counts.update(str(a.get("action", "")).upper() for a in accepted["actions"])
                print(f"  {nid} {sid}#{wi}: {len(segment)} messages -> "
                      f"{dict(collections.Counter(a.get('action') for a in accepted['actions']))}", flush=True)
        final = env.snapshot(derived_only=True)
        return {"schema": SCHEMA, "prompt_version": prompt_version(),
                "teacher": {"model": teacher_model, "temperature": 0.0,
                            "thinking": False, "max_tokens": max_tokens},
                "window_size": window_size, "network_id": nid,
                "source": {"path": str(source), "meta": sample.get("meta") or {}},
                "transitions": transitions, "final_state": final,
                "qa_pairs": sample.get("qa_pairs") or [],
                "stats": {"transitions": len(transitions), "actions": dict(counts),
                          "derived_entries": len(final["entries"]),
                          "update_edges": len(final["update_edges"])}}
    finally:
        env.close()


def atomic_dump(data, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(destination)


def validate_file(path: Path, require_current_prompt=True) -> List[str]:
    try: data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc: return [f"cannot parse JSON: {exc}"]
    errors = []
    if data.get("schema") != SCHEMA: errors.append(f"wrong schema: {data.get('schema')!r}")
    if require_current_prompt and data.get("prompt_version") != prompt_version():
        errors.append("prompt_version is stale")
    transitions = data.get("transitions")
    if not isinstance(transitions, list) or not transitions: return errors + ["transitions is empty"]
    for i, tr in enumerate(transitions):
        if not (tr.get("report") or {}).get("valid"):
            errors.append(f"transition[{i}] invalid: {(tr.get('report') or {}).get('errors')}")
        legacy = [a.get("action") for a in tr.get("actions") or []
                  if str(a.get("action", "")).upper() in ("WRITE", "PROMOTE")]
        if legacy: errors.append(f"transition[{i}] legacy actions: {legacy}")
        prompt = tr.get("prompt") or []
        user = prompt[1].get("content", "") if len(prompt) > 1 else ""
        topic = str((tr.get("metadata") or {}).get("source_topic") or "")
        if topic and topic in user: errors.append(f"transition[{i}] leaked benchmark topic")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate trajectories with the teacher")
    gen.add_argument("--input", nargs="+", required=True)
    gen.add_argument("--out", default="writer_training/training_trajectories")
    gen.add_argument("--teacher-model", default="deepseek-v4-flash")
    gen.add_argument("--window-size", type=int, default=25)
    gen.add_argument("--retries", type=int, default=3)
    gen.add_argument("--max-tokens", type=int, default=16384)
    gen.add_argument("--limit", type=int, default=0)
    gen.add_argument("--skip-existing", action="store_true")
    gen.add_argument("--overwrite", action="store_true")
    val = sub.add_parser("validate", help="Validate trajectories offline")
    val.add_argument("--input", nargs="+", required=True)
    val.add_argument("--allow-stale-prompt", action="store_true")
    args = parser.parse_args(); files = source_files(args.input)
    if args.command == "validate":
        failed = 0
        for path in files:
            errors = validate_file(path, not args.allow_stale_prompt)
            print(f"{'FAIL' if errors else 'OK  '} {path}")
            for error in errors: print(f"    - {error}")
            failed += bool(errors)
        if failed: raise SystemExit(f"{failed}/{len(files)} files failed")
        return
    if args.skip_existing and args.overwrite: parser.error("choose only one of --skip-existing/--overwrite")
    load_project_env(); teacher = OpenAICompatLLM(default_model=args.teacher_model)
    out = Path(args.out) if Path(args.out).is_absolute() else HERE / args.out
    selected = files[:args.limit] if args.limit > 0 else files
    for source in selected:
        destination = out / f"{source.stem}.json"
        if destination.exists() and args.skip_existing: print(f"SKIP {destination}"); continue
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"{destination} exists; use --skip-existing or --overwrite")
        print(f"== {source.name} -> {destination.name} ==", flush=True)
        data = generate_network(json.loads(source.read_text(encoding="utf-8")), source, teacher,
                                teacher_model=args.teacher_model, window_size=args.window_size,
                                retries=args.retries, max_tokens=args.max_tokens)
        atomic_dump(data, destination); print(f"OK {destination}: {data['stats']}", flush=True)


if __name__ == "__main__": main()
