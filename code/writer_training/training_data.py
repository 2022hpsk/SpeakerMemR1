"""The single loader for writer trajectory data; legacy snapshot data is rejected."""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from make_training_data import SCHEMA, validate_file
from training_rollout import prompt_version

HERE = Path(__file__).resolve().parents[1]


def resolve_files(items: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for item in items:
        path = Path(item)
        candidates = [path] if path.is_absolute() else [HERE / path, HERE / "writer_training" / path]
        path = next((p for p in candidates if p.exists()), candidates[0])
        if path.is_dir(): files.extend(sorted(path.glob("*.json")))
        elif path.is_file(): files.append(path)
        else: raise FileNotFoundError(path)
    return list(dict.fromkeys(files))


def load_networks(items: Sequence[str], *, require_current_prompt: bool = True) -> List[Dict[str, Any]]:
    networks = []
    for path in resolve_files(items):
        errors = validate_file(path, require_current_prompt=require_current_prompt)
        if errors: raise ValueError(f"invalid trajectory data {path}: {'; '.join(errors)}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA: raise ValueError(f"legacy data is forbidden: {path}")
        data["_path"] = str(path); networks.append(data)
    if not networks: raise ValueError("no trajectory networks found")
    return networks


def load_sft_examples(items: Sequence[str]) -> List[Dict[str, Any]]:
    examples = []
    for network in load_networks(items):
        for i, transition in enumerate(network["transitions"]):
            examples.append({"network_id": network["network_id"], "position": i,
                             "messages": transition["prompt"],
                             "target": transition["target_text"],
                             "actions": transition["actions"]})
    return examples


def action_counts(networks: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"ADD": 0, "UPDATE": 0, "NOOP": 0}
    for network in networks:
        for transition in network["transitions"]:
            for action in transition.get("actions") or []:
                kind = str(action.get("action", "")).upper()
                if kind in counts: counts[kind] += 1
    return counts
