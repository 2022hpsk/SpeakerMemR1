"""Writer reward: validity, full speaker-bucket memory state, and optional terminal QA."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple



VALID_WEIGHT = 0.20
MEMORY_WEIGHT = 0.45
QA_WEIGHT = 0.35


def _tokens(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z0-9]+", text) + re.findall(r"[\u3400-\u9fff]", text)


def text_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb: return 0.0
    sa, sb = set(ta), set(tb); overlap = len(sa & sb)
    p, r = overlap / len(sa), overlap / len(sb)
    token_f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return 0.6 * token_f1 + 0.4 * SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()


def _best_matching(candidates: Sequence[Tuple[float, int, int]]) -> List[Tuple[float, int, int]]:
    """Maximum-weight one-to-one matching; prefer Hungarian matching and fall back to deterministic greedy matching when scipy is unavailable."""
    if not candidates: return []
    pred_ids, ref_ids = sorted({i for _, i, _ in candidates}), sorted({j for _, _, j in candidates})
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        weights = np.zeros((len(pred_ids), len(ref_ids)), dtype=float)
        pi, ri = {v: i for i, v in enumerate(pred_ids)}, {v: j for j, v in enumerate(ref_ids)}
        for score, i, j in candidates: weights[pi[i], ri[j]] = max(weights[pi[i], ri[j]], score)
        rows, cols = linear_sum_assignment(-weights)
        return [(float(weights[i, j]), pred_ids[i], ref_ids[j]) for i, j in zip(rows, cols) if weights[i, j] > 0]
    except ImportError:
        used_p, used_r, out = set(), set(), []
        for score, i, j in sorted(candidates, reverse=True):
            if i not in used_p and j not in used_r:
                used_p.add(i); used_r.add(j); out.append((score, i, j))
        return out


def _soft_f1(pred: Sequence[Dict[str, Any]], ref: Sequence[Dict[str, Any]], threshold=0.5):
    if not ref: return 1.0 if not pred else 0.0
    if not pred: return 0.0
    candidates: List[Tuple[float, int, int]] = []
    for i, p in enumerate(pred):
        for j, g in enumerate(ref):
            if (p.get("owner") == g.get("owner") and p.get("layer") == g.get("layer")
                    and p.get("source") == g.get("source")):
                sim = text_similarity(str(p.get("content", "")), str(g.get("content", "")))
                if sim >= threshold: candidates.append((sim, i, j))
    tp = sum(sim for sim, _, _ in _best_matching(candidates))
    precision, recall = tp / len(pred), tp / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _speaker_levenshtein(pred: Sequence[Dict[str, Any]], ref: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Soft-F1 over the complete derived state, bucketed by owner/GROUP and including superseded history."""
    pred_by_owner, ref_by_owner = {}, {}
    for entry in pred: pred_by_owner.setdefault(str(entry.get("owner", "")), []).append(entry)
    for entry in ref: ref_by_owner.setdefault(str(entry.get("owner", "")), []).append(entry)
    owners = sorted(set(pred_by_owner) | set(ref_by_owner))
    if not owners: return {"macro": 1.0, "worst": 1.0, "score": 1.0, "buckets": 0.0}
    scores = [_soft_f1(pred_by_owner.get(owner, []), ref_by_owner.get(owner, [])) for owner in owners]
    macro, worst = sum(scores) / len(scores), min(scores)
    return {"macro": macro, "worst": worst, "score": 0.8 * macro + 0.2 * worst,
            "buckets": float(len(owners))}


def _chain_score(pred_state, ref_state):
    """Match old-state and new-state semantics without comparing randomly generated entry_id values."""
    def changes(state):
        entries = {e.get("entry_id"): e for e in state.get("entries", [])}
        out = []
        for edge in state.get("update_edges", []):
            old, new = entries.get(edge.get("old"), {}), entries.get(edge.get("new"), {})
            out.append({"owner": old.get("owner"), "layer": old.get("layer"),
                        "old": str(old.get("content", "")), "new": str(new.get("content", ""))})
        return out
    pred, ref = changes(pred_state), changes(ref_state)
    if not ref: return 1.0 if not pred else 0.0
    if not pred: return 0.0
    candidates = []
    for i, p in enumerate(pred):
        for j, g in enumerate(ref):
            if p["owner"] != g["owner"] or p["layer"] != g["layer"]: continue
            score = 0.5 * text_similarity(p["old"], g["old"]) + 0.5 * text_similarity(p["new"], g["new"])
            if score >= 0.5: candidates.append((score, i, j))
    score = sum(sim for sim, _, _ in _best_matching(candidates))
    precision, recall = score / len(pred), score / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _action_key(action):
    if not isinstance(action, dict): return ("INVALID", repr(action))
    kind = str(action.get("action", "")).upper()
    if kind == "ADD": return (kind, str(action.get("owner", "")).strip(), str(action.get("source", "")).strip(), str(action.get("layer", "")).strip(), " ".join(_tokens(str(action.get("content", "")))))
    if kind == "UPDATE": return (kind, str(action.get("entry_id", "")).strip(), " ".join(_tokens(str(action.get("content", "")))))
    if kind == "NOOP": return (kind,)
    return (kind, repr(sorted(action.items())))


def _stop_score(actions, pre_state):
    if not actions: return 0.0
    if len(actions) == 1 and isinstance(actions[0], dict) and str(actions[0].get("action", "")).upper() == "NOOP": return 1.0
    entries = {str(e.get("entry_id")): e for e in (pre_state or {}).get("entries", [])}
    existing = {("ADD", str(e.get("owner", "")).strip(), str(e.get("source", "")).strip(), str(e.get("layer", "")).strip(), " ".join(_tokens(str(e.get("content", ""))))) for e in entries.values()}
    seen, last_novel = set(), -1
    for index, action in enumerate(actions):
        key = _action_key(action)
        kind = key[0]
        novel = False
        if kind == "ADD": novel = key not in seen and key not in existing
        elif kind == "UPDATE":
            old = entries.get(str(action.get("entry_id", "")).strip()) if isinstance(action, dict) else None
            novel = bool(old and not old.get("superseded_by") and " ".join(_tokens(str(old.get("content", "")))) != key[2])
        if novel: last_novel = index
        seen.add(key)
    return 1.0 if last_novel == len(actions) - 1 else 0.0


def validity_components(pred_transition):
    report = pred_transition.get("report") or {}
    actions = pred_transition.get("actions")
    attempted = int(report.get("attempted") or (len(actions) if isinstance(actions, list) else 0))
    r_json = 1.0 if report.get("json_valid") else 0.0
    if attempted <= 0:
        return {"json": r_json, "schema": 0.0, "duplicate": 0.0, "stop": 0.0, "legacy": 0.0, "score": 0.0}
    schema_count = report.get("schema_valid_actions")
    if schema_count is None:
        schema_count = max(0, attempted - len(report.get("errors") or [])) if r_json else 0
    r_schema = min(1.0, max(0.0, float(schema_count) / attempted))
    keys = [_action_key(action) for action in actions] if isinstance(actions, list) else []
    r_duplicate = min(1.0, max(0.0, len(set(keys)) / attempted)) if keys else 0.0
    r_stop = _stop_score(actions if isinstance(actions, list) else [], pred_transition.get("pre_state", {}))
    legacy = max(0.0, 1.0 - len(report.get("errors") or []) / attempted) if r_json else 0.0
    score = 0.25 * r_json + 0.35 * r_schema + 0.25 * r_duplicate + 0.15 * r_stop
    return {"json": r_json, "schema": r_schema, "duplicate": r_duplicate, "stop": r_stop, "legacy": legacy, "score": score}


def validity_score(report):
    return validity_components({"report": report}).get("score", 0.0)


def memory_score(pred_state, ref_state):
    state = _speaker_levenshtein(pred_state.get("entries", []), ref_state.get("entries", []))
    chains = _chain_score(pred_state, ref_state)
    return {"state": state["score"], "macro": state["macro"], "worst_bucket": state["worst"],
            "buckets": state["buckets"], "chains": chains, "score": 0.8 * state["score"] + 0.2 * chains}


def reward_v4(pred_transition: Dict[str, Any], ref_transition: Dict[str, Any],
              qa_score: Optional[float] = None) -> Dict[str, float]:
    """Dense Writer reward with optional terminal R_QA_delta in [-1, 1]."""
    validity = validity_components(pred_transition)
    valid = validity["score"]
    memory = memory_score(pred_transition.get("post_state", {}), ref_transition.get("post_state", {}))
    if qa_score is None:
        qa, total = -1.0, 0.25 * valid + 0.75 * memory["score"]
    else:
        qa = min(1.0, max(-1.0, float(qa_score)))
        total = VALID_WEIGHT * valid + MEMORY_WEIGHT * memory["score"] + QA_WEIGHT * qa
    return {"valid": valid, "memory": memory["score"], "state": memory["state"],
            "macro": memory["macro"], "worst_bucket": memory["worst_bucket"],
            "buckets": memory["buckets"], "chains": memory["chains"], "qa": qa, "reward": total,
            "json": validity["json"], "schema": validity["schema"],
            "duplicate": validity["duplicate"], "stop": validity["stop"],
            "valid_legacy": validity["legacy"]}
