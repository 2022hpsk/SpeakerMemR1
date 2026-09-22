
"""Create five human-authored training networks with UPDATE/NOOP coverage."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_training_data import SCHEMA, atomic_dump
from training_rollout import StableHashEmbedder, V4RolloutEnv, prompt_version

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "writer_training" / "training_trajectories"


def msg(speaker, text, turn):
    return {"speaker": speaker, "text": text, "ts": f"2026-08-{10 + turn // 5:02d}T10:{turn % 5:02d}:00Z"}


def head(env, owner, layer, source=None):
    heads = env.snapshot(derived_only=True)["heads"]
    matches = [e for e in heads if e.get("owner") == owner and e.get("layer") == layer
               and (source is None or e.get("source") == source)]
    if not matches:
        raise RuntimeError(f"missing head {owner}/{layer}/{source}")
    return matches[-1]["entry_id"]


def add(owner, source, layer, content, utype="fact"):
    return {"action": "ADD", "owner": owner, "source": source, "layer": layer,
            "utype": utype, "content": content}


def update(entry_id, content):
    return {"action": "UPDATE", "entry_id": entry_id, "content": content}


def build_network(nid, sessions, qa_pairs):
    env = V4RolloutEnv(embedder=StableHashEmbedder(), provenance_thresh=-1.0)
    transitions, counts = [], collections.Counter()
    try:
        for si, (turns, make_actions) in enumerate(sessions):
            sid = f"{nid}_s{si + 1:02d}"
            ts = turns[0]["ts"]
            prompt = env.observe(turns, session_id=sid, ts=ts)
            actions = make_actions(env)
            tr = env.step(turns, actions, session_id=sid, ts=ts)
            if not tr["report"]["valid"]:
                raise RuntimeError(f"{nid}/{sid}: {tr['report']['errors']}")
            tr.update({"segment_index": 0, "source_session_index": si,
                       "metadata": {"source_topic": f"manual_topic_{nid}_{si + 1}"}})
            transitions.append(tr)
            counts.update(str(a["action"]).upper() for a in actions)
        final = env.snapshot(derived_only=True)
    finally:
        env.close()
    normalized_qa = []
    for q in qa_pairs:
        row = dict(q)
        row.setdefault("difficulty", "medium")
        row.setdefault("answer_format", "long_form")
        row.setdefault("options", [])
        row.setdefault("correct_option", "")
        row.setdefault("evidence_anchors", [])
        row.setdefault("temporal_anchors", [])
        normalized_qa.append(row)
    return {
        "schema": SCHEMA,
        "prompt_version": prompt_version(),
        "teacher": {"model": "human-authored", "temperature": 0.0,
                     "thinking": False, "max_tokens": 4096},
        "window_size": 5,
        "network_id": nid,
        "source": {"path": "manual_networks", "meta": {"source_kind": "human_authored"}},
        "transitions": transitions,
        "final_state": final,
        "qa_pairs": normalized_qa,
        "stats": {"transitions": len(transitions), "actions": dict(counts),
                   "derived_entries": len(final["entries"]),
                   "update_edges": len(final["update_edges"])}
    }


def update_networks():
    networks = []


    def alpha():
        a1 = [msg("Alice", "Release 1.0 owner: Alice; first review is Friday.", 0), msg("Bob", "I will review the API checklist.", 1), msg("Alice", "The test environment is ready.", 2), msg("Bob", "I prefer concise checklists.", 3), msg("Alice", "No blocker is known yet.", 4)]
        a2 = [msg("Carol", "Carol will prepare the migration notes.", 5), msg("Alice", "The first review moved to Thursday.", 6), msg("Bob", "The API checklist now includes auth cases.", 7), msg("Carol", "Migration notes will cover rollback.", 8), msg("Alice", "Release scope stays limited to the API.", 9)]
        a3 = [msg("Alice", "The review is complete; two auth cases failed.", 10), msg("Bob", "I will retest auth failures.", 11), msg("Carol", "Rollback notes are drafted.", 12), msg("Alice", "The deadline is Friday again.", 13), msg("Bob", "The checklist should keep the auth section first.", 14)]
        a4 = [msg("Carol", "The migration dry run passed.", 15), msg("Alice", "The release candidate is ready for QA.", 16), msg("Bob", "QA will retest the two auth cases.", 17), msg("Carol", "Rollback is tested in staging.", 18), msg("Alice", "We should ship only after QA sign-off.", 19)]
        a5 = [msg("Alice", "QA fixed both auth cases.", 20), msg("Bob", "The checklist is now approved.", 21), msg("Carol", "Migration notes include the tested rollback.", 22), msg("Alice", "The release candidate is approved.", 23), msg("Bob", "I support shipping Friday.", 24)]
        a6 = [msg("Alice", "Release 1.0 shipped Friday.", 25), msg("Bob", "Post-release monitoring starts today.", 26), msg("Carol", "No rollback was needed.", 27), msg("Alice", "The API scope was unchanged.", 28), msg("Bob", "The final checklist is archived.", 29)]
        def f0(e): return [add("Alice", "Alice", "per_speaker_core", "Alice owns Release 1.0", "decision"), add("Bob", "Bob", "per_speaker_profile", "Bob prefers concise checklists", "stance")]
        def f1(e): return [add("Carol", "Carol", "per_speaker_core", "Carol prepares migration notes"), add("GROUP", "Alice", "group_interaction", "The team schedules the first review for Friday", "decision")]
        def f2(e): return [update(head(e, "Alice", "per_speaker_core", "Alice"), "Alice owns Release 1.0 and the review"), update(head(e, "GROUP", "group_interaction", "Alice"), "The team moves the first review to Thursday",)]
        def f3(e): return [update(head(e, "Carol", "per_speaker_core", "Carol"), "Carol prepares migration notes with rollback"), update(head(e, "Bob", "per_speaker_profile", "Bob"), "Bob prefers concise checklists with auth cases",)]
        def f4(e): return [update(head(e, "Alice", "per_speaker_core", "Alice"), "Alice owns the approved Release 1.0 candidate")]
        def f5(e): return [update(head(e, "GROUP", "group_interaction", "Alice"), "The team ships Release 1.0 on Friday after QA sign-off")]
        return [(a1, f0), (a2, f1), (a3, f2), (a4, f3), (a5, f4), (a6, f5)]

    def beta():
        sessions = []
        for i in range(7):
            base = i * 5
            sessions.append(([msg("Nina", f"Budget checkpoint {i + 1} is discussed.", base), msg("Omar", "Omar owns the cost sheet.", base + 1), msg("Nina", "The supplier quote is under review.", base + 2), msg("Omar", "I will verify the tax line.", base + 3), msg("Nina", "The team needs a signed quote.", base + 4)], None))
        def f0(e): return [add("Nina", "Nina", "per_speaker_core", "Nina tracks the budget", "decision"), add("Omar", "Omar", "per_speaker_core", "Omar owns the cost sheet")]
        def f1(e): return [add("GROUP", "Nina", "group_interaction", "The team requires a signed supplier quote", "decision")]
        def fu(e): return [update(head(e, "Nina", "per_speaker_core", "Nina"), f"Nina tracks budget checkpoint {len(e.snapshot(derived_only=True)['update_edges']) + 3}" )]
        sessions[0] = (sessions[0][0], f0); sessions[1] = (sessions[1][0], f1)
        for i in range(2, 7): sessions[i] = (sessions[i][0], fu)
        return sessions

    def gamma():
        sessions = []
        for i in range(8):
            base = i * 5
            sessions.append(([msg("Pia", f"Experiment phase {i + 1} starts.", base), msg("Quinn", "Quinn records the measurements.", base + 1), msg("Pia", "The target metric is latency.", base + 2), msg("Quinn", "The latest run is logged.", base + 3), msg("Pia", "The next run will use the same target.", base + 4)], None))
        def f0(e): return [add("Pia", "Pia", "per_speaker_core", "Pia owns the experiment", "decision"), add("Quinn", "Quinn", "per_speaker_core", "Quinn records measurements")]
        def f1(e): return [add("GROUP", "Pia", "group_insight", "The group evaluates latency across phases", "observation")]
        def fu(e): return [update(head(e, "Pia", "per_speaker_core", "Pia"), f"Pia owns experiment phase {len(e.snapshot(derived_only=True)['update_edges']) + 3}")]
        sessions[0] = (sessions[0][0], f0); sessions[1] = (sessions[1][0], f1)
        for i in range(2, 8): sessions[i] = (sessions[i][0], fu)
        return sessions

    for nid, factory, qa in [
        ("manual_update_alpha", alpha, [
            {"id": "q1", "category": "update", "question": "Who owns Release 1.0 after the final updates?", "answer": "Alice owns the approved Release 1.0 candidate."},
            {"id": "q2", "category": "update", "question": "When did the team ship Release 1.0?", "answer": "The team shipped it on Friday after QA sign-off."},
            {"id": "q3", "category": "profile", "question": "What kind of checklist does Bob prefer?", "answer": "Bob prefers concise checklists with auth cases."},
            {"id": "q4", "category": "group", "question": "What did the team require before shipping?", "answer": "QA sign-off was required before shipping."},
        ]),
        ("manual_update_beta", beta, [
            {"id": "q1", "category": "update", "question": "Who owns the cost sheet?", "answer": "Omar owns the cost sheet."},
            {"id": "q2", "category": "update", "question": "Which budget checkpoint is the latest one recorded?", "answer": "The latest recorded checkpoint is checkpoint 7."},
            {"id": "q3", "category": "group", "question": "What did the team require from the supplier?", "answer": "A signed supplier quote."},
        ]),
        ("manual_update_gamma", gamma, [
            {"id": "q1", "category": "update", "question": "Who owns the experiment?", "answer": "Pia owns the experiment."},
            {"id": "q2", "category": "update", "question": "Which phase is the latest update about?", "answer": "The latest update is about experiment phase 8."},
            {"id": "q3", "category": "group", "question": "What metric does the group evaluate?", "answer": "The group evaluates latency across phases."},
        ]),
    ]:
        networks.append((nid, factory(), qa))
    return networks


def noop_networks():
    return [
        ("manual_noop_greeting", [([msg("Rae", "Good morning everyone.", 0), msg("Sam", "Good morning.", 1)], lambda e: [{"action": "NOOP"}])], [
            {"id": "q1", "category": "noop", "question": "Did the group record a durable decision?", "answer": "No, the exchange was only a greeting."}]),
        ("manual_noop_ack", [([msg("Tao", "Thanks for the update.", 0), msg("Uma", "You are welcome.", 1), msg("Tao", "That is all for now.", 2)], lambda e: [{"action": "NOOP"}])], [
            {"id": "q1", "category": "noop", "question": "Was a new project fact recorded in this exchange?", "answer": "No, the exchange contains only an acknowledgement."}]),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for nid, sessions, qa in update_networks() + noop_networks():
        data = build_network(nid, sessions, qa)
        atomic_dump(data, OUT / f"{nid}.json")
        print(nid, data["stats"])


if __name__ == "__main__":
    main()
