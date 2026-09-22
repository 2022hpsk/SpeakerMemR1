"""Offline tests for writer training data, rollout, and rewards."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "writer_training"))
from make_training_data import generate_network
from training_data import load_sft_examples
from training_rewards import memory_score, reward_v4
from training_rollout import StableHashEmbedder, V4RolloutEnv, parse_actions
from train_process_grpo import (apply_structural_failure_penalty,
                                   apply_terminal_qa_returns,
                                   assign_advantages,
                                   structural_failure_reasons)
from speakermem.agents.writer import Writer
from speakermem.config import WriterConfig
from speakermem.types import Message


def env_factory(): return V4RolloutEnv(embedder=StableHashEmbedder(), provenance_thresh=-1.0)


def test_add_update_builds_non_destructive_chain():
    env = env_factory()
    try:
        first = env.step([{"speaker": "Alice", "text": "Progress is 38%", "ts": "2026-01-01"}],
                         [{"action": "ADD", "owner": "Alice", "source": "Alice",
                           "layer": "per_speaker_core", "utype": "fact", "content": "Project progress is 38%"}],
                         session_id="s1", ts="2026-01-01")
        old = first["post_state"]["heads"][0]["entry_id"]
        second = env.step([{"speaker": "Alice", "text": "Progress is 97%", "ts": "2026-01-02"}],
                          [{"action": "UPDATE", "entry_id": old, "content": "Project progress is 97%"}],
                          session_id="s2", ts="2026-01-02")
        entries = {e["entry_id"]: e for e in second["post_state"]["entries"]}
        new = entries[old]["superseded_by"]
        assert new and old in entries[new]["links"]
        assert entries[old]["content"] == "Project progress is 38%"
        assert entries[new]["content"] == "Project progress is 97%"
        assert second["report"]["valid"]
    finally: env.close()


def test_invalid_update_and_mixed_noop_are_rejected():
    env = env_factory()
    try:
        bad = env.validate_actions([{"action": "UPDATE", "entry_id": "missing", "content": "new"}])
        mixed = env.validate_actions([{"action": "NOOP"},
            {"action": "ADD", "owner": "A", "source": "A", "layer": "per_speaker_core",
             "utype": "fact", "content": "x"}])
        assert not bad.valid and "CURRENT MEMORY" in bad.errors[0]
        assert not mixed.valid and "only action" in mixed.errors[0]
    finally: env.close()


def test_prompt_never_leaks_topic_label():
    env = env_factory()
    try:
        obs = env.observe([{"speaker": "A", "text": "hello", "topic": "SECRET_TOPIC"}],
                          session_id="s1", ts="2026-01-01")
        user = obs["messages"][1]["content"]
        assert "CURRENT MEMORY" in user and "SECRET_TOPIC" not in user
    finally: env.close()


def test_truncated_json_is_atomically_rejected():
    parsed = parse_actions('{"actions":[{"action":"NOOP"}, {')
    assert not parsed["json_valid"] and parsed["actions"] == []


class FakeTeacher:
    def chat(self, system, user, **kwargs):
        ids = re.findall(r"\n\s+([0-9a-f]{8}) \[", user)
        actions = ([{"action": "UPDATE", "entry_id": ids[-1], "content": "Project progress is 97%"}]
                   if ids else [{"action": "ADD", "owner": "Alice", "source": "Alice",
                                 "layer": "per_speaker_core", "utype": "fact",
                                 "content": "Project progress is 38%"}])
        return json.dumps({"actions": actions}, ensure_ascii=False)


def test_generator_produces_v4_trajectory():
    sample = {"meta": {"network_id": "tiny"}, "sessions": [
        {"session_id": "s1", "ts": "2026-01-01", "topic": "gold-one",
         "turns": [{"speaker": "Alice", "text": "Progress is 38%"}]},
        {"session_id": "s2", "ts": "2026-01-02", "topic": "gold-two",
         "turns": [{"speaker": "Alice", "text": "Progress is 97%"}]}]}
    data = generate_network(sample, Path("tiny.json"), FakeTeacher(), teacher_model="fake",
                            env_factory=env_factory)
    assert [t["actions"][0]["action"] for t in data["transitions"]] == ["ADD", "UPDATE"]
    assert len(data["final_state"]["update_edges"]) == 1
    assert "gold-one" not in data["transitions"][0]["prompt"][1]["content"]


def test_reward_prefers_teacher_transition():
    env = env_factory()
    try:
        teacher = env.step([{"speaker": "A", "text": "I handle training"}],
                           [{"action": "ADD", "owner": "A", "source": "A",
                             "layer": "per_speaker_core", "utype": "fact", "content": "A handles training"}],
                           session_id="s1")
        bad = {"report": {"json_valid": False, "attempted": 0, "errors": ["bad"]},
               "post_state": {"heads": [], "entries": [], "update_edges": []}}
        assert reward_v4(teacher, teacher)["reward"] == 1.0
        assert reward_v4(bad, teacher)["reward"] < 1.0
    finally: env.close()


def test_terminal_qa_return_is_discounted_across_each_trajectory():
    trajectories = []
    for _ in range(2):
        trajectory = []
        for pos in range(5):
            components = {"valid": 1.0, "memory": 0.4, "qa": -1.0, "reward": 0.55}
            trajectory.append({"pos": pos, "reward": 0.55, "components": components})
        trajectories.append(trajectory)

    summaries = apply_terminal_qa_returns(trajectories, [{"delta": 1.0, "combined": 1.0, "s1_only": 0.0, "judged": 4}, {"delta": -1.0, "combined": 0.0, "s1_only": 1.0, "judged": 4}], gamma=0.95)

    assert len(summaries) == 2
    assert trajectories[0][0]["components"]["qa_discount"] == 0.95 ** 4
    assert trajectories[0][-1]["components"]["qa"] == 1.0
    assert trajectories[1][-1]["components"]["qa"] == -1.0
    assert trajectories[1][-1]["components"]["qa_delta_terminal"] == -1.0
    expected_first = 0.2 + 0.45 * 0.4 + 0.35 * (0.95 ** 4)
    assert abs(trajectories[0][0]["reward"] - expected_first) < 1e-12
    assert abs(trajectories[1][0]["reward"] - (0.2 + 0.45 * 0.4 - 0.35 * (0.95 ** 4))) < 1e-12


def test_structural_penalty_preserves_terminal_qa_credit():
    pred = {"report": {"json_valid": False, "attempted": 0, "errors": ["bad"]}}
    reasons = structural_failure_reasons(pred, 2048, 2048, 8)
    assert reasons == ["invalid_json", "invalid_actions", "max_tokens"]
    components = apply_structural_failure_penalty(
        {"valid": 0.0, "memory": 1.0, "qa": -1.0, "reward": 0.75},
        reasons, penalty=-10.0)
    trajectories = [[{"pos": 0, "reward": components["reward"],
                      "components": components}]]
    apply_terminal_qa_returns(trajectories, [1.0], gamma=0.95)
    assert abs(trajectories[0][0]["reward"] - 0.3) < 1e-12
    assert components["hard_failure"] == 0.0
    assert components["structural_failure"] == 1.0
    assert trajectories[0][0]["components"]["qa_terminal"] == 1.0


def test_structural_failures_do_not_force_fixed_negative_advantage():
    trajectories = []
    for _ in range(8):
        components = {"structural_failure": 1.0, "structural_penalty": -0.2,
                      "reward": 0.4}
        trajectories.append([{"pos": 0, "reward": 0.4,
                              "components": components}])
    _, zero_positions = assign_advantages(
        trajectories, min_std=0.02, hard_failure_advantage=-3.0)
    assert zero_positions == 1
    assert [row[0]["advantage"] for row in trajectories] == [0.0] * 8


def test_chain_reward_ignores_random_entry_ids():
    left, right = env_factory(), env_factory()
    try:
        add = [{"action": "ADD", "owner": "A", "source": "A", "layer": "per_speaker_core",
                "utype": "fact", "content": "Progress is 38%"}]
        l1 = left.step([{"speaker": "A", "content": "38%"}], add, session_id="s1")
        r1 = right.step([{"speaker": "A", "content": "38%"}], add, session_id="s1")
        lid, rid = l1["post_state"]["heads"][0]["entry_id"], r1["post_state"]["heads"][0]["entry_id"]
        l2 = left.step([{"speaker": "A", "content": "97%"}],
                       [{"action": "UPDATE", "entry_id": lid, "content": "Progress is 97%"}], session_id="s2")
        r2 = right.step([{"speaker": "A", "content": "97%"}],
                        [{"action": "UPDATE", "entry_id": rid, "content": "Progress is 97%"}], session_id="s2")
        assert lid != rid and reward_v4(l2, r2)["chains"] == 1.0
    finally:
        left.close(); right.close()


def test_memory_reward_scores_full_history_not_only_heads():
    ref = {"entries": [
        {"entry_id": "old", "owner": "Alice", "source": "Alice", "layer": "per_speaker_core", "content": "Progress is 38%"},
        {"entry_id": "new", "owner": "Alice", "source": "Alice", "layer": "per_speaker_core", "content": "Progress is 97%"}],
        "update_edges": [{"old": "old", "new": "new"}]}
    pred = {"entries": [ref["entries"][1]], "update_edges": []}
    score = memory_score(pred, ref)
    assert score["state"] < 1.0 and score["chains"] == 0.0 and score["score"] < 0.8


def test_memory_reward_penalizes_wrong_speaker_bucket():
    ref = {"entries": [
        {"owner": "Alice", "source": "Alice", "layer": "per_speaker_core", "content": "Likes rock climbing."},
        {"owner": "Bob", "source": "Bob", "layer": "per_speaker_core", "content": "Likes baking."}], "update_edges": []}
    wrong_owner = {"entries": [
        {"owner": "Bob", "source": "Alice", "layer": "per_speaker_core", "content": "Likes rock climbing."},
        {"owner": "Bob", "source": "Bob", "layer": "per_speaker_core", "content": "Likes baking."}], "update_edges": []}
    assert memory_score(ref, ref)["score"] == 1.0
    corrupted = memory_score(wrong_owner, ref)
    assert corrupted["macro"] < 1.0 and corrupted["worst_bucket"] == 0.0


def test_v4_loader_returns_exact_prompt_and_completion(tmp_path):
    sample = {"meta": {"network_id": "load"}, "sessions": [
        {"session_id": "s1", "turns": [{"speaker": "Alice", "text": "Progress is 38%"}]}]}
    data = generate_network(sample, Path("load.json"), FakeTeacher(), teacher_model="fake",
                            env_factory=env_factory)
    path = tmp_path / "load.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    examples = load_sft_examples([str(path)])
    assert len(examples) == 1
    assert examples[0]["messages"] == data["transitions"][0]["prompt"]
    assert examples[0]["target"] == data["transitions"][0]["target_text"]

class FakeWriterLLM:
    def __init__(self, raw): self.raw = raw
    def chat(self, system, user, **kwargs): return self.raw


def _run_writer_raw(raw):
    env = env_factory()
    llm = FakeWriterLLM(raw)
    writer = Writer(env.memory, lambda: llm,
                    WriterConfig(keep_verbatim=False, provenance=False, max_tokens=128))
    msg = Message(msg_id="m1", speaker="A", content="A and B each state a preference",
                  ts="2026-01-01", session="s1")
    writer._derive("s1", [msg], 0, ["A", "B"])
    return env


def test_writer_recovers_unique_prefix_from_truncated_repeated_tail(capsys):
    first = {"action": "ADD", "owner": "A", "source": "A",
             "layer": "per_speaker_core", "utype": "stance", "content": "A likes tea"}
    repeated = {"action": "ADD", "owner": "B", "source": "B",
                "layer": "per_speaker_core", "utype": "stance", "content": "B likes coffee"}
    raw = json.dumps({"actions": [first, repeated, repeated, repeated]})[:-2] + ",{"
    env = _run_writer_raw(raw)
    try:
        derived = [e for e in env.memory.active_entries() if e.layer == "per_speaker_core"]
        assert [e.content for e in derived] == ["A likes tea", "B likes coffee"]
        assert "[writer recover]" in capsys.readouterr().out
    finally:
        env.close()


def test_writer_still_rejects_ordinary_truncated_json(capsys):
    action = {"action": "ADD", "owner": "A", "source": "A",
              "layer": "per_speaker_core", "utype": "stance", "content": "A likes tea"}
    raw = json.dumps({"actions": [action]})[:-2] + ",{"
    env = _run_writer_raw(raw)
    try:
        assert not [e for e in env.memory.active_entries() if e.layer == "per_speaker_core"]
        assert "[writer reject]" in capsys.readouterr().out
    finally:
        env.close()
