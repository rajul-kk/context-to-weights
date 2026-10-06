import random

import pytest

from sleep.examples import (ReplayBuffer, SleepExample, from_compaction, from_compaction_mix,
                            from_compaction_stratified, from_reflection, from_uniform)

KEPT = [(f"kept {i}", f"fact{i}" if i < 2 else None) for i in range(6)]
DROPPED = ([(f"drop a{i}", "a") for i in range(2)] + [(f"drop b{i}", "b") for i in range(2)]
           + [(f"drop c{i}", "c") for i in range(2)] + [(f"filler {i}", None) for i in range(10)])


@pytest.fixture
def event(make_event):
    return make_event(KEPT, DROPPED)


def test_from_compaction_keeps_only_kept_spans_by_default(event):
    out = from_compaction(event)
    assert [e.target for e in out] == [t for t, _ in KEPT]
    assert all(e.kept for e in out)


def test_from_compaction_can_include_dropped(event):
    assert len(from_compaction(event, include_dropped=True)) == len(KEPT) + len(DROPPED)


def test_mix_zero_reproduces_the_kept_set(event):
    out = from_compaction_mix(event, random.Random(0), 0.0)
    assert sorted(e.target for e in out) == sorted(t for t, _ in KEPT)


@pytest.mark.parametrize("mix", [0.25, 0.5, 1.0])
def test_mix_keeps_the_budget_matched(event, mix):
    out = from_compaction_mix(event, random.Random(1), mix)
    assert len(out) == len(KEPT)
    assert sum(1 for e in out if not e.kept) == round(mix * len(KEPT))


def test_mix_never_draws_more_dropped_spans_than_exist(make_event):
    event = make_event(KEPT, [("only", None)])
    out = from_compaction_mix(event, random.Random(0), 1.0)
    assert sum(1 for e in out if not e.kept) == 1
    assert len(out) == len(KEPT)


def test_stratified_mix_spreads_over_fact_keys(event):
    out = from_compaction_stratified(event, random.Random(3), 0.5)
    dropped = [e for e in out if not e.kept]
    assert len(out) == len(KEPT)
    assert len(dropped) == 3
    assert len({e.fact_key for e in dropped}) == 3


def test_stratified_mix_zero_is_the_kept_set(event):
    out = from_compaction_stratified(event, random.Random(0), 0.0)
    assert sorted(e.target for e in out) == sorted(t for t, _ in KEPT)


def test_mix_is_deterministic_for_a_seed(event):
    a = from_compaction_mix(event, random.Random(5), 0.5)
    b = from_compaction_mix(event, random.Random(5), 0.5)
    assert [e.target for e in a] == [e.target for e in b]


def test_uniform_defaults_to_the_kept_count_and_samples_all_spans(event):
    out = from_uniform(event, random.Random(0))
    assert len(out) == len(KEPT)
    assert all(e.kept for e in out)
    seen = set()
    for seed in range(40):
        seen |= {e.target for e in from_uniform(event, random.Random(seed))}
    assert any(t.startswith("filler") for t in seen)


def test_uniform_respects_explicit_zero_and_clips_to_available(event):
    assert from_uniform(event, random.Random(0), n=0) == []
    assert len(from_uniform(event, random.Random(0), n=10_000)) == len(KEPT) + len(DROPPED)


def test_reflection_wraps_text(event):
    out = from_reflection(event, "a note")
    assert [e.target for e in out] == ["a note"]
    assert out[0].source == "reflection" and out[0].fact_key is None


def test_replay_buffer_never_exceeds_capacity():
    buf = ReplayBuffer(capacity=5, seed=0)
    buf.add_all([SleepExample("p", f"t{i}", True, "s", "x") for i in range(100)])
    assert len(buf.items) == 5
    assert buf.seen == 100


def test_replay_buffer_keeps_every_item_below_capacity():
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add_all([SleepExample("p", f"t{i}", True, "s", "x") for i in range(4)])
    assert [e.target for e in buf.items] == ["t0", "t1", "t2", "t3"]


def test_replay_buffer_sampling_and_empty_case():
    assert ReplayBuffer().sample(3) == []
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add_all([SleepExample("p", f"t{i}", True, "s", "x") for i in range(4)])
    assert len(buf.sample(10)) == 4
    assert len(buf.sample(2)) == 2


def test_replay_buffer_state_round_trip():
    buf = ReplayBuffer(capacity=3, seed=0)
    buf.add_all([SleepExample("p", f"t{i}", True, "s", "x", fact_key=f"k{i}") for i in range(8)])
    clone = ReplayBuffer()
    clone.load_state_dict(buf.state_dict())
    assert clone.capacity == 3 and clone.seen == 8
    assert [e.to_dict() for e in clone.items] == [e.to_dict() for e in buf.items]
