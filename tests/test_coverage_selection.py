import random
from collections import Counter

import pytest

from common.schema import CompactionEvent, Span
from sleep.examples import from_coverage
from sleep.loop import build_examples


def _event(spec):
    spans = [Span(span_id=f"s{i}", turn_idx=t, text=f"span {i}", kept=k, carries_fact=f)
             for i, (t, k, f) in enumerate(spec)]
    return CompactionEvent(event_id=0, traj_id="t0", turn_range=[0, 9], spans=spans, summary="",
                           tokens_before=0, tokens_after=0)


SPEC = [(0, True, "a"), (1, True, "a"), (2, True, "a"), (3, False, "b"), (4, False, "b"),
        (5, False, "c"), (6, False, None), (7, False, None)]


def test_fact_coverage_keeps_the_budget_and_spreads_over_keys():
    out = from_coverage(_event(SPEC), random.Random(0), by="fact")
    assert len(out) == 3
    assert len({e.fact_key for e in out}) == 3
    assert all(e.source == "coverage_fact" and e.kept for e in out)


def test_salience_decides_inside_a_group():
    for seed in range(20):
        out = from_coverage(_event(SPEC), random.Random(seed), by="fact")
        a = [e for e in out if e.fact_key == "a"]
        assert all(e.target in ("span 0", "span 1", "span 2") for e in a)


def test_turn_coverage_draws_one_span_per_turn_bucket():
    out = from_coverage(_event(SPEC), random.Random(1), by="turn")
    turns = sorted(int(e.target.split()[1]) for e in out)
    assert turns[0] in (0, 1) and turns[1] == 2 and turns[2] in (5, 6, 7)


def test_budget_larger_than_groups_round_robins_and_is_seeded():
    ev = _event(SPEC)
    out = from_coverage(ev, random.Random(3), by="fact", n=6)
    assert len(out) == 6 and max(Counter(e.fact_key for e in out).values()) == 2
    again = from_coverage(ev, random.Random(3), by="fact", n=6)
    assert [e.target for e in out] == [e.target for e in again]
    assert len(from_coverage(ev, random.Random(0), by="fact", n=99)) == len(SPEC)


def test_unknown_key_and_loop_dispatch():
    with pytest.raises(ValueError, match="unknown coverage key"):
        from_coverage(_event(SPEC), random.Random(0), by="topic")
    out = build_examples("coverage", [_event(SPEC)], random.Random(0), {}, include_dropped=False,
                         coverage_by="turn")
    assert len(out) == 3 and {e.source for e in out} == {"coverage_turn"}
