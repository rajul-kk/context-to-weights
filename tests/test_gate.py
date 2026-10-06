import random

import pytest

from distill.gate import apply_gate, gate_stats, token_weights


def make_row(n_spans=8, span_len=2):
    spans = []
    kl = []
    for s in range(n_spans):
        idx = list(range(s * span_len, (s + 1) * span_len))
        spans.append({"token_idx": idx, "kl_mean": float(n_spans - s)})
        kl.extend([float(n_spans - s)] * span_len)
    return {"kl": kl, "spans": spans}


def active(weights):
    return [i for i, w in enumerate(weights) if w == 1.0]


def test_uniform_weights_every_token_fully():
    row = make_row()
    assert token_weights(row, "uniform", 0.25, "span", 0.0, random.Random(0)) == [1.0] * 16


def test_kl_top_selects_the_highest_scoring_spans():
    row = make_row()
    weights = token_weights(row, "kl_top", 0.25, "span", 0.0, random.Random(0))
    assert active(weights) == [0, 1, 2, 3]
    assert sum(weights) == 4.0


def test_floor_weight_applies_to_unselected_tokens():
    row = make_row()
    weights = token_weights(row, "kl_top", 0.25, "span", 0.1, random.Random(0))
    assert weights[:4] == [1.0] * 4
    assert weights[4:] == [0.1] * 12


def test_random_matches_the_kl_budget_in_active_tokens():
    row = make_row()
    for seed in range(20):
        weights = token_weights(row, "random", 0.25, "span", 0.0, random.Random(seed))
        assert len(active(weights)) == 4


def test_random_selection_varies_with_the_seed():
    row = make_row()
    choices = {tuple(active(token_weights(row, "random", 0.25, "span", 0.0, random.Random(s))))
               for s in range(30)}
    assert len(choices) > 1


def test_random_is_deterministic_for_a_seed():
    row = make_row()
    a = token_weights(row, "random", 0.25, "span", 0.0, random.Random(4))
    b = token_weights(row, "random", 0.25, "span", 0.0, random.Random(4))
    assert a == b


def test_token_granularity_picks_individual_tokens():
    row = {"kl": [0.1, 0.9, 0.2, 0.8], "spans": []}
    weights = token_weights(row, "kl_top", 0.5, "token", 0.0, random.Random(0))
    assert weights == [0.0, 1.0, 0.0, 1.0]


def test_at_least_one_unit_is_always_selected():
    row = make_row()
    weights = token_weights(row, "kl_top", 0.0001, "span", 0.0, random.Random(0))
    assert len(active(weights)) == 2


def test_row_without_spans_falls_back_to_full_weight():
    row = {"kl": [0.5, 0.5, 0.5], "spans": []}
    assert token_weights(row, "kl_top", 0.25, "span", 0.0, random.Random(0)) == [1.0] * 3


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        token_weights(make_row(), "bogus", 0.25, "span", 0.0, random.Random(0))


def test_apply_gate_and_stats():
    cfg = {"gate": {"policy": "kl_top", "top_frac": 0.25, "granularity": "span", "floor_weight": 0.0}}
    rows = apply_gate([make_row(), make_row()], cfg)
    assert all("weights" in r for r in rows)
    stats = gate_stats(rows)
    assert stats == {"tokens": 32, "active_tokens": 8, "active_frac": 0.25}


def test_gate_stats_of_nothing():
    assert gate_stats([])["active_frac"] == 0.0
