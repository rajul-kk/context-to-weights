import pytest

from eval.significance import holm, observed_facts_kept, permutation_test


def event(kept_flags, fact_flags):
    return {"spans": [{"kept": k, "carries_fact": "f" if f else None}
                      for k, f in zip(kept_flags, fact_flags)]}


FACTS = [True, True] + [False] * 8


def kept_with_facts():
    return event([True, True, True, True] + [False] * 6, FACTS)


def kept_without_facts():
    return event([False, False, True, True, True, True] + [False] * 4, FACTS)


def test_holm_known_example():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_is_bounded_and_never_below_the_raw_p():
    raw = [0.001, 0.2, 0.5, 0.9]
    adjusted = holm(raw)
    assert all(a >= p for a, p in zip(adjusted, raw))
    assert all(a <= 1.0 for a in adjusted)


def test_holm_single_value_is_unchanged():
    assert holm([0.04]) == [0.04]


def test_observed_facts_kept_counts_only_kept_fact_spans():
    assert observed_facts_kept([kept_with_facts(), kept_without_facts()]) == 2


def test_permutation_test_detects_a_compactor_that_keeps_facts():
    result = permutation_test([kept_with_facts() for _ in range(6)], n_iter=2000, seed=0)
    assert result["observed_facts_kept"] == 12
    assert result["clustered_sigma"] > 3
    assert result["p_one_sided"] < 0.01


def test_permutation_test_sees_no_signal_when_facts_are_dropped():
    result = permutation_test([kept_without_facts() for _ in range(6)], n_iter=2000, seed=0)
    assert result["observed_facts_kept"] == 0
    assert result["clustered_sigma"] < 0
    assert result["p_one_sided"] > 0.9


def test_permutation_test_is_deterministic_for_a_seed():
    events = [kept_with_facts(), kept_without_facts()] * 3
    assert permutation_test(events, n_iter=500, seed=1) == permutation_test(events, n_iter=500, seed=1)


def test_permutation_test_without_usable_events_returns_none():
    assert permutation_test([event([True, True], [False, False])], n_iter=100) is None
