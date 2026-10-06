import math

import pytest

from common.stats import T_CRIT, critical, paired_t, verdict_for


def test_verdict_thresholds():
    assert verdict_for(2.0) == "clears control"
    assert verdict_for(1.99) == "indistinguishable"
    assert verdict_for(-2.0) == "below control"
    assert verdict_for(0.0, tie="tie") == "tie"


def test_verdict_nan_label():
    assert verdict_for(float("nan")) == "undetermined"
    assert verdict_for(float("nan"), nan="unknown") == "unknown"


def test_critical_values():
    assert critical(1) == 2.0
    assert critical(3) == 4.303
    assert critical(5) == 2.776
    assert critical(31) == 2.042
    assert critical(60) == 2.0


def test_critical_table_decreases_with_degrees_of_freedom():
    values = [T_CRIT[d] for d in sorted(T_CRIT)]
    assert values == sorted(values, reverse=True)
    assert sorted(T_CRIT) == list(range(1, 31))


def test_paired_t_known_example():
    mean, stat, dof = paired_t([1, 2, 3], [0, 0, 0])
    assert mean == 2
    assert dof == 2
    assert stat == pytest.approx(2 * math.sqrt(3))


def test_paired_t_sign_follows_difference():
    assert paired_t([1, 2, 4], [2, 4, 5])[1] < 0
    assert paired_t([2, 4, 5], [1, 2, 4])[1] > 0


def test_paired_t_zero_variance_is_nan():
    assert math.isnan(paired_t([1, 1, 1], [0, 0, 0])[1])


def test_paired_t_needs_two_pairs():
    with pytest.raises(ValueError):
        paired_t([1], [0])
