import math

import pytest

from common.stats import T_CRIT, critical, holm, mcnemar_exact, paired_t, t_two_sided_p, verdict_for


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


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9)


def test_mcnemar_exact_balanced_and_empty_give_one():
    assert mcnemar_exact(4, 4) == 1.0
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_exact_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 3)


@pytest.mark.parametrize("dof", sorted(T_CRIT))
def test_t_two_sided_p_is_five_percent_at_the_critical_value(dof):
    assert t_two_sided_p(T_CRIT[dof], dof) == pytest.approx(0.05, abs=2e-4)
    assert t_two_sided_p(-T_CRIT[dof], dof) == pytest.approx(0.05, abs=2e-4)


def test_t_two_sided_p_closed_forms():
    assert t_two_sided_p(0.0, 3) == pytest.approx(1.0)
    assert t_two_sided_p(1.0, 1) == pytest.approx(0.5)
    assert t_two_sided_p(3.0, 2) == pytest.approx(1 - 3 / math.sqrt(11))
    assert math.isnan(t_two_sided_p(float("nan"), 4))
    with pytest.raises(ValueError):
        t_two_sided_p(1.0, 0)


def test_holm_steps_down_and_keeps_order():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.001]) == pytest.approx([0.5, 0.002])
    assert holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])
    assert holm([]) == []
