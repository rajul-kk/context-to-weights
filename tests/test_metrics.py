import math

from eval.metrics import aggregate, answer_match, median_mean, normalize


def test_normalize_strips_case_punctuation_and_whitespace():
    assert normalize("  Port:   5432!  ") == "port 5432"


def test_normalize_keeps_path_and_version_characters():
    assert normalize("v1.2/api+x-y") == "v1.2/api+x-y"


def test_answer_match_is_case_and_punctuation_insensitive():
    assert answer_match("The header is X-Auth-Token.", "x-auth-token")
    assert answer_match("It uses PostgreSQL, I think", "postgresql")


def test_answer_match_uses_aliases():
    assert answer_match("we picked pg", "postgresql", aliases=["pg"])
    assert not answer_match("we picked mysql", "postgresql", aliases=["pg"])


def test_answer_match_ignores_empty_gold():
    assert not answer_match("anything", "")
    assert not answer_match("anything", "", aliases=[""])


def test_median_mean_handles_empty_and_values():
    empty = median_mean([])
    assert math.isnan(empty["median"]) and empty["n"] == 0
    assert median_mean([1.0, 2.0, 9.0]) == {"median": 2.0, "mean": 4.0, "n": 3}


def record(correct, in_context, ce=None):
    return {"correct": correct, "fact_in_context": in_context, "prompt_tokens": 10, "ce": ce or []}


def test_aggregate_separates_evicted_accuracy():
    out = aggregate([record(True, True), record(False, True), record(True, False),
                     record(False, False), record(False, False)])
    assert out["n"] == 5
    assert out["retention_accuracy"] == 0.4
    assert out["n_evicted"] == 3
    assert out["evicted_accuracy"] == 1 / 3


def test_aggregate_evicted_accuracy_is_nan_without_evicted_probes():
    out = aggregate([record(True, True), record(False, True)])
    assert out["n_evicted"] == 0
    assert math.isnan(out["evicted_accuracy"])


def test_aggregate_cross_entropy_statistics():
    out = aggregate([record(True, False, ce=[1.0, 3.0]), record(True, True, ce=[2.0])])
    assert out["token_ce_median"] == 2.0
    assert out["token_ce_mean"] == 2.0
    assert out["probe_ce_median"] == 2.0
    assert out["evicted_ce_median"] == 2.0
    assert out["evicted_ce_mean"] == 2.0


def test_aggregate_of_nothing():
    out = aggregate([])
    assert out["n"] == 0
    assert out["retention_accuracy"] == 0.0
