import json
import math

import pytest
from myrios_eval import SCHEMA_VERSION, __version__
from myrios_eval.adapters import from_myrios_skill_summary
from myrios_eval.cost import (inference_flops, lora_params, lora_train_flops, full_train_flops,
                              qwen_attention_shapes)
from myrios_eval.metrics import (backward_transfer, earliest_k, final_global_last_night,
                                 final_per_seed, forgetting, history_from_snapshots,
                                 mean_prompt_tokens, success_rate, token_saving)
from myrios_eval.schema import make_arm, make_result, read_result, validate, write_result
from myrios_eval.stats import (aggregate_seeds, bootstrap_ci, compare_arms, paired_bootstrap_ci)


def test_versions_are_set():
    assert __version__ == "0.1.0"
    assert SCHEMA_VERSION == "1.0"


def test_success_rate_accepts_bools_and_fractions_and_is_nan_when_empty():
    assert success_rate([True, False, True, True]) == 0.75
    assert success_rate([0.5, 1.0]) == 0.75
    assert math.isnan(success_rate([]))
    assert math.isnan(mean_prompt_tokens([]))
    assert mean_prompt_tokens([10, 20]) == 15


def test_token_saving_matches_skill_report_definition():
    assert token_saving(250, 1000) == 0.75
    assert math.isnan(token_saving(250, 0))


def snap(seed, night, rate):
    return {"seed": seed, "night": night, "r": rate}


def test_final_per_seed_keeps_seeds_that_end_early():
    snaps = [snap(0, 1, 0.2), snap(0, 2, 0.4), snap(1, 1, 0.9)]
    assert final_per_seed(snaps, lambda s: s["r"]) == {0: 0.4, 1: 0.9}
    assert final_global_last_night(snaps, lambda s: s["r"]) == 0.4


def test_earliest_k_orders_by_age_then_tier():
    skills = [{"skills_since_learned": 5, "tier": 1, "success_rate": 1.0},
              {"skills_since_learned": 5, "tier": 0, "success_rate": 0.0},
              {"skills_since_learned": 1, "tier": 0, "success_rate": 0.5},
              {"skills_since_learned": 3, "tier": 2, "success_rate": 0.5}]
    assert earliest_k(skills, 2) == 0.5
    assert earliest_k(skills, 3) == 0.5
    assert math.isnan(earliest_k([], 3))


def test_backward_transfer_and_forgetting_on_a_known_matrix():
    history = {"a": [(1, 0.8), (2, 0.6), (3, 0.4)],
               "b": [(2, 0.5), (3, 0.7)],
               "c": [(3, 0.9)]}
    assert backward_transfer(history) == pytest.approx(((0.4 - 0.8) + (0.7 - 0.5)) / 2)
    assert forgetting(history) == pytest.approx(((0.8 - 0.4) + (0.5 - 0.7)) / 2)
    assert math.isnan(backward_transfer({"c": [(3, 0.9)]}))
    assert math.isnan(forgetting({}))


def test_history_from_snapshots():
    snaps = [{"night": 1, "skills": [{"skill": "a", "success_rate": 1.0}]},
             {"night": 2, "skills": [{"skill": "a", "success_rate": 0.5},
                                     {"skill": "b", "success_rate": 1.0}]}]
    assert history_from_snapshots(snaps) == {"a": [(1, 1.0), (2, 0.5)], "b": [(2, 1.0)]}


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    values = [0.70, 0.74, 0.73, 0.75, 0.71, 0.72]
    lo, hi = bootstrap_ci(values)
    assert (lo, hi) == bootstrap_ci(values)
    assert lo < sum(values) / len(values) < hi
    assert min(values) <= lo and hi <= max(values)
    assert bootstrap_ci([0.3]) == (0.3, 0.3)
    assert all(math.isnan(v) for v in bootstrap_ci([]))


def test_paired_bootstrap_needs_equal_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1, 2], [1])
    lo, hi = paired_bootstrap_ci([1.0, 1.1, 0.9, 1.2], [0.0, 0.1, 0.0, 0.1])
    assert lo > 0


def test_aggregate_seeds_one_seed_and_none_values():
    one = aggregate_seeds([0.5])
    assert (one["mean"], one["sd"], one["n"], one["ci_low"], one["ci_high"]) == (
        0.5, 0.0, 1, 0.5, 0.5)
    assert aggregate_seeds([None, float("nan")])["n"] == 0
    assert "bootstrap" in aggregate_seeds([0.1, 0.2])["ci"]


def test_compare_arms_pairs_on_common_seeds():
    a = {0: 0.80, 1: 0.82, 2: 0.79, 3: 0.81}
    b = {0: 0.70, 1: 0.71, 2: 0.69, 3: 0.72, 9: 0.0}
    out = compare_arms(a, b)
    assert out["seeds"] == [0, 1, 2, 3] and out["dof"] == 3
    assert out["verdict"] == "a ahead"
    assert out["ci_low"] > 0
    assert compare_arms({0: 1.0}, {0: 0.0})["verdict"] == "undetermined"
    assert compare_arms({0: 0.5, 1: 0.7}, {0: 0.5, 1: 0.7})["verdict"] == "identical"
    assert compare_arms({0: 1.0, 1: 2.0}, {0: 0.0, 1: 1.0})["verdict"] == "undetermined"


def test_lora_params_and_the_skill_library_constant():
    assert lora_params([(1536, 1536)] * 4, 16, n_layers=24) == 4_718_592
    real = lora_params(qwen_attention_shapes(1536, 12, 2), 16, n_layers=28)
    assert real == 28 * 16 * (2 * 3072 + 2 * (1536 + 256))
    assert real != 4_718_592
    with pytest.raises(ValueError):
        lora_params([(4, 4)], -1)


def test_flop_estimates():
    assert inference_flops(1e9, 100) == 2e11
    assert full_train_flops(1e9, 100) == 6e11
    assert lora_train_flops(1e9, 100, 1e6) == 4e11 + 6e8
    assert lora_train_flops(1e9, 100, 1e6) < full_train_flops(1e9, 100)


def arm(name="ours", conditions=None, metrics=None):
    rows = [{"seed": s, "n_items": 10, "metrics": dict(metrics or {"success_rate": 0.5 + s / 10})}
            for s in range(3)]
    return make_arm(name, rows, conditions=conditions)


def doc(**kw):
    return make_result("exp", "Myrios", "abc123", [arm()], created="2026-11-05T00:00:00+00:00",
                       **kw)


def test_result_round_trips_and_writes_nan_as_null(tmp_path):
    d = make_result("exp", "Myrios", "abc", [arm(metrics={"success_rate": float("nan")})],
                    created="2026-11-05T00:00:00+00:00")
    path = write_result(tmp_path / "out.json", d)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    back = read_result(path)
    assert back["arms"][0]["per_seed"][0]["metrics"]["success_rate"] is None
    assert json.loads(text)["schema_version"] == "1.0"


def test_validate_rejects_bad_documents():
    good = doc()
    assert validate(good)
    for mutate, fragment in (
        (lambda d: d.pop("commit"), "missing"),
        (lambda d: d.update(schema_version="2.0"), "major"),
        (lambda d: d["arms"].append(arm()), "duplicate arm"),
        (lambda d: d["arms"][0]["per_seed"][1].update(seed=0), "repeated seed"),
        (lambda d: d["arms"][0]["per_seed"][0].update(seed="0"), "integer"),
        (lambda d: d["arms"][0]["per_seed"][0]["metrics"].update(success_rate="x"), "number"),
        (lambda d: d["arms"][0]["aggregate"].update(other={}), "no per-seed value"),
    ):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        with pytest.raises(ValueError, match=fragment):
            validate(bad)


def test_arms_with_different_conditions_are_distinct():
    make_result("exp", "r", "c", [arm(conditions={"library_size": 10}),
                                  arm(conditions={"library_size": 50})])


def test_tokens_and_flops_require_their_provenance():
    with pytest.raises(ValueError, match="token_counter"):
        make_result("exp", "r", "c", [arm(metrics={"prompt_tokens": 100.0})])
    with pytest.raises(ValueError, match="flops_method"):
        make_result("exp", "r", "c", [arm(metrics={"train_flops": 1e12})])
    make_result("exp", "r", "c", [arm(metrics={"prompt_tokens": 100.0})],
                token_counter="hf:x", prompt_tokens_scope="chat_template/per_call")


def test_myrios_skill_summary_adapter():
    s = {"n": 96, "pass_rate": 0.75, "pass_rate_in_group": 0.8, "pass_rate_out_group": 0.6,
         "mean_prompt_tokens": 61.9}
    a = from_myrios_skill_summary({0: s, 1: dict(s, pass_rate=0.77)}, "ours")
    assert a["aggregate"]["success_rate"]["mean"] == pytest.approx(0.76)
    assert [r["seed"] for r in a["per_seed"]] == [0, 1]
