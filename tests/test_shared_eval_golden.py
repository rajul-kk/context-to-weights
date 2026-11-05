import ast
import json
import math
import os
import random
import statistics
from pathlib import Path

import pytest
from helpers import ROOT
from myrios_eval import stats as shared_stats
from myrios_eval.adapters import (from_skill_library_results, myrios_skill_metrics,
                                  skill_library_legacy_finals, skill_library_per_seed)
from myrios_eval.metrics import mean_prompt_tokens, success_rate

from common import stats as old_stats
from eval import declare_report, kvquant_report, significance

DATA = json.loads((ROOT / "paper" / "figures" / "data.json").read_text(encoding="utf-8"))
PAPER = (ROOT / "paper" / "draft_combined.md").read_text(encoding="utf-8")
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "golden_skill_eval.json")
                     .read_text(encoding="utf-8"))
SKILL_LIBRARY = Path(os.environ.get("SKILL_LIBRARY_ROOT", ROOT.parent / "skill-library"))
SL_RESULTS = SKILL_LIBRARY / "results" / "cpu" / "results.json"


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def rates(counts, totals):
    totals = totals if isinstance(totals, list) else [totals] * len(counts)
    return [c / n for c, n in zip(counts, totals)]


def all_paper_pairs():
    pairs = []
    routing = DATA["routing"]
    for scale, modes in routing.items():
        names = sorted(modes)
        for key in ("hit_rate", "content_dependence"):
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    pairs.append(([r[key] for r in modes[a]], [r[key] for r in modes[b]]))
    c = DATA["consolidation"]
    comp = rates(c["compaction_recovered"], c["n_evicted"])
    uni = rates(c["uniform_recovered"], c["n_evicted"])
    pairs += [(uni, comp), (uni[:3], comp[:3])]
    d = DATA["distillation"]
    kl, rnd, un = (rates(d[k], d["n_tasks"]) for k in ("kl_top", "random", "uniform"))
    pairs += [(kl, rnd), (rnd, un), (kl, un)]
    return pairs


def test_paired_t_is_identical_to_common_stats_on_every_paper_pair():
    pairs = all_paper_pairs()
    assert len(pairs) > 30
    for a, b in pairs:
        old = old_stats.paired_t(a, b)
        new = shared_stats.paired_t(a, b)
        assert all(same(x, y) for x, y in zip(old, new))


def test_critical_and_verdict_are_identical_to_common_stats():
    assert shared_stats.T_CRIT == old_stats.T_CRIT
    for n in range(0, 40):
        assert shared_stats.critical(n) == old_stats.critical(n)
    for sigma in (-3.0, -2.0, -1.99, 0.0, 1.99, 2.0, 5.0, float("nan")):
        assert shared_stats.verdict_for(sigma) == old_stats.verdict_for(sigma)


def test_shared_paired_t_reproduces_the_paper_statistics():
    c = DATA["consolidation"]
    comp = rates(c["compaction_recovered"], c["n_evicted"])
    uni = rates(c["uniform_recovered"], c["n_evicted"])
    assert f"t(4) = {shared_stats.paired_t(uni, comp)[1]:.2f}" in PAPER
    assert f"t(2) = {shared_stats.paired_t(uni[:3], comp[:3])[1]:.2f}" in PAPER
    assert round(statistics.mean(uni), 3) == 0.076 and round(statistics.mean(comp), 3) == 0.017
    d = DATA["distillation"]
    kl, rnd, un = (rates(d[k], d["n_tasks"]) for k in ("kl_top", "random", "uniform"))
    for a, b in ((kl, rnd), (rnd, un), (kl, un)):
        assert f"t(4) = {shared_stats.paired_t(a, b)[1]:.2f}" in PAPER
    assert [round(success_rate(x), 3) for x in (kl, rnd, un)] == [
        0.765, 0.744, 0.708]


def test_mean_sd_reproduces_every_routing_cell_and_matches_old_copies():
    for scale, modes in DATA["routing"].items():
        for mode, runs in modes.items():
            for key in ("hit_rate", "content_dependence"):
                values = [r[key] for r in runs]
                m, sd = shared_stats.mean_sd(values)
                assert (m, sd) == kvquant_report.mean_sd(values)
                assert f"{m:.3f} ± {sd:.3f}" in PAPER


def test_aggregate_seeds_matches_declare_report_groups_exactly():
    groups = declare_report.group(declare_report.load(ROOT / "results" / "declare"))
    assert len(groups) == 12
    checked = 0
    for g in groups:
        for key in declare_report.METRICS:
            if key not in g:
                continue
            agg = shared_stats.aggregate_seeds([s[key] for s in g["runs"] if key in s],
                                              n_resamples=200)
            assert agg["mean"] == g[key]
            assert agg["sd"] == g[key + "_sd"]
            assert agg["n"] == g["k"]
            assert agg["ci_low"] <= agg["mean"] <= agg["ci_high"]
            checked += 1
    assert checked > 100


def test_holm_and_binomial_se_match_the_old_functions():
    rng = random.Random(3)
    for _ in range(50):
        ps = [rng.random() for _ in range(rng.randint(1, 8))]
        assert shared_stats.holm(ps) == significance.holm(ps)
    from eval.report import _se
    for acc, n in ((0.123, 114), (0.0, 112), (0.5, 0), (float("nan"), 10), (None, 4)):
        assert shared_stats.binomial_se(acc, n) == _se(acc, n)


def old_skill_summarize():
    source = (ROOT / "eval" / "skill_eval.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "summarize")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "eval/skill_eval.py", "exec"), namespace)
    return namespace["summarize"]


OLD_SUMMARIZE = old_skill_summarize()


@pytest.mark.parametrize("label", sorted(FIXTURE["arms"]))
def test_shared_skill_metrics_reproduce_saved_summaries(label):
    arm = FIXTURE["arms"][label]
    records = arm["records"]
    saved = arm["summary"]
    new = myrios_skill_metrics(records)
    old = OLD_SUMMARIZE(records, label)
    assert len(records) == saved["n"]
    for new_key, old_key in (("success_rate", "pass_rate"),
                             ("success_rate_in_group", "pass_rate_in_group"),
                             ("success_rate_out_group", "pass_rate_out_group"),
                             ("prompt_tokens", "mean_prompt_tokens")):
        assert same(new[new_key], saved[old_key])
        assert same(new[new_key], old[old_key])
    assert mean_prompt_tokens(r["prompt_tokens"] for r in records) == saved["mean_prompt_tokens"]


def test_fixture_matches_local_artifacts_when_present():
    root = ROOT / "artifacts" / "runs_skill" / "report"
    if not root.exists():
        pytest.skip("local skill run artifacts not present")
    for label, arm in FIXTURE["arms"].items():
        saved = json.loads((root / f"{label}.summary.json").read_text(encoding="utf-8"))
        for key, v in arm["summary"].items():
            assert same(v, saved[key])
        lines = (root / f"{label}.records.jsonl").read_text(encoding="utf-8").splitlines()
        recs = [json.loads(line) for line in lines if line.strip()]
        assert [{k: r[k] for k in ("passed", "in_group", "prompt_tokens")} for r in recs] == \
            arm["records"]


@pytest.fixture(scope="module")
def skill_library_results():
    if not SL_RESULTS.exists():
        pytest.skip(f"{SL_RESULTS} not present")
    return json.loads(SL_RESULTS.read_text(encoding="utf-8"))


def test_legacy_rule_reproduces_skill_library_headline_numbers(skill_library_results):
    methods = skill_library_results["methods"]
    assert set(methods) == {"in_context", "seq_no_replay", "isolated", "single_shot", "ours"}
    for key, m in methods.items():
        legacy = skill_library_legacy_finals(m)
        assert legacy["final_success_rate"] == pytest.approx(m["final_success_rate"], abs=1e-12)
        assert legacy["final_earliest_3"] == pytest.approx(m["final_earliest_3"], abs=1e-12)
        assert legacy["final_stored_params"] == m["final_stored_params"]
        assert legacy["mean_prompt_tokens"] == pytest.approx(
            m["tokens"]["mean_prompt_tokens"], abs=1e-9)


def test_per_seed_rule_reproduces_skill_library_per_seed_finals(skill_library_results):
    for key, m in skill_library_results["methods"].items():
        rows = skill_library_per_seed(m)
        assert [r["metrics"]["success_rate"] for r in rows] == pytest.approx(
            m["per_seed_final"], abs=1e-12)


def test_skill_library_payload_converts_to_a_valid_result(skill_library_results, tmp_path):
    from myrios_eval.schema import read_result, write_result
    doc = from_skill_library_results(skill_library_results, "skill-library/cpu_main", "906d77a",
                                     conditions={"library_size": 15},
                                     created="2026-11-05T00:00:00+00:00", n_resamples=200)
    path = write_result(tmp_path / "r.json", doc)
    back = read_result(path)
    arms = {a["arm"]: a for a in back["arms"]}
    assert arms["ours"]["aggregate"]["success_rate"]["n"] == 20
    assert arms["seq_no_replay"]["aggregate"]["forgetting"]["mean"] > \
        arms["ours"]["aggregate"]["forgetting"]["mean"]
    legacy = back["extra"]["legacy_global_last_night"]["ours"]
    assert legacy["final_success_rate"] == pytest.approx(
        skill_library_results["methods"]["ours"]["final_success_rate"])
