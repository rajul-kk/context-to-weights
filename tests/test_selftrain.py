import json
import subprocess
import sys

import pytest
from helpers import ROOT

from selftrain import gate_report
from selftrain.signals import (auroc, ece, mean_logprob, parse_confidence, verdict,
                               yes_probability)
from selftrain.tasks import FAMILIES, Task, check, extract, make_tasks


def test_tasks_are_deterministic_per_seed_and_differ_across_seeds():
    assert make_tasks(25, 3) == make_tasks(25, 3)
    assert make_tasks(25, 3) != make_tasks(25, 4)


def test_tasks_cycle_through_every_family_evenly():
    tasks = make_tasks(50, 0)
    assert [t.family for t in tasks[:5]] == list(FAMILIES)
    assert {f: sum(t.family == f for t in tasks) for f in FAMILIES} == {f: 10 for f in FAMILIES}


def operands(question, sep):
    body = question.removeprefix("What is ").removesuffix("?")
    return map(int, body.split(sep))


def test_stored_answers_are_correct():
    for t in make_tasks(200, 1):
        if t.family == "add":
            a, b = operands(t.question, " + ")
            assert int(t.answer) == a + b
        if t.family == "mul":
            a, b = operands(t.question, " * ")
            assert int(t.answer) == a * b
        if t.family == "reverse":
            assert t.answer == t.question.split("'")[1][::-1]
        if t.family == "count":
            target, letters = t.question.split("'")[1], t.question.split("'")[3]
            assert int(t.answer) == letters.count(target)
        if t.family == "sort":
            values = [int(v) for v in t.question.split(": ")[1].strip(".").split(", ")]
            assert t.answer == ", ".join(map(str, sorted(values)))


def test_every_task_gold_answer_passes_its_own_checker():
    for t in make_tasks(100, 2):
        assert check(t, f"reasoning\nAnswer: {t.answer}")


def test_checker_rejects_wrong_missing_and_unmarked_answers():
    t = Task("add", "What is 2 + 2?", "4")
    assert not check(t, "Answer: 5")
    assert not check(t, "The sum is 4")
    assert not check(t, "")
    assert not check(t, None)
    assert check(t, "work\nAnswer: 4.")
    assert check(t, "Answer: **4**")


def test_checker_uses_the_last_answer_line():
    t = Task("add", "What is 2 + 2?", "4")
    assert check(t, "Answer: 5\nwait\nAnswer: 4")
    assert not check(t, "Answer: 4\nwait\nAnswer: 5")


def test_checker_handles_thousands_separators_and_lists():
    assert check(Task("mul", "q", "1234"), "Answer: 1,234")
    assert check(Task("sort", "q", "3, 12, 40"), "Answer: [3, 12, 40]")
    assert not check(Task("sort", "q", "3, 12, 40"), "Answer: 12, 3, 40")
    assert check(Task("reverse", "q", "cba"), "Answer: 'c b a'")


def test_extract_returns_none_without_a_marker():
    assert extract("nothing here") is None
    assert extract("Answer: 7") == "7"


def test_auroc_known_values():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75


def test_auroc_is_undefined_with_one_class():
    assert auroc([0.1, 0.2], [1, 1]) is None
    assert auroc([0.1, 0.2], [0, 0]) is None


def test_auroc_accepts_booleans_and_ties_across_classes():
    assert auroc([1, 1, 2], [False, True, True]) == pytest.approx(0.75)


def test_ece_is_zero_when_calibrated_and_large_when_not():
    assert ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]) == 0.0
    assert ece([1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0]) == 1.0
    assert ece([0.5, 0.5], [1, 0]) == 0.0
    assert ece([], []) is None


def test_ece_puts_confidence_one_in_the_last_bin():
    assert ece([1.0], [1]) == 0.0


def test_parse_confidence_formats():
    assert parse_confidence("85") == 0.85
    assert parse_confidence("85%") == 0.85
    assert parse_confidence("0.7") == 0.7
    assert parse_confidence("1") == 0.01
    assert parse_confidence("150") == 1.0
    assert parse_confidence("unsure") is None
    assert parse_confidence(None) is None


def test_mean_logprob_and_yes_probability():
    assert mean_logprob([-1.0, -3.0]) == -2.0
    assert mean_logprob([]) == -50.0
    assert yes_probability(0.0, 0.0) == 0.5
    assert yes_probability(5.0, 0.0) > 0.99
    assert yes_probability(0.0, 5.0) < 0.01


def seeds(measured, stated):
    return [{"pyes": m, "verbal": s} for m, s in zip(measured, stated)]


def test_verdict_passes_when_measured_clearly_beats_stated():
    v = verdict(seeds([0.72, 0.70, 0.74], [0.52, 0.55, 0.50]))
    assert v["passed"] and all(v["checks"].values())


def test_verdict_fails_below_the_floor():
    v = verdict(seeds([0.58, 0.57, 0.59], [0.40, 0.41, 0.42]))
    assert not v["passed"] and "measured_above_floor" in v["reason"]


def test_verdict_fails_on_a_small_gap():
    v = verdict(seeds([0.70, 0.71, 0.72], [0.68, 0.69, 0.67]))
    assert not v["passed"] and "gap_at_least_margin" in v["reason"]


def test_verdict_fails_when_seed_ranges_overlap():
    v = verdict(seeds([0.80, 0.62, 0.80], [0.50, 0.66, 0.50]))
    assert not v["passed"] and "seed_ranges_disjoint" in v["reason"]


def test_verdict_needs_enough_seeds():
    v = verdict(seeds([0.9, 0.9], [0.5, 0.5]))
    assert not v["passed"] and "seeds" in v["reason"]


def summary(model, seed, pyes, verbal):
    return {"model": model, "seed": seed, "n": 100, "pass_rate": 0.4,
            "pass_by_family": {"add": 0.9, "mul": 0.1},
            "auroc": {"pyes": pyes, "logprob": 0.6, "verbal": verbal},
            "ece": {"pyes": 0.1, "verbal": 0.3}, "unparsed_verbal_rate": 0.05}


def write_runs(root, model, values):
    for seed, (pyes, verbal) in enumerate(values):
        folder = root / f"{model.split('/')[-1]}_s{seed}"
        folder.mkdir(parents=True)
        (folder / "g1.summary.json").write_text(json.dumps(summary(model, seed, pyes, verbal)),
                                                encoding="utf-8")


def test_report_groups_models_and_renders_a_verdict(tmp_path):
    write_runs(tmp_path, "org/Small", [(0.72, 0.5), (0.7, 0.52), (0.74, 0.49)])
    write_runs(tmp_path, "org/Large", [(0.6, 0.58), (0.61, 0.6), (0.59, 0.57)])
    text, verdicts = gate_report.render(gate_report.group(gate_report.load(tmp_path)))
    assert verdicts["org/Small"]["passed"] and not verdicts["org/Large"]["passed"]
    assert "- Small: PASS" in text and "- Large: FAIL" in text
    assert "add 0.90, mul 0.10" in text


def test_report_cli_writes_the_markdown(tmp_path):
    write_runs(tmp_path / "runs", "org/Small", [(0.72, 0.5), (0.7, 0.52), (0.74, 0.49)])
    out = tmp_path / "g1.md"
    done = subprocess.run([sys.executable, "-m", "selftrain.gate_report", "--run-root",
                           str(tmp_path / "runs"), "--out", str(out)], cwd=ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "Small: PASS" in out.read_text(encoding="utf-8")


def test_report_cli_fails_on_an_empty_run_root(tmp_path):
    done = subprocess.run([sys.executable, "-m", "selftrain.gate_report", "--run-root",
                           str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode != 0 and "no g1.summary.json" in done.stderr
