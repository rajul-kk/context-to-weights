import json
import subprocess
import sys
from pathlib import Path

import pytest

from common.stats import mcnemar_exact, paired_t
from eval import declare_report

ROOT = Path(__file__).resolve().parents[1]

CHANCE = {"random_control": 0.125, "best_constant_control": 0.148, "n": 384}


def write_run(root, model, seed, mode, hit, cd):
    folder = root / f"{model.split('/')[-1]}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    summary = {"label": mode, "model": model, "seed": seed, "hit_rate": hit, "content_dependence": cd,
               "sigma_over_constant": 5.0, **CHANCE}
    (folder / f"{mode}.summary.json").write_text(json.dumps(summary), encoding="utf-8")


def populate(root, model, read_cd, late_cd, seeds=(0, 1, 2)):
    for seed, r, a in zip(seeds, read_cd, late_cd):
        write_run(root, model, seed, "read", 0.3 + r, r)
        write_run(root, model, seed, "attention_late", 0.3 + a, a)
        write_run(root, model, seed, "attention", 0.2, 0.05)


def report(root, tmp_path):
    out = tmp_path / "report.md"
    done = subprocess.run([sys.executable, "-m", "eval.declare_report", "--run-root", str(root),
                           "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return out.read_text(encoding="utf-8") if out.exists() else done.stdout


WIN = ([0.20, 0.21, 0.205], [0.11, 0.12, 0.10])
TIE = ([0.24, 0.25, 0.23], [0.25, 0.23, 0.24])
LOSS = ([0.10, 0.11, 0.105], [0.20, 0.21, 0.19])


def test_load_reads_per_mode_summaries_and_defaults_the_seed(tmp_path):
    write_run(tmp_path, "org/Qwen2.5-0.5B-Instruct", 2, "read", 0.3, 0.2)
    folder = tmp_path / "Qwen2.5-0.5B-Instruct_s0"
    folder.mkdir()
    (folder / "read.summary.json").write_text(json.dumps({"label": "read", "model": "m", "n": 5}),
                                              encoding="utf-8")
    seeds = sorted(r["seed"] for r in declare_report.load(tmp_path))
    assert seeds == [0, 2]


def test_load_skips_summaries_without_probes(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "read.summary.json").write_text(json.dumps({"label": "read", "n": 0}), encoding="utf-8")
    assert declare_report.load(tmp_path) == []


def test_group_averages_over_seeds(tmp_path):
    populate(tmp_path, "org/Qwen2.5-0.5B-Instruct", *WIN)
    groups = declare_report.group(declare_report.load(tmp_path))
    read = next(g for g in groups if g["label"] == "read")
    assert read["k"] == 3
    assert read["content_dependence"] == pytest.approx(sum(WIN[0]) / 3)
    assert read["seeds"] == [0, 1, 2]


def test_paired_matches_the_shared_paired_t(tmp_path):
    populate(tmp_path, "org/Qwen2.5-0.5B-Instruct", *WIN)
    groups = declare_report.group(declare_report.load(tmp_path))
    read = next(g for g in groups if g["label"] == "read")
    late = next(g for g in groups if g["label"] == "attention_late")
    diff, stat, name, runs = declare_report.paired(read, late, "content_dependence")
    mean, expected, dof = paired_t(WIN[0], WIN[1])
    assert (diff, stat, runs, name) == (pytest.approx(mean), pytest.approx(expected), 3, f"t({dof})")


def test_paired_uses_only_shared_seeds(tmp_path):
    populate(tmp_path, "org/Qwen2.5-0.5B-Instruct", *WIN)
    write_run(tmp_path, "org/Qwen2.5-0.5B-Instruct", 9, "read", 0.9, 0.9)
    groups = declare_report.group(declare_report.load(tmp_path))
    read = next(g for g in groups if g["label"] == "read")
    late = next(g for g in groups if g["label"] == "attention_late")
    assert declare_report.paired(read, late, "content_dependence")[3] == 3


def test_paired_falls_back_to_a_binomial_sigma_for_one_seed(tmp_path):
    populate(tmp_path, "org/Qwen2.5-0.5B-Instruct", [0.2], [0.1], seeds=(0,))
    groups = declare_report.group(declare_report.load(tmp_path))
    read = next(g for g in groups if g["label"] == "read")
    late = next(g for g in groups if g["label"] == "attention_late")
    _, _, name, runs = declare_report.paired(read, late, "hit_rate")
    assert (name, runs) == ("sigma", 1)


def test_a_significant_win_and_a_tie_are_reported_as_such(tmp_path):
    populate(tmp_path, "org/Qwen2.5-0.5B-Instruct", *WIN)
    populate(tmp_path, "org/Qwen2.5-1.5B-Instruct", *TIE)
    text = report(tmp_path, tmp_path)
    assert "wins significantly at Qwen2.5-0.5B" in text
    assert "Qwen2.5-1.5B is a tie, not a reversal" in text
    assert "reverses" not in text


def test_a_reversal_needs_both_directions_to_be_significant(tmp_path):
    populate(tmp_path, "org/Qwen2.5-0.5B-Instruct", *WIN)
    populate(tmp_path, "org/Qwen2.5-7B-Instruct", *LOSS)
    text = report(tmp_path, tmp_path)
    assert "The ordering reverses" in text
    assert "wins significantly at Qwen2.5-0.5B" in text
    assert "loses significantly at Qwen2.5-7B" in text


def test_a_single_seed_run_is_flagged(tmp_path):
    populate(tmp_path, "org/Qwen2.5-0.5B-Instruct", [0.2], [0.1], seeds=(0,))
    assert "Single seed" in report(tmp_path, tmp_path)


def test_unbalanced_seeds_are_flagged(tmp_path):
    populate(tmp_path, "org/Qwen2.5-0.5B-Instruct", *WIN)
    populate(tmp_path, "org/Qwen2.5-1.5B-Instruct", [0.2, 0.21], [0.1, 0.12], seeds=(0, 1))
    assert "Unbalanced seeds" in report(tmp_path, tmp_path)


def test_an_empty_run_root_fails_cleanly(tmp_path):
    done = subprocess.run([sys.executable, "-m", "eval.declare_report", "--run-root", str(tmp_path)],
                          cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 1
    assert "no summaries" in done.stdout


def write_records(root, model, seed, mode, hits, questions=None):
    folder = root / f"{model.split('/')[-1]}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    questions = questions or [f"q{i}" for i in range(len(hits))]
    summary = {"label": mode, "model": model, "seed": seed, "hit_rate": sum(hits) / len(hits),
               "content_dependence": 0.1, "sigma_over_constant": 5.0, **CHANCE, "n": len(hits)}
    (folder / f"{mode}.summary.json").write_text(json.dumps(summary), encoding="utf-8")
    lines = [json.dumps({"question": q, "hit": bool(h)}) for q, h in zip(questions, hits)]
    (folder / f"{mode}.records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_probe_level_counts_discordant_pairs_per_seed(tmp_path):
    model = "org/Qwen3.8-27B"
    write_records(tmp_path, model, 0, "read", [1, 1, 1, 0, 0, 1])
    write_records(tmp_path, model, 0, "generate", [0, 0, 1, 1, 0, 1])
    got = declare_report.probe_level(declare_report.load_records(tmp_path), model)
    seed0 = got["per_seed"][0]
    assert (seed0["seed"], seed0["only_read"], seed0["only_generate"]) == (0, 2, 1)
    assert seed0["p"] == pytest.approx(mcnemar_exact(2, 1))
    assert got["n_probes"] == 6


def test_probe_level_pairs_seeds_by_question_and_bootstraps_over_probes(tmp_path):
    model = "org/Qwen3.8-27B"
    qs = ["a", "b", "c", "d"]
    write_records(tmp_path, model, 0, "read", [1, 1, 0, 0], qs)
    write_records(tmp_path, model, 0, "generate", [1, 0, 0, 0], qs)
    write_records(tmp_path, model, 1, "read", [0, 1, 1, 0], ["d", "a", "b", "c"])
    write_records(tmp_path, model, 1, "generate", [0, 1, 0, 0], ["d", "a", "b", "c"])
    got = declare_report.probe_level(declare_report.load_records(tmp_path), model)
    assert got["n_probes"] == 4
    assert got["seeds"] == [0, 1]
    assert got["mean_diff"] == pytest.approx((0 + 1 + 0 + 0) / 4)
    lo, hi = got["ci"]
    assert lo <= got["mean_diff"] <= hi


def test_probe_level_refuses_mismatched_probe_sets(tmp_path):
    model = "org/Qwen3.8-27B"
    write_records(tmp_path, model, 0, "read", [1, 0], ["a", "b"])
    write_records(tmp_path, model, 0, "generate", [1, 0], ["a", "c"])
    with pytest.raises(ValueError):
        declare_report.probe_level(declare_report.load_records(tmp_path), model)


def test_report_includes_probe_level_section_when_records_exist(tmp_path):
    model = "org/Qwen3.8-27B"
    write_records(tmp_path, model, 0, "read", [1, 1, 1, 0, 0, 1])
    write_records(tmp_path, model, 0, "generate", [0, 0, 1, 1, 0, 1])
    text = report(tmp_path, tmp_path)
    assert "## Probe level: read against generate" in text
    assert "exact McNemar" in text


def test_report_omits_probe_level_section_without_records(tmp_path):
    populate(tmp_path, "org/Qwen2.5-0.5B-Instruct", *WIN)
    assert "Probe level" not in report(tmp_path, tmp_path)
