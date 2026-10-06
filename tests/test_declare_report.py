import json
import subprocess
import sys
from pathlib import Path

import pytest

from common.stats import paired_t
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
