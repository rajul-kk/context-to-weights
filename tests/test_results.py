import json
import shutil
import subprocess
import sys

import pytest
from helpers import ROOT

from eval import declare_report

RESULTS = ROOT / "results" / "declare"
PAPER = (ROOT / "paper" / "draft_combined.md").read_text(encoding="utf-8").replace("**", "")


def groups():
    return declare_report.group(declare_report.load(RESULTS))


def scale(group):
    return declare_report.short(group["model"]).split("-")[-1]


def test_results_cover_three_scales_four_modes_three_seeds():
    found = groups()
    assert len(found) == 12
    assert {(scale(g), g["label"]) for g in found} == {
        (s, m) for s in ("0.5B", "1.5B", "7B") for m in ("generate", "attention", "attention_late", "read")}
    assert all(g["seeds"] == [0, 1, 2] and g["n"] == 384 for g in found)


def test_small_and_large_models_were_scored_on_identical_layouts():
    for seed in (0, 1, 2):
        small = json.loads((RESULTS / "gold" / f"gold_s{seed}.json").read_text(encoding="utf-8"))
        large = json.loads((RESULTS / "gold" / f"gold384_s{seed}.json").read_text(encoding="utf-8"))
        assert small == large


def test_gold_files_are_distributions_over_eight_regions():
    paths = sorted((RESULTS / "gold").glob("gold*.json"))
    assert len(paths) == 6
    for path in paths:
        distribution = json.loads(path.read_text(encoding="utf-8"))
        assert len(distribution) == 8
        assert sum(distribution.values()) == 384


def test_data_json_routing_matches_the_raw_summaries():
    data = json.loads((ROOT / "paper" / "figures" / "data.json").read_text(encoding="utf-8"))["routing"]
    for g in groups():
        from_data = data[scale(g)][g["label"]]
        assert [r["seed"] for r in from_data] == g["seeds"]
        for run, row in zip(g["runs"], from_data):
            assert run["hit_rate"] == pytest.approx(row["hit_rate"], abs=1e-12)
            assert run["content_dependence"] == pytest.approx(row["content_dependence"], abs=1e-12)


def test_every_column_of_the_paper_routing_table_matches_the_raw_summaries():
    for g in groups():
        row = (f"| {scale(g)} | {g['label']} | {g['hit_rate']:.3f} ± {g['hit_rate_sd']:.3f} "
               f"| {g['sigma_over_constant']:.1f} "
               f"| {g['content_dependence']:.3f} ± {g['content_dependence_sd']:.3f} "
               f"| {g['slot_stable_rate']:.3f} | {g['unparsed_rate']:.3f} |")
        assert row in PAPER, row


def test_the_report_regenerates_the_papers_headline_verdict(tmp_path):
    copy = tmp_path / "declare"
    shutil.copytree(RESULTS, copy)
    out = tmp_path / "report.md"
    done = subprocess.run([sys.executable, "-m", "eval.declare_report", "--run-root", str(copy),
                           "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    text = out.read_text(encoding="utf-8")
    assert "wins significantly at Qwen2.5-0.5B, Qwen2.5-7B; Qwen2.5-1.5B is a tie, not a reversal" in text
