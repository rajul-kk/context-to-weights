import json
import statistics
from pathlib import Path

import pytest

from common.stats import paired_t

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "paper" / "figures" / "data.json").read_text(encoding="utf-8"))
PAPER = (ROOT / "paper" / "draft_combined.md").read_text(encoding="utf-8")
ROUTING = DATA["routing"]
SCALES = ("0.5B", "1.5B", "7B")
MODES = ("generate", "attention", "attention_late", "read")


def series(scale, mode, key):
    return [r[key] for r in ROUTING[scale][mode]]


def rates(counts, totals):
    totals = totals if isinstance(totals, list) else [totals] * len(counts)
    return [c / n for c, n in zip(counts, totals)]


def minus(text):
    return text.replace("-", "−")


def stated(prefix, value, dof):
    return f"{prefix}({dof}) = {value}"


@pytest.mark.parametrize("scale,key,expected", [
    ("0.5B", "hit_rate", 24.88), ("0.5B", "content_dependence", 4.91),
    ("1.5B", "hit_rate", 0.30), ("1.5B", "content_dependence", -0.85),
    ("7B", "hit_rate", 12.03), ("7B", "content_dependence", 4.69),
])
def test_read_versus_late_attention(scale, key, expected):
    _, stat, dof = paired_t(series(scale, "read", key), series(scale, "attention_late", key))
    assert dof == 2
    assert stat == pytest.approx(expected, abs=0.006)
    assert minus(f"t(2) = {expected:.2f}") in PAPER or minus(f"t(2) = {expected:.1f}") in PAPER


@pytest.mark.parametrize("key,expected,shown", [
    ("hit_rate", 24.0, "t(2) = 24.0"), ("content_dependence", 12.34, "t(2) = 12.3"),
])
def test_read_versus_generate_at_7b(key, expected, shown):
    _, stat, _ = paired_t(series("7B", "read", key), series("7B", "generate", key))
    assert stat == pytest.approx(expected, abs=0.006)
    assert shown in PAPER


def test_routing_table_rows_match_the_data():
    for scale in SCALES:
        for mode in MODES:
            for key in ("hit_rate", "content_dependence"):
                values = series(scale, mode, key)
                cell = f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"
                assert cell in PAPER, (scale, mode, key, cell)


def test_consolidation_statistics():
    c = DATA["consolidation"]
    compaction = rates(c["compaction_recovered"], c["n_evicted"])
    uniform = rates(c["uniform_recovered"], c["n_evicted"])
    assert statistics.mean(uniform) == pytest.approx(0.076, abs=0.0005)
    assert statistics.mean(compaction) == pytest.approx(0.017, abs=0.0005)
    assert paired_t(uniform, compaction)[1] == pytest.approx(3.10, abs=0.006)
    assert paired_t(uniform[:3], compaction[:3])[1] == pytest.approx(1.81, abs=0.006)
    assert "t(4) = 3.10" in PAPER and "t(2) = 1.81" in PAPER


def test_consolidation_counts_are_consistent():
    c = DATA["consolidation"]
    for name in ("compaction_recovered", "uniform_recovered"):
        assert all(0 <= k <= n for k, n in zip(c[name], c["n_evicted"]))
    assert c["n_evicted"] == [114, 102, 112, 98, 112]


def test_distillation_statistics():
    d = DATA["distillation"]
    kl, random_, uniform = (rates(d[k], d["n_tasks"]) for k in ("kl_top", "random", "uniform"))
    diff, stat, _ = paired_t(kl, random_)
    assert (round(diff, 3), round(stat, 2)) == (0.021, 0.74)
    diff, stat, _ = paired_t(random_, uniform)
    assert (round(diff, 3), round(stat, 2)) == (0.035, 3.90)
    diff, stat, _ = paired_t(kl, uniform)
    assert (round(diff, 3), round(stat, 2)) == (0.056, 2.33)
    means = [statistics.mean(x) for x in (kl, random_, uniform)]
    assert [round(m, 3) for m in means] == [0.765, 0.744, 0.708]
    for text in ("t(4) = 0.74", "t(4) = 3.90", "t(4) = 2.33", "`random` by 0.035"):
        assert text in PAPER


def test_distillation_table_matches_the_counts():
    d = DATA["distillation"]
    for seed, kl, rnd, uni in zip(d["seed"], d["kl_top"], d["random"], d["uniform"]):
        row = "| {} | {} | {} | {} |".format(seed, *(f"{v / d['n_tasks']:.3f}" for v in (kl, rnd, uni)))
        near = [row, row.replace("0.812", "0.813")]
        assert any(r in PAPER for r in near), row
