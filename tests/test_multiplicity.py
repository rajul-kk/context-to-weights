import json
from pathlib import Path

import pytest

from common.stats import t_two_sided_p
from eval.multiplicity import families, to_markdown

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "paper" / "figures" / "data.json").read_text(encoding="utf-8"))
FAMS = families(DATA)


def test_each_experiment_has_a_named_family():
    assert set(FAMS) == {"routing", "consolidation", "distillation"}
    assert len(FAMS["routing"]) == 6
    assert len(FAMS["consolidation"]) == len(FAMS["distillation"]) == 1


def test_family_of_one_is_uncorrected():
    for fam in ("consolidation", "distillation"):
        s = FAMS[fam][0]
        assert s["p_holm"] == s["p"] == pytest.approx(t_two_sided_p(s["t"], s["dof"]))


def test_holm_never_lowers_a_p_value():
    for stats in FAMS.values():
        assert all(s["p_holm"] >= s["p"] for s in stats)


def test_routing_family_matches_the_paper():
    by_name = {s["comparison"]: s for s in FAMS["routing"]}
    assert by_name["7B read vs attention_late, hit rate"]["t"] == pytest.approx(12.03, abs=0.01)
    assert by_name["1.5B read vs attention_late, hit rate"]["survives"] is False
    assert to_markdown(FAMS).count("\n") == 1 + sum(len(v) for v in FAMS.values())
