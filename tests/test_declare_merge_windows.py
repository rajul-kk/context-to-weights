import random

import pytest

from common.io import read_json, read_jsonl, write_json, write_jsonl
from declare import merge_windows, run


def _records(n, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        gold, choice, shuffled_gold = rng.randrange(8), rng.randrange(8), rng.randrange(8)
        pchoice = rng.randrange(8)
        out.append({"question": f"q{seed}-{i}", "gold": gold, "choice": choice, "hit": choice == gold,
                    "n_regions": 8, "attended_frac": 0.125, "raw": "", "shuffled_gold": shuffled_gold,
                    "shuffled_choice": pchoice, "shuffled_hit": pchoice == shuffled_gold,
                    "slot_stable": pchoice == choice})
    return out


def _window(path, mode, records, offset):
    s = run.summarize(records, mode, 8)
    s.update(model="m", seed=0, kv_bits=0, kv_scheme=None, elicitor_unparsed=1, layout_offset=offset)
    write_jsonl(path / f"{mode}.records.jsonl", records)
    write_json(path / f"{mode}.summary.json", s)


def test_windows_merge_to_the_single_run_summary(tmp_path):
    recs = _records(30, 1)
    for i, (lo, hi) in enumerate(((0, 12), (12, 20), (20, 30))):
        _window(tmp_path / f"w{i}", "read", recs[lo:hi], lo)
    merged_recs, s = merge_windows.merge_mode([tmp_path / "w2", tmp_path / "w0", tmp_path / "w1"], "read")
    assert merged_recs == recs
    whole = run.summarize(recs, "read", 8)
    for k, v in whole.items():
        assert s[k] == pytest.approx(v) if isinstance(v, float) else s[k] == v
    assert s["elicitor_unparsed"] == 3 and s["merged_windows"] == [[0, 12], [12, 8], [20, 10]]


def test_a_window_without_layout_offset_counts_as_the_first(tmp_path):
    recs = _records(10, 2)
    _window(tmp_path / "a", "read", recs[:6], 0)
    s = read_json(tmp_path / "a" / "read.summary.json")
    del s["layout_offset"]
    write_json(tmp_path / "a" / "read.summary.json", s)
    _window(tmp_path / "b", "read", recs[6:], 6)
    assert merge_windows.merge_mode([tmp_path / "a", tmp_path / "b"], "read")[0] == recs


def test_gaps_and_mismatched_settings_are_refused(tmp_path):
    recs = _records(20, 3)
    _window(tmp_path / "a", "read", recs[:8], 0)
    _window(tmp_path / "b", "read", recs[10:], 10)
    with pytest.raises(SystemExit, match="do not tile"):
        merge_windows.merge_mode([tmp_path / "a", tmp_path / "b"], "read")
    _window(tmp_path / "c", "read", recs[8:], 8)
    s = read_json(tmp_path / "c" / "read.summary.json")
    s["seed"] = 1
    write_json(tmp_path / "c" / "read.summary.json", s)
    with pytest.raises(SystemExit, match="seed differs"):
        merge_windows.merge_mode([tmp_path / "a", tmp_path / "c"], "read")


def test_main_writes_every_mode_and_summary_all(tmp_path):
    recs = _records(16, 4)
    for mode in ("read", "generate"):
        _window(tmp_path / "w0", mode, recs[:8], 0)
        _window(tmp_path / "w1", mode, recs[8:], 8)
    merge_windows.main(["--out", str(tmp_path / "all"), str(tmp_path / "w0"), str(tmp_path / "w1")])
    assert len(read_jsonl(tmp_path / "all" / "generate.records.jsonl")) == 16
    assert sorted(s["label"] for s in read_json(tmp_path / "all" / "summary_all.json")) == ["generate", "read"]
