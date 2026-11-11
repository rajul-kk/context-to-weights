import json

import pytest

from declare import run


def _layouts(golds):
    return [{"gold": g} for g in golds]


def test_gold_distribution_is_sorted_and_string_keyed():
    counts = run.gold_counts(_layouts([3, 0, 3, 1]))
    assert list(counts) == [0, 1, 3]
    assert run.gold_distribution(counts) == {"0": 1, "1": 1, "3": 2}


def test_expected_gold_accepts_a_match_and_refuses_a_mismatch():
    got = {"0": 1, "3": 2}
    run.check_expected_gold(got, json.dumps({0: 1, 3: 2}))
    with pytest.raises(SystemExit, match="does not match the expected"):
        run.check_expected_gold(got, json.dumps({0: 3}))


def test_gold_ref_is_written_once_then_checked(tmp_path):
    ref = tmp_path / "gold.json"
    run.check_gold_ref({"0": 1, "3": 2}, ref)
    assert json.loads(ref.read_text()) == {"0": 1, "3": 2}
    run.check_gold_ref({"0": 1, "3": 2}, ref)
    with pytest.raises(SystemExit, match="does not match"):
        run.check_gold_ref({"0": 2, "3": 1}, ref)


def test_take_window_keeps_an_exact_prefix():
    layouts = _layouts(range(5))
    assert run.take_window(layouts, 0, 0) is layouts
    assert run.take_window(layouts, 0, 3) == layouts[:3]


def test_offset_and_take_tile_the_layouts():
    layouts = _layouts(range(384))
    windows = [run.take_window(layouts, off, 96) for off in (0, 96, 192, 288)]
    assert [w[0]["gold"] for w in windows] == [0, 96, 192, 288]
    assert sum(windows, []) == layouts
    assert run.take_window(layouts, 300, 0) == layouts[300:]
    assert run.take_window(layouts, 380, 96) == layouts[380:]


def test_offset_past_the_end_or_negative_is_refused():
    with pytest.raises(SystemExit, match="leaves no layouts"):
        run.take_window(_layouts(range(5)), 5, 0)
    with pytest.raises(SystemExit, match="must not be negative"):
        run.take_window(_layouts(range(5)), -1, 2)


def test_parse_args_defaults():
    args = run.parse_args([])
    assert args.modes == ["generate", "read"]
    assert (args.limit, args.take, args.offset, args.kv_bits, args.kv_scheme) == (0, 0, 0, 0, "token")
