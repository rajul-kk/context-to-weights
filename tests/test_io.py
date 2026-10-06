import random

import pytest

from common.io import (deep_merge, load_config, parse_overrides, read_jsonl, set_seed,
                       write_jsonl)


def test_parse_overrides_builds_nested_dicts():
    out = parse_overrides(["sleep.lr=0.1", "gate.floor_weight=0", "model.base=Qwen/Qwen2.5-7B-Instruct"])
    assert out == {"sleep": {"lr": 0.1}, "gate": {"floor_weight": 0},
                   "model": {"base": "Qwen/Qwen2.5-7B-Instruct"}}


def test_parse_overrides_types():
    out = parse_overrides(["a=true", "b=5", "c=null", "d=[1, 2]"])
    assert out == {"a": True, "b": 5, "c": None, "d": [1, 2]}


@pytest.mark.parametrize("raw", ["1e-4", "5E-5", "2.5e3", "1.0e-4", "-3e2", ".5e1"])
def test_scientific_notation_becomes_float(raw):
    value = parse_overrides([f"x={raw}"])["x"]
    assert isinstance(value, float)
    assert value == float(raw)


@pytest.mark.parametrize("raw", ["1e-4x", "e5", "abc", "1e", "v1e5", "nan", "inf"])
def test_other_strings_stay_strings(raw):
    assert parse_overrides([f"x={raw}"])["x"] == raw


def test_parse_overrides_empty():
    assert parse_overrides(None) == {}
    assert parse_overrides([]) == {}


def test_deep_merge_does_not_mutate_inputs():
    a = {"x": {"y": 1, "z": 2}, "k": 1}
    b = {"x": {"y": 9}}
    merged = deep_merge(a, b)
    assert merged == {"x": {"y": 9, "z": 2}, "k": 1}
    assert a == {"x": {"y": 1, "z": 2}, "k": 1}
    assert b == {"x": {"y": 9}}


def test_deep_merge_replaces_non_dict_values():
    assert deep_merge({"x": {"y": 1}}, {"x": 5}) == {"x": 5}


def test_load_config_inherits_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 0\nmodel:\n  base: a\n  dtype: fp16\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nmodel:\n  base: b\n", encoding="utf-8")
    cfg = load_config(tmp_path / "child.yaml", {"seed": 3})
    assert cfg == {"seed": 3, "model": {"base": "b", "dtype": "fp16"}}


def test_load_config_follows_chained_bases(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1\ny: 1\nz: 1\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\ny: 2\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text("_base_: b.yaml\nz: 3\n", encoding="utf-8")
    assert load_config(tmp_path / "c.yaml") == {"x": 1, "y": 2, "z": 3}


def test_jsonl_round_trip_keeps_unicode(tmp_path):
    rows = [{"a": 1, "text": "naïve σ"}, {"a": 2, "text": ""}]
    path = tmp_path / "nested" / "rows.jsonl"
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_set_seed_makes_random_reproducible():
    set_seed(7)
    first = [random.random() for _ in range(3)]
    set_seed(7)
    assert [random.random() for _ in range(3)] == first
