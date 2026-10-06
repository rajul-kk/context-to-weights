import random

from common.schema import Turn
from declare.regions import (build_regions, gold_region, permute, region_tokens, render,
                             render_spans)


def turns(n, words=10):
    return [Turn(idx=i, role="user", content=" ".join(f"w{i}x{j}" for j in range(words)))
            for i in range(n)]


def test_build_regions_returns_exactly_n_regions_and_keeps_every_line_in_order():
    regions = build_regions(turns(40), 8)
    assert len(regions) == 8
    flat = [line for r in regions for line in r]
    assert flat == [f"user: {t.content}" for t in turns(40)]


def test_build_regions_balances_token_cost():
    regions = build_regions(turns(40), 8)
    sizes = [len(r) for r in regions]
    assert max(sizes) - min(sizes) <= 1


def test_build_regions_pads_with_empty_regions():
    regions = build_regions(turns(2), 8)
    assert len(regions) == 8
    assert sum(1 for r in regions if not r) >= 6


def test_gold_region_finds_the_answer_or_a_marker():
    regions = [["alpha"], ["the key is 42"], ["omega marker here"]]
    assert gold_region(regions, "42") == 1
    assert gold_region(regions, "missing", markers=["marker"]) == 2
    assert gold_region(regions, "missing") is None
    assert gold_region(regions, "") is None


def test_render_labels_every_region_and_marks_empty_ones():
    text, tokens = render([["a b"], []])
    assert "[REGION 0]\na b" in text
    assert "[REGION 1]\n(empty)" in text
    assert tokens > 0


def test_render_spans_offsets_point_at_each_block():
    regions = [["first"], ["second line", "third"], []]
    text, spans = render_spans(regions)
    assert len(spans) == 3
    for i, (start, end) in enumerate(spans):
        assert text[start:end].startswith(f"[REGION {i}]")
    assert text == render_spans(regions)[0]


def test_region_tokens_counts_each_region():
    counts = region_tokens([["one two three"], []])
    assert counts[0] > counts[1] >= 1


def test_permute_is_a_permutation_with_a_matching_order():
    regions = [[f"r{i}"] for i in range(8)]
    shuffled, order = permute(regions, random.Random(0))
    assert sorted(order) == list(range(8))
    assert shuffled == [regions[i] for i in order]
