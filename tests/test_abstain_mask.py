import pytest

from sleep.abstain import abstain_flags


def contexts():
    return [
        {"context": ["Ann lives in Lisbon.", "The dog is Biscuit."],
         "probes": [{"answer": "Lisbon"}, {"answer": "Biscuit"}, {"answer": "Paris"}]},
        {"context": ["Nothing kept."],
         "probes": [{"answer": "green"}, {"answer": "kept"}, {"answer": "blue"}]},
    ]


def test_compaction_mask_flags_evicted_answers():
    assert abstain_flags(contexts()) == [False, False, True, True, False, True]


def test_random_mask_matches_size_and_is_reproducible():
    real = abstain_flags(contexts())
    a = abstain_flags(contexts(), "random", seed=3)
    assert sum(a) == sum(real) and len(a) == len(real)
    assert a == abstain_flags(contexts(), "random", seed=3)
    assert len({tuple(abstain_flags(contexts(), "random", seed=s)) for s in range(20)}) > 1


def test_unknown_mask():
    with pytest.raises(ValueError):
        abstain_flags(contexts(), "uniform")
