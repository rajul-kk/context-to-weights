import pytest

from data.load_locomo import build_trajectory, eligible_answers, fold_split


def sample():
    return {
        "conversation": {
            "session_2": [{"speaker": "Ann", "text": "We moved to Lisbon last spring."}],
            "session_1": [{"speaker": "Bob", "text": "My dog is called Biscuit."},
                          {"speaker": "Ann", "text": "Yes, I remember."}],
        },
        "qa": [
            {"question": "Where did Ann move?", "answer": "lisbon"},
            {"question": "Dog's name?", "answer": "Biscuit"},
            {"question": "Did Ann remember?", "answer": "Yes"},
            {"question": "Favourite colour?", "answer": "green"},
            {"question": "Adversarial?", "adversarial_answer": "x"},
        ],
    }


def test_eligible_answers_filters_verbatim_and_short():
    assert [a for _, a in eligible_answers(sample())] == ["lisbon", "Biscuit", "Yes", "green"]
    assert [a for _, a in eligible_answers(sample(), verbatim_only=True)] == ["lisbon", "Biscuit", "Yes"]
    assert [a for _, a in eligible_answers(sample(), verbatim_only=True, min_answer_chars=4)] == ["lisbon", "Biscuit"]


def test_build_trajectory_matches_facts_case_insensitively_in_session_order():
    t = build_trajectory("t", sample(), 10, verbatim_only=True, min_answer_chars=4)
    assert [p.answer for p in t.probes] == ["lisbon", "Biscuit"]
    assert t.turns[0].content.startswith("Bob:") and t.turns[-1].content.startswith("Ann: We moved")
    assert sorted(f.value for f in t.facts) == ["Biscuit", "lisbon"]


def test_max_probes_applies_after_filtering():
    t = build_trajectory("t", sample(), 1, verbatim_only=True, min_answer_chars=4)
    assert [p.answer for p in t.probes] == ["lisbon"]


def test_fold_split_covers_every_item_once():
    seen = []
    for f in range(2):
        ev, tr = fold_split(10, 2, f)
        assert sorted(ev + tr) == list(range(10)) and not set(ev) & set(tr)
        seen += ev
    assert sorted(seen) == list(range(10))
    with pytest.raises(ValueError):
        fold_split(10, 2, 2)
