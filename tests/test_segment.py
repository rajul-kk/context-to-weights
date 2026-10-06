from common.schema import Turn
from compactor.segment import count_tokens, segment_turns, span_tags, split_sentences


def test_split_sentences_splits_on_terminal_punctuation():
    text = "The deploy key rotates every ninety days. The owner is the platform team! Is that clearly understood by all?"
    assert split_sentences(text) == [
        "The deploy key rotates every ninety days.",
        "The owner is the platform team!",
        "Is that clearly understood by all?",
    ]


def test_split_sentences_merges_short_fragments_into_the_previous_one():
    out = split_sentences("The deploy key rotates every ninety days. Ok. Then we moved on to other work.")
    assert out[0] == "The deploy key rotates every ninety days. Ok."
    assert len(out) == 2


def test_split_sentences_of_empty_text():
    assert split_sentences("   ") == []


def test_segment_turns_assigns_stable_span_ids():
    turn = Turn(idx=3, role="assistant", content="First sentence is long enough here. Second sentence is also long enough.")
    spans = segment_turns([turn])
    assert [s["span_id"] for s in spans] == ["t3s0", "t3s1"]
    assert all(s["turn_idx"] == 3 and s["role"] == "assistant" for s in spans)


def test_segment_turns_turn_granularity_keeps_the_whole_turn():
    turn = Turn(idx=0, role="user", content="One sentence that is long enough. Another sentence that is long enough too.")
    assert len(segment_turns([turn], granularity="turn")) == 1
    assert len(segment_turns([turn])) == 2


def test_span_tags_keep_fact_label_only_on_the_marked_sentence():
    tags = ["fact", "auth_header", "x-auth-token"]
    assert span_tags(tags, "We use the x-auth-token header.") == ["fact", "auth_header"]
    assert span_tags(tags, "Unrelated chatter about lunch.") == ["filler"]


def test_span_tags_pass_through_for_non_fact_turns():
    assert span_tags(["filler"], "anything") == ["filler"]
    assert span_tags([], "anything") == []


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_count_tokens_uses_the_tokenizer_when_given():
    assert count_tokens("a b c d", FakeTokenizer()) == 4


def test_count_tokens_falls_back_to_a_word_estimate():
    assert count_tokens("one two three four") == int(4 * 1.35)
    assert count_tokens("") == 1
