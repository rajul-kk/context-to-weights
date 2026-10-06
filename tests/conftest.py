import pytest

from common.schema import CompactionEvent, Span


@pytest.fixture
def make_event():
    def build(kept, dropped, traj_id="t0"):
        spans = []
        for i, (text, fact) in enumerate(kept):
            spans.append(Span(span_id=f"k{i}", turn_idx=i, text=text, kept=True, carries_fact=fact))
        for i, (text, fact) in enumerate(dropped):
            spans.append(Span(span_id=f"d{i}", turn_idx=100 + i, text=text, kept=False,
                              carries_fact=fact))
        return CompactionEvent(event_id=0, traj_id=traj_id, turn_range=[0, 1], spans=spans,
                               summary="", tokens_before=100, tokens_after=40)

    return build
