from common.schema import CompactionEvent, Span
from compactor.segment import count_tokens, segment_turns


class CompactionRunner:
    def __init__(self, compactor, context_budget=1024, keep_recent_turns=8,
                 keep_frac=0.25, granularity="sentence", tokenizer=None):
        self.compactor = compactor
        self.context_budget = context_budget
        self.keep_recent_turns = keep_recent_turns
        self.keep_frac = keep_frac
        self.granularity = granularity
        self.tokenizer = tokenizer

    def _tok(self, text):
        return count_tokens(text, self.tokenizer)

    def run(self, trajectory):
        carry = []
        recent = []
        events = []
        event_id = 0

        for turn in trajectory.turns:
            recent.append(turn)
            ctx_tokens = sum(self._tok(c) for c in carry) + sum(self._tok(t.content) for t in recent)
            if ctx_tokens <= self.context_budget:
                continue

            to_compact = recent[: max(0, len(recent) - self.keep_recent_turns)]
            recent = recent[len(to_compact):]
            if not to_compact:
                continue

            spans = segment_turns(to_compact, self.granularity)
            for i, c in enumerate(carry):
                spans.insert(i, {"span_id": f"carry{event_id}_{i}", "turn_idx": -1,
                                 "role": "system", "text": c, "tags": ["carry"]})
            if not spans:
                continue

            tokens_before = sum(self._tok(s["text"]) for s in spans)
            kept_idx, summary = self.compactor.compact(spans, self.keep_frac)
            kept_set = set(kept_idx)

            span_records = []
            for i, s in enumerate(spans):
                span_records.append(
                    Span(
                        span_id=s["span_id"],
                        turn_idx=s["turn_idx"],
                        text=s["text"],
                        kept=i in kept_set,
                        carries_fact=_fact_key(s["tags"]),
                    )
                )

            carry = [spans[i]["text"] for i in sorted(kept_set)] + [summary]
            tokens_after = sum(self._tok(c) for c in carry)

            events.append(
                CompactionEvent(
                    event_id=event_id,
                    traj_id=trajectory.traj_id,
                    turn_range=[to_compact[0].idx, to_compact[-1].idx],
                    spans=span_records,
                    summary=summary,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    decided_by=getattr(self.compactor, "last_decided_by", self.compactor.name),
                )
            )
            event_id += 1

        final_context = carry + [f"{t.role}: {t.content}" for t in recent]
        return events, final_context


def _fact_key(tags):
    if "fact" in tags and len(tags) > 1:
        return tags[1]
    return None
