COMPACTION_SYSTEM = (
    "You compact a software-engineering conversation so it fits a smaller context budget. "
    "You never invent facts and you never paraphrase a decision."
)

COMPACTION_INSTRUCTION = """Below are numbered spans from a conversation.

Keep a span verbatim only if losing it would make a later question unanswerable: concrete
decisions, names, versions, identifiers, numeric limits, ownership, and commitments.
Drop chit-chat, restatements, generic advice, and anything recoverable from the rest.

Keep at most {budget} spans.

Reply with exactly two lines:
KEEP: <comma-separated span numbers, or NONE>
SUMMARY: <one or two sentences covering the dropped spans>

Spans:
{spans}
"""


def build_compaction_prompt(span_texts, budget):
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(span_texts))
    return COMPACTION_INSTRUCTION.format(budget=budget, spans=numbered)
