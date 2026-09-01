COMPACTION_SYSTEM = (
    "You compact a software-engineering conversation so it fits a smaller context budget. "
    "You never invent facts and you never paraphrase a decision. "
    "You reply in the requested format and nothing else."
)

COMPACTION_INSTRUCTION = """Numbered spans from a conversation:

{spans}

Select the spans worth keeping verbatim. Keep a span only if losing it would make a later
question unanswerable: concrete decisions, names, versions, identifiers, numeric limits,
ownership, commitments. Drop chit-chat, restatements, generic advice, and anything
recoverable from the rest.

Keep at most {budget} of the {n_spans} spans above.

Reply with exactly two lines and nothing else. Do not repeat the spans.

KEEP: <comma-separated span numbers, or NONE>
SUMMARY: <one or two sentences covering the dropped spans>

For example, if spans 2, 5 and 9 were the ones worth keeping:

KEEP: 2, 5, 9
SUMMARY: The rest was scheduling talk and a lint cleanup with no decisions attached.

Now give your answer for the {n_spans} spans above.
"""


def build_compaction_prompt(span_texts, budget):
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(span_texts))
    return COMPACTION_INSTRUCTION.format(budget=budget, spans=numbered,
                                         n_spans=len(span_texts))
