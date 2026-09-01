COMPACTION_SYSTEM = (
    "You compact a software-engineering conversation so it fits a smaller context budget. "
    "You never invent facts and you never paraphrase a decision. "
    "You reply in the requested format and nothing else."
)

COMPACTION_INSTRUCTION = """Numbered spans from a conversation:

{spans}

Select the {budget} spans out of {n_spans} that are most worth keeping verbatim. Rank by how
badly a later question would suffer if the span were lost: concrete decisions, names,
versions, identifiers, numeric limits, ownership and commitments matter most; chit-chat,
restatements and generic advice matter least.

You must select exactly {budget} span numbers. Selecting none is not an answer.

Reply with exactly two lines and nothing else. Do not repeat the spans.

KEEP: <{budget} comma-separated span numbers between 0 and {last}>
SUMMARY: <one or two sentences describing what the dropped spans were about>

Now answer for the {n_spans} spans above.
"""


def build_compaction_prompt(span_texts, budget):
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(span_texts))
    return COMPACTION_INSTRUCTION.format(budget=budget, spans=numbered,
                                         n_spans=len(span_texts),
                                         last=len(span_texts) - 1)
