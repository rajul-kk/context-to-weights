QA_SYSTEM = (
    "You answer questions about an earlier engineering conversation. "
    "Reply with the answer only, no explanation."
)


def build_prompt(context, question):
    body = "\n".join(context).strip()
    if body:
        return f"Retained notes from the conversation:\n{body}\n\nQuestion: {question}"
    return f"Question: {question}"
