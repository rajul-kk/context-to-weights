import re

_SENT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text, min_chars=25):
    parts = [p.strip() for p in _SENT.split(text.strip()) if p.strip()]
    if not parts:
        return []
    merged = []
    for p in parts:
        if merged and len(p) < min_chars:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def segment_turns(turns, granularity="sentence"):
    out = []
    for t in turns:
        if granularity == "turn":
            pieces = [t.content.strip()]
        else:
            pieces = split_sentences(t.content)
        for j, piece in enumerate(pieces):
            out.append(
                {
                    "span_id": f"t{t.idx}s{j}",
                    "turn_idx": t.idx,
                    "role": t.role,
                    "text": piece,
                    "tags": span_tags(t.tags, piece),
                }
            )
    return out


def span_tags(turn_tags, piece):
    if len(turn_tags) >= 3 and turn_tags[0] == "fact":
        markers = [m for m in turn_tags[2:] if m]
        if any(m in piece or piece in m for m in markers):
            return [turn_tags[0], turn_tags[1]]
        return ["filler"]
    return list(turn_tags)


def count_tokens(text, tokenizer=None):
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(1, int(len(text.split()) * 1.35))
