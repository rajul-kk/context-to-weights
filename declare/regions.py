import random

from compactor.segment import count_tokens


def build_regions(turns, n_regions, tokenizer=None):
    lines = [f"{t.role}: {t.content}" for t in turns]
    costs = [count_tokens(x, tokenizer) for x in lines]
    total = sum(costs)
    target = max(1, total // n_regions)

    regions = []
    current = []
    running = 0
    for line, cost in zip(lines, costs):
        current.append(line)
        running += cost
        if running >= target and len(regions) < n_regions - 1:
            regions.append(current)
            current = []
            running = 0
    if current:
        regions.append(current)
    while len(regions) < n_regions:
        regions.append([])
    return regions[:n_regions]


def gold_region(regions, answer, markers=None):
    needles = [answer] + list(markers or [])
    for i, lines in enumerate(regions):
        body = " ".join(lines)
        if any(n and n in body for n in needles):
            return i
    return None


def render(regions, tokenizer=None):
    blocks = []
    for i, lines in enumerate(regions):
        body = "\n".join(lines) if lines else "(empty)"
        blocks.append(f"[REGION {i}]\n{body}")
    text = "\n\n".join(blocks)
    return text, count_tokens(text, tokenizer)


def region_tokens(regions, tokenizer=None):
    return [count_tokens("\n".join(lines) or "(empty)", tokenizer) for lines in regions]


def permute(regions, rng):
    order = list(range(len(regions)))
    rng.shuffle(order)
    return [regions[i] for i in order], order
