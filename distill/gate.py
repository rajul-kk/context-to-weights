import random

POLICIES = ["kl_top", "uniform", "random"]


def _selected_count(n, top_frac):
    return max(1, round(n * top_frac))


def token_weights(row, policy, top_frac, granularity, floor_weight, rng):
    n = len(row["kl"])
    if policy == "uniform":
        return [1.0] * n

    if granularity == "token":
        units = [{"token_idx": [i], "score": row["kl"][i]} for i in range(n)]
    else:
        units = [{"token_idx": s["token_idx"], "score": s["kl_mean"]} for s in row["spans"]]
    if not units:
        return [1.0] * n

    k = _selected_count(len(units), top_frac)
    ranked = sorted(range(len(units)), key=lambda i: -units[i]["score"])[:k]
    if policy == "kl_top":
        chosen = ranked
    elif policy == "random":
        budget = sum(len(units[i]["token_idx"]) for i in ranked)
        order = list(range(len(units)))
        rng.shuffle(order)
        chosen = []
        taken = 0
        for i in order:
            size = len(units[i]["token_idx"])
            if taken >= budget:
                break
            if chosen and abs(taken + size - budget) > abs(taken - budget):
                continue
            chosen.append(i)
            taken += size
    else:
        raise ValueError(policy)

    weights = [floor_weight] * n
    for u in chosen:
        for i in units[u]["token_idx"]:
            if 0 <= i < n:
                weights[i] = 1.0
    return weights


def apply_gate(rows, cfg, seed=0):
    gc = cfg["gate"]
    rng = random.Random(seed)
    out = []
    for row in rows:
        w = token_weights(row, gc["policy"], gc["top_frac"], gc["granularity"],
                          gc["floor_weight"], rng)
        out.append(dict(row, weights=w))
    return out


def gate_stats(rows):
    total = 0
    active = 0
    for r in rows:
        total += len(r["weights"])
        active += sum(1 for w in r["weights"] if w > 0)
    return {"tokens": total, "active_tokens": active,
            "active_frac": active / total if total else 0.0}
