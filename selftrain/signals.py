import math
import re

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    positives = sum(1 for _, y in pairs if y)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average = (i + j) / 2 + 1
        rank_sum += average * sum(1 for k in range(i, j + 1) if pairs[k][1])
        i = j + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def ece(confidences, labels, bins=10):
    total = len(confidences)
    if total == 0:
        return None
    buckets = [[] for _ in range(bins)]
    for c, y in zip(confidences, labels):
        buckets[min(int(c * bins), bins - 1)].append((c, y))
    gap = 0.0
    for bucket in buckets:
        if bucket:
            mean_conf = sum(c for c, _ in bucket) / len(bucket)
            accuracy = sum(1 for _, y in bucket if y) / len(bucket)
            gap += len(bucket) / total * abs(mean_conf - accuracy)
    return gap


def mean_logprob(values, empty=-50.0):
    return sum(values) / len(values) if values else empty


def yes_probability(logit_yes, logit_no):
    return 1.0 / (1.0 + math.exp(logit_no - logit_yes))


def parse_confidence(text):
    match = _NUMBER.search(text or "")
    if not match:
        return None
    value = float(match.group())
    if "." in match.group() and value <= 1.0:
        return value
    return min(value, 100.0) / 100.0


def verdict(per_seed, margin=0.05, floor=0.6, min_seeds=3):
    if len(per_seed) < min_seeds:
        return {"passed": False, "reason": f"needs {min_seeds} seeds, have {len(per_seed)}"}
    measured = [s["pyes"] for s in per_seed]
    stated = [s["verbal"] for s in per_seed]
    mean_measured = sum(measured) / len(measured)
    gap = mean_measured - sum(stated) / len(stated)
    checks = {
        "measured_above_floor": mean_measured > floor,
        "gap_at_least_margin": gap >= margin,
        "seed_ranges_disjoint": min(measured) > max(stated),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {"passed": not failed, "mean_measured": mean_measured, "gap": gap, "checks": checks,
            "reason": "all checks hold" if not failed else "failed: " + ", ".join(failed)}
