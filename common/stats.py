import math
import statistics

T_CRIT = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
    17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def verdict_for(sigma, nan="undetermined", tie="indistinguishable"):
    if sigma != sigma:
        return nan
    if sigma >= 2.0:
        return "clears control"
    if sigma <= -2.0:
        return "below control"
    return tie


def critical(n_runs):
    if n_runs <= 1:
        return 2.0
    return T_CRIT.get(n_runs - 1, 2.0)


def paired_t(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    if len(diffs) < 2:
        raise ValueError("a paired t needs at least two pairs")
    mean = sum(diffs) / len(diffs)
    sd = statistics.stdev(diffs)
    stat = mean / (sd / len(diffs) ** 0.5) if sd else float("nan")
    return mean, stat, len(diffs) - 1


def mcnemar_exact(only_a, only_b):
    if only_a < 0 or only_b < 0:
        raise ValueError("discordant counts must be non-negative")
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def t_two_sided_p(stat, dof):
    if dof < 1 or int(dof) != dof:
        raise ValueError("dof must be a positive integer")
    if stat != stat:
        return float("nan")
    theta = math.atan(abs(stat) / math.sqrt(dof))
    s, c2 = math.sin(theta), math.cos(theta) ** 2
    if dof % 2:
        term, total = 1.0, 1.0 if dof > 1 else 0.0
        for k in range(3, dof - 1, 2):
            term *= (k - 1) / k * c2
            total += term
        area = 2 / math.pi * (theta + s * math.cos(theta) * total)
    else:
        term, total = 1.0, 1.0
        for k in range(2, dof - 1, 2):
            term *= (k - 1) / k * c2
            total += term
        area = s * total
    return min(1.0, max(0.0, 1.0 - area))


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted
