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
