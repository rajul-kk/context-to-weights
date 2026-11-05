import math
import random
import statistics

T_CRIT = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
    17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

N_RESAMPLES = 10000
BOOTSTRAP_SEED = 0
CI_LEVEL = 0.95


def ci_description(n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED, level=CI_LEVEL):
    return f"percentile bootstrap {level * 100:g}%, {n_resamples} resamples, seed {seed}"


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


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


def binomial_se(rate, n):
    if not n or rate is None or rate != rate:
        return 0.0
    return (rate * (1 - rate) / n) ** 0.5


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvalues[i])
        running = max(running, val)
        adjusted[i] = running
    return adjusted


def _percentile(sorted_values, q):
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci(values, n_resamples=N_RESAMPLES, level=CI_LEVEL, seed=BOOTSTRAP_SEED,
                 stat=statistics.fmean):
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    tail = (1 - level) / 2
    return _percentile(draws, tail), _percentile(draws, 1 - tail)


def paired_bootstrap_ci(a, b, n_resamples=N_RESAMPLES, level=CI_LEVEL, seed=BOOTSTRAP_SEED):
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    return bootstrap_ci([x - y for x, y in zip(a, b)], n_resamples, level, seed)


def aggregate_seeds(values, n_resamples=N_RESAMPLES, level=CI_LEVEL, seed=BOOTSTRAP_SEED):
    values = [v for v in values if v is not None and v == v]
    if not values:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0,
                "ci_low": float("nan"), "ci_high": float("nan"),
                "ci": ci_description(n_resamples, seed, level)}
    mean = sum(values) / len(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    lo, hi = bootstrap_ci(values, n_resamples, level, seed)
    return {"mean": mean, "sd": sd, "n": len(values), "ci_low": lo, "ci_high": hi,
            "ci": ci_description(n_resamples, seed, level)}


def compare_arms(a_by_seed, b_by_seed, n_resamples=N_RESAMPLES, level=CI_LEVEL,
                 seed=BOOTSTRAP_SEED):
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    a = [a_by_seed[s] for s in seeds]
    b = [b_by_seed[s] for s in seeds]
    out = {"seeds": seeds, "n": len(seeds)}
    if len(seeds) < 2:
        out.update({"mean_diff": float("nan"), "t": float("nan"), "dof": 0,
                    "critical": float("nan"), "verdict": "undetermined",
                    "ci_low": float("nan"), "ci_high": float("nan")})
        return out
    diff, t, dof = paired_t(a, b)
    crit = critical(len(seeds))
    if t != t:
        verdict = "undetermined"
    elif abs(t) < crit:
        verdict = "indistinguishable"
    else:
        verdict = "a ahead" if diff > 0 else "b ahead"
    lo, hi = paired_bootstrap_ci(a, b, n_resamples, level, seed)
    out.update({"mean_diff": diff, "t": t, "dof": dof, "critical": crit, "verdict": verdict,
                "ci_low": lo, "ci_high": hi, "ci": ci_description(n_resamples, seed, level)})
    return out
