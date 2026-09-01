import re
import statistics

_PUNCT = re.compile(r"[^\w\s./+-]")
_WS = re.compile(r"\s+")


def normalize(text):
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


def answer_match(prediction, gold, aliases=None):
    p = normalize(prediction)
    candidates = [gold] + list(aliases or [])
    for c in candidates:
        g = normalize(c)
        if g and g in p:
            return True
    return False


def median_mean(values):
    if not values:
        return {"median": float("nan"), "mean": float("nan"), "n": 0}
    return {
        "median": float(statistics.median(values)),
        "mean": float(sum(values) / len(values)),
        "n": len(values),
    }


def aggregate(records):
    ce_all = []
    per_probe_median = []
    correct = 0
    prompt_tokens = []
    for r in records:
        correct += int(r["correct"])
        prompt_tokens.append(r["prompt_tokens"])
        if r.get("ce"):
            ce_all.extend(r["ce"])
            per_probe_median.append(statistics.median(r["ce"]))
    out = {
        "n": len(records),
        "retention_accuracy": correct / len(records) if records else 0.0,
        "mean_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0.0,
    }
    tok = median_mean(ce_all)
    out["token_ce_median"] = tok["median"]
    out["token_ce_mean"] = tok["mean"]
    probe = median_mean(per_probe_median)
    out["probe_ce_median"] = probe["median"]
    out["probe_ce_mean"] = probe["mean"]

    evicted = [r for r in records if not r.get("fact_in_context", False)]
    out["n_evicted"] = len(evicted)
    out["evicted_accuracy"] = (
        sum(int(r["correct"]) for r in evicted) / len(evicted) if evicted else float("nan")
    )
    evicted_ce = [c for r in evicted for c in (r.get("ce") or [])]
    ev = median_mean(evicted_ce)
    out["evicted_ce_median"] = ev["median"]
    out["evicted_ce_mean"] = ev["mean"]
    return out
