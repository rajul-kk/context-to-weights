import math
import re

from compactor.prompts import COMPACTION_SYSTEM, build_compaction_prompt

CUES = [
    r"\bsettled on\b", r"\bfixed at\b", r"\bagreed\b", r"\bpins?\b", r"\bowns?\b",
    r"\bmust\b", r"\bnever\b", r"\bonly\b", r"\bdecided\b", r"\block in\b",
    r"\brejected\b", r"\bsign off\b", r"\bkill switch\b", r"\bnowhere else\b",
]
_CUE_RE = re.compile("|".join(CUES), re.IGNORECASE)
_NUMERIC = re.compile(r"\d")
_IDENT = re.compile(r"[A-Z][A-Za-z0-9]*[-_.][A-Za-z0-9_.-]+|[A-Z]{2,}[A-Z_]*|\b\w+\.\w+\.\w+\b")
_VERSION = re.compile(r"\b\d+\.\d+(\.\d+)?\b")


class Compactor:
    name = "base"

    def select(self, spans, budget):
        raise NotImplementedError

    def compact(self, spans, keep_frac):
        budget = max(1, math.ceil(len(spans) * keep_frac))
        kept_idx, summary = self.select(spans, budget)
        kept_idx = sorted(set(i for i in kept_idx if 0 <= i < len(spans)))[:budget]
        return kept_idx, summary


class HeuristicCompactor(Compactor):
    name = "heuristic"

    def score(self, text):
        s = 0.0
        if _CUE_RE.search(text):
            s += 2.0
        if _VERSION.search(text):
            s += 1.5
        if _NUMERIC.search(text):
            s += 1.0
        s += 0.6 * min(3, len(_IDENT.findall(text)))
        s += min(0.5, len(text) / 600.0)
        return s

    def select(self, spans, budget):
        scored = sorted(range(len(spans)), key=lambda i: (-self.score(spans[i]["text"]), i))
        kept = scored[:budget]
        dropped = [spans[i]["text"] for i in scored[budget:]]
        summary = _extractive_summary(dropped)
        return kept, summary


class ModelCompactor(Compactor):
    name = "model"

    def __init__(self, generator, fallback=None, max_new_tokens=192):
        self.generator = generator
        self.fallback = fallback or HeuristicCompactor()
        self.max_new_tokens = max_new_tokens

    def select(self, spans, budget):
        prompt = build_compaction_prompt([s["text"] for s in spans], budget)
        try:
            raw = self.generator(COMPACTION_SYSTEM, prompt, self.max_new_tokens)
        except Exception:
            return self.fallback.select(spans, budget)
        kept, summary = parse_compaction_reply(raw, len(spans))
        if kept is None:
            return self.fallback.select(spans, budget)
        if not summary:
            summary = _extractive_summary([spans[i]["text"] for i in range(len(spans)) if i not in set(kept)])
        return kept, summary


def parse_compaction_reply(raw, n_spans):
    kept = None
    summary = ""
    for line in raw.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("KEEP:"):
            payload = line.split(":", 1)[1].strip()
            if payload.upper().startswith("NONE"):
                kept = []
            else:
                kept = [int(m) for m in re.findall(r"\d+", payload) if int(m) < n_spans]
        elif upper.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
    return kept, summary


def _extractive_summary(dropped, max_items=3):
    if not dropped:
        return "No further detail was dropped."
    picks = dropped[:max_items]
    body = " ".join(p.rstrip(".") + "." for p in picks)
    extra = len(dropped) - len(picks)
    if extra > 0:
        body += f" ({extra} further minor exchanges omitted.)"
    return body
