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

    def __init__(self):
        self.last_decided_by = self.name

    def select(self, spans, budget):
        raise NotImplementedError

    def compact(self, spans, keep_frac):
        budget = max(1, math.ceil(len(spans) * keep_frac))
        kept_idx, summary = self.select(spans, budget)
        kept_idx = sorted(set(i for i in kept_idx if 0 <= i < len(spans)))[:budget]
        return kept_idx, summary


class HeuristicCompactor(Compactor):
    name = "heuristic"

    def __init__(self):
        super().__init__()

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

    def __init__(self, generator, fallback=None, max_new_tokens=192, strict=False):
        super().__init__()
        self.generator = generator
        self.fallback = fallback or HeuristicCompactor()
        self.max_new_tokens = max_new_tokens
        self.strict = strict
        self.calls = 0
        self.fallbacks = 0
        self.empty_keeps = 0
        self.last_raw = None
        self.failed_replies = []

    @property
    def fallback_rate(self):
        return self.fallbacks / self.calls if self.calls else 0.0

    def _fall_back(self, spans, budget, raw):
        self.fallbacks += 1
        self.last_decided_by = "heuristic-fallback"
        if raw is not None and len(self.failed_replies) < 20:
            self.failed_replies.append(raw)
        if self.strict:
            return [], _extractive_summary([s["text"] for s in spans])
        return self.fallback.select(spans, budget)

    def select(self, spans, budget):
        self.calls += 1
        self.last_decided_by = "model"
        prompt = build_compaction_prompt([s["text"] for s in spans], budget)
        try:
            raw = self.generator(COMPACTION_SYSTEM, prompt, self.max_new_tokens)
        except Exception:
            return self._fall_back(spans, budget, None)
        self.last_raw = raw
        kept, summary = parse_compaction_reply(raw, len(spans))
        if kept is None:
            return self._fall_back(spans, budget, raw)
        if not kept:
            self.empty_keeps += 1
        if not summary:
            summary = _extractive_summary([spans[i]["text"] for i in range(len(spans)) if i not in set(kept)])
        return kept, summary


_NUMBER_LINE = re.compile(r"^[\s\[\]\d,]+$")


def _parse_keep_payload(payload, n_spans):
    if payload.upper().startswith("NONE"):
        return []
    return [int(m) for m in re.findall(r"\d+", payload) if int(m) < n_spans]


def parse_compaction_reply(raw, n_spans):
    kept = None
    summary = ""
    for line in raw.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("KEEP:"):
            kept = _parse_keep_payload(line.split(":", 1)[1].strip(), n_spans)
        elif upper.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()

    if kept is not None:
        return kept, summary

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return None, ""
    head = lines[0]
    if head.upper().startswith("NONE") or (_NUMBER_LINE.match(head) and re.search(r"\d", head)):
        return _parse_keep_payload(head, n_spans), " ".join(lines[1:]).strip()
    return None, ""


def _extractive_summary(dropped, max_items=3):
    if not dropped:
        return "No further detail was dropped."
    picks = dropped[:max_items]
    body = " ".join(p.rstrip(".") + "." for p in picks)
    extra = len(dropped) - len(picks)
    if extra > 0:
        body += f" ({extra} further minor exchanges omitted.)"
    return body
