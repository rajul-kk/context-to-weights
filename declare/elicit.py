import re

import torch

from declare.regions import render
from sleep.lm import batch_generate, chat_text

DECLARE_SYSTEM = (
    "You route your own attention. Before answering, you declare which single region of the "
    "conversation you must read. You reply in the requested format and nothing else."
)

DECLARE_TEMPLATE = """{context}

Question: {question}

Declare the one region you must read to answer this question. Reply with exactly one line:

FOCUS: <region number between 0 and {last}>"""

PROBE_SYSTEM = (
    "You judge whether one region of a conversation contains the information needed to "
    "answer a question."
)

PROBE_TEMPLATE = """[REGION {idx}]
{body}

Question: {question}

Does this region contain the information needed to answer the question? Answer Yes or No."""

_FOCUS = re.compile(r"FOCUS\s*:\s*(\d+)", re.IGNORECASE)
_ANYNUM = re.compile(r"\d+")


def parse_declaration(raw, n_regions):
    m = _FOCUS.search(raw)
    if m:
        v = int(m.group(1))
        return v if 0 <= v < n_regions else None
    nums = [int(x) for x in _ANYNUM.findall(raw)]
    for v in nums:
        if 0 <= v < n_regions:
            return v
    return None


class GenerateElicitor:
    name = "generate"

    def __init__(self, model, tokenizer, max_new_tokens=24):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.calls = 0
        self.unparsed = 0

    def declare(self, regions, question):
        self.calls += 1
        context, _ = render(regions, self.tokenizer)
        prompt = DECLARE_TEMPLATE.format(context=context, question=question,
                                         last=len(regions) - 1)
        raw = batch_generate(self.model, self.tokenizer, [(DECLARE_SYSTEM, prompt)],
                             max_new_tokens=self.max_new_tokens, batch_size=1)[0]
        choice = parse_declaration(raw, len(regions))
        if choice is None:
            self.unparsed += 1
        return choice, raw


class ReadElicitor:
    name = "read"

    def __init__(self, model, tokenizer, max_length=2048, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.calls = 0
        self.unparsed = 0
        self.yes_ids = self._variants(["Yes", " Yes", "yes", " yes"])
        self.no_ids = self._variants(["No", " No", "no", " no"])

    def _variants(self, words):
        ids = set()
        for w in words:
            enc = self.tokenizer.encode(w, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    @torch.no_grad()
    def scores(self, regions, question):
        device = next(self.model.parameters()).device
        prompts = []
        for i, lines in enumerate(regions):
            body = "\n".join(lines) if lines else "(empty)"
            prompts.append(chat_text(self.tokenizer, PROBE_SYSTEM,
                                     PROBE_TEMPLATE.format(idx=i, body=body,
                                                           question=question)))
        out = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i: i + self.batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                                 max_length=self.max_length).to(device)
            logits = self.model(**enc).logits[:, -1].float()
            lp = torch.log_softmax(logits, dim=-1)
            yes = torch.logsumexp(lp[:, self.yes_ids], dim=-1)
            no = torch.logsumexp(lp[:, self.no_ids], dim=-1)
            out.extend((yes - no).tolist())
        return out

    def declare(self, regions, question):
        self.calls += 1
        s = self.scores(regions, question)
        best = max(range(len(s)), key=lambda i: s[i])
        return best, ", ".join(f"{v:.2f}" for v in s)


def build_elicitor(kind, model, tokenizer, cfg):
    if kind == "generate":
        return GenerateElicitor(model, tokenizer,
                                max_new_tokens=cfg["declare"]["max_new_tokens"])
    if kind == "read":
        return ReadElicitor(model, tokenizer, max_length=cfg["model"]["max_length"],
                            batch_size=cfg["declare"]["probe_batch_size"])
    raise ValueError(kind)
