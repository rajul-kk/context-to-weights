import re

import torch

from declare.regions import render, render_spans
from sleep.lm import batch_generate, chat_text

ATTEND_SYSTEM = "You answer questions about a conversation."

ATTEND_TEMPLATE = """{context}

Question: {question}

Answer:"""

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


LAST_ROW = []


def probe_attention(module, query, key, value, attention_mask=None, scaling=None,
                    dropout=0.0, **kwargs):
    from transformers.models.llama.modeling_llama import repeat_kv

    groups = query.shape[1] // key.shape[1]
    k = repeat_kv(key, groups)
    v = repeat_kv(value, groups)
    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    mask = attention_mask
    if mask is not None:
        mask = mask[:, :, :, : k.shape[-2]]
    out = torch.nn.functional.scaled_dot_product_attention(
        query, k, v, attn_mask=mask, dropout_p=0.0,
        is_causal=mask is None and query.shape[-2] > 1)
    out = out.transpose(1, 2).contiguous()

    if query.shape[-2] > 1:
        logits = (query[:, :, -1:, :] @ k.transpose(-1, -2)) * scaling
        if mask is not None:
            logits = logits + mask[:, :, -1:, :]
        LAST_ROW.append(logits.softmax(dim=-1).detach().float()[:, :, 0, :].cpu())
    return out, None


def register_probe():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["myrios_probe"] = probe_attention
    return "myrios_probe"


class AttentionElicitor:
    name = "attention"

    def __init__(self, model, tokenizer, reduce="sum", layer_frac=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.reduce = reduce
        self.layer_frac = layer_frac
        self.calls = 0
        self.unparsed = 0
        if getattr(model.config, "_attn_implementation", None) != "myrios_probe":
            raise RuntimeError(
                "attention probing needs the model loaded with the myrios_probe attention "
                "implementation; declare/run.py does this when an attention mode is requested.")

    def _token_spans(self, offsets, char_spans, shift):
        out = []
        for lo, hi in char_spans:
            idx = [i for i, (a, b) in enumerate(offsets)
                   if b > lo + shift and a < hi + shift and b > a]
            out.append(idx)
        return out

    @torch.no_grad()
    def scores(self, regions, question):
        device = next(self.model.parameters()).device
        context, char_spans = render_spans(regions)
        body = ATTEND_TEMPLATE.format(context=context, question=question)
        prompt = chat_text(self.tokenizer, ATTEND_SYSTEM, body)
        shift = prompt.index(context) if context in prompt else 0

        enc = self.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True,
                             add_special_tokens=False)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        spans = self._token_spans(offsets, char_spans, shift)

        LAST_ROW.clear()
        self.model(**enc, use_cache=False)
        captured = list(LAST_ROW)
        LAST_ROW.clear()
        if not captured:
            raise RuntimeError("no attention rows captured by the probe attention function")

        keep = int(len(captured) * self.layer_frac)
        rows = captured[keep:] if keep < len(captured) else captured
        attn = torch.stack([r.mean(dim=1)[0] for r in rows]).mean(dim=0)

        out = []
        for idx in spans:
            if not idx:
                out.append(float("-inf"))
                continue
            mass = attn[idx].sum().item()
            out.append(mass / len(idx) if self.reduce == "mean" else mass)
        return out

    def declare(self, regions, question):
        self.calls += 1
        s = self.scores(regions, question)
        best = max(range(len(s)), key=lambda i: s[i])
        return best, ", ".join(f"{v:.4f}" for v in s)


def build_elicitor(kind, model, tokenizer, cfg):
    dc = cfg["declare"]
    if kind == "generate":
        return GenerateElicitor(model, tokenizer, max_new_tokens=dc["max_new_tokens"])
    if kind == "read":
        return ReadElicitor(model, tokenizer, max_length=cfg["model"]["max_length"],
                            batch_size=dc["probe_batch_size"])
    if kind == "attention":
        return AttentionElicitor(model, tokenizer, reduce=dc.get("attention_reduce", "sum"),
                                 layer_frac=dc.get("attention_layer_frac", 0.0))
    if kind == "attention_mean":
        return AttentionElicitor(model, tokenizer, reduce="mean",
                                 layer_frac=dc.get("attention_layer_frac", 0.0))
    if kind == "attention_late":
        return AttentionElicitor(model, tokenizer,
                                 reduce=dc.get("attention_reduce", "sum"), layer_frac=0.5)
    raise ValueError(kind)
