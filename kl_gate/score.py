import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import load_config, parse_overrides, set_seed, write_jsonl
from kl_gate.skillset import SKILL_SYSTEM, load_all, with_skill, without_skill
from sleep.lm import chat_text, load_backbone

_SPAN_SPLIT = re.compile(r"(?<=[.!?:\n])")


def span_bounds(text, granularity="span"):
    if granularity == "token":
        return [(0, len(text))]
    bounds = []
    start = 0
    for m in _SPAN_SPLIT.finditer(text):
        end = m.end()
        if end - start < 8:
            continue
        bounds.append((start, end))
        start = end
    if start < len(text):
        bounds.append((start, len(text)))
    return bounds or [(0, len(text))]


def _encode(tokenizer, text):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


@torch.no_grad()
def _logprobs(model, prefix_ids, response_ids):
    device = next(model.parameters()).device
    ids = torch.tensor([prefix_ids + response_ids], device=device)
    logits = model(input_ids=ids).logits[0].float()
    start = len(prefix_ids) - 1
    slice_ = logits[start: start + len(response_ids)]
    return torch.log_softmax(slice_, dim=-1)


def score_demo(model, tokenizer, doc, query, response, max_length):
    teacher_prefix = chat_text(tokenizer, SKILL_SYSTEM, with_skill(doc, query))
    student_prefix = chat_text(tokenizer, SKILL_SYSTEM, without_skill(query))
    t_ids = tokenizer(teacher_prefix, add_special_tokens=False)["input_ids"][-max_length:]
    s_ids = tokenizer(student_prefix, add_special_tokens=False)["input_ids"][-max_length:]
    r_ids, offsets = _encode(tokenizer, response)
    if not r_ids:
        return None

    t_lp = _logprobs(model, t_ids, r_ids)
    s_lp = _logprobs(model, s_ids, r_ids)
    p_t = t_lp.exp()
    kl = (p_t * (t_lp - s_lp)).sum(dim=-1).clamp(min=0.0)

    idx = torch.arange(len(r_ids), device=kl.device)
    gold = torch.tensor(r_ids, device=kl.device)
    teacher_ce = (-t_lp[idx, gold]).tolist()
    student_ce = (-s_lp[idx, gold]).tolist()

    return {
        "response": response,
        "token_ids": r_ids,
        "token_text": [tokenizer.decode([i]) for i in r_ids],
        "offsets": [list(o) for o in offsets],
        "kl": kl.tolist(),
        "teacher_ce": teacher_ce,
        "student_ce": student_ce,
    }


def aggregate_spans(scored, granularity="span"):
    bounds = span_bounds(scored["response"], granularity)
    spans = []
    for a, b in bounds:
        members = [i for i, (s, e) in enumerate(scored["offsets"]) if s >= a and s < b]
        if not members:
            continue
        vals = [scored["kl"][i] for i in members]
        spans.append(
            {
                "start": a,
                "end": b,
                "text": scored["response"][a:b],
                "token_idx": members,
                "kl_mean": sum(vals) / len(vals),
                "kl_max": max(vals),
                "n_tokens": len(members),
            }
        )
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/skill_base.yaml")
    ap.add_argument("--skills-root", default="skills/toy")
    ap.add_argument("--skills", nargs="*", default=None)
    ap.add_argument("--granularity", default="span", choices=["span", "token"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    model, tokenizer = load_backbone(cfg)

    rows = []
    for skill in load_all(args.skills_root, args.skills):
        for i, demo in enumerate(skill["demos"]):
            scored = score_demo(model, tokenizer, skill["doc"], demo["query"],
                                demo["response"], cfg["model"]["max_length"])
            if scored is None:
                continue
            spans = aggregate_spans(scored, args.granularity)
            rows.append(
                {
                    "skill": skill["name"],
                    "category": skill["category"],
                    "demo_idx": i,
                    "query": demo["query"],
                    "kind": demo["kind"],
                    "granularity": args.granularity,
                    **scored,
                    "spans": spans,
                }
            )
        print(f"scored {skill['name']}: {len(skill['demos'])} demos")

    write_jsonl(args.out, rows)
    print(f"{len(rows)} scored demos -> {args.out}")


if __name__ == "__main__":
    main()
