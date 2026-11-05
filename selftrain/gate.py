import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.io import ensure_dir, load_config, parse_overrides, set_seed, write_json, write_jsonl
from selftrain.signals import (auroc, ece, mean_logprob, parse_confidence, within_family_auroc,
                               yes_probability)
from selftrain.tasks import FAMILIES, check, make_tasks
from sleep.lm import batch_generate, chat_text, load_backbone

JUDGE = ("Question: {question}\n\nProposed solution:\n{solution}\n\n"
         "Is the final answer correct? Reply with only yes or no.")
CONFIDENCE = ("Question: {question}\n\nProposed solution:\n{solution}\n\n"
              "State your confidence that the final answer is correct as a number from 0 to 100. "
              "Reply with only the number.")


def eos_ids(model, tokenizer):
    ids = model.generation_config.eos_token_id
    ids = ids if isinstance(ids, list) else [ids]
    return {i for i in ids + [tokenizer.eos_token_id] if i is not None}


@torch.no_grad()
def sample(model, tokenizer, tasks, temperature, max_new_tokens, batch_size):
    device = next(model.parameters()).device
    stops = eos_ids(model, tokenizer)
    out = []
    for i in range(0, len(tasks), batch_size):
        chunk = tasks[i: i + batch_size]
        texts = [chat_text(tokenizer, None, t.prompt) for t in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=1.0, top_k=0, repetition_penalty=1.0,
                             pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True,
                             output_logits=True)
        start = enc["input_ids"].shape[1]
        logps = torch.stack([torch.log_softmax(s.float(), dim=-1) for s in gen.logits], dim=1)
        for j in range(len(chunk)):
            ids = gen.sequences[j][start:].tolist()
            cut = next((k for k, tok in enumerate(ids) if tok in stops), len(ids))
            chosen = [logps[j, k, ids[k]].item() for k in range(cut)]
            out.append({"text": tokenizer.decode(ids[:cut], skip_special_tokens=True).strip(),
                        "logprob": mean_logprob(chosen)})
    return out


def first_token(tokenizer, word):
    return tokenizer(word, add_special_tokens=False)["input_ids"][0]


@torch.no_grad()
def yes_scores(model, tokenizer, prompts, batch_size):
    device = next(model.parameters()).device
    yes = [first_token(tokenizer, w) for w in ("Yes", "yes")]
    no = [first_token(tokenizer, w) for w in ("No", "no")]
    out = []
    for i in range(0, len(prompts), batch_size):
        texts = [chat_text(tokenizer, None, p) for p in prompts[i: i + batch_size]]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        logits = model(**enc).logits[:, -1].float()
        for row in logits:
            out.append(yes_probability(torch.logsumexp(row[yes], 0).item(),
                                       torch.logsumexp(row[no], 0).item()))
    return out


def verbal_scores(model, tokenizer, prompts, batch_size):
    replies = batch_generate(model, tokenizer, [(None, p) for p in prompts], max_new_tokens=6,
                             batch_size=batch_size)
    parsed = [parse_confidence(r) for r in replies]
    return [0.5 if p is None else p for p in parsed], sum(p is None for p in parsed) / len(parsed)


def run_seed(model, tokenizer, cfg, seed, n):
    g = cfg["selftrain"]
    set_seed(seed)
    tasks = make_tasks(n, seed, g.get("families") or FAMILIES)
    samples = sample(model, tokenizer, tasks, g["temperature"], g["max_new_tokens"], g["batch_size"])
    labels = [check(t, s["text"]) for t, s in zip(tasks, samples)]
    judge = [JUDGE.format(question=t.question, solution=s["text"]) for t, s in zip(tasks, samples)]
    ask = [CONFIDENCE.format(question=t.question, solution=s["text"]) for t, s in zip(tasks, samples)]
    pyes = yes_scores(model, tokenizer, judge, g["batch_size"])
    verbal, unparsed = verbal_scores(model, tokenizer, ask, g["batch_size"])
    logprob = [s["logprob"] for s in samples]
    by_family = defaultdict(list)
    for t, y in zip(tasks, labels):
        by_family[t.family].append(y)
    families = [t.family for t in tasks]
    scores = {"pyes": pyes, "logprob": logprob, "verbal": verbal}
    within = {k: within_family_auroc(v, labels, families) for k, v in scores.items()}
    summary = {
        "model": cfg["model"]["base"], "seed": seed, "n": n,
        "pass_rate": sum(labels) / n,
        "pass_by_family": {f: sum(v) / len(v) for f, v in by_family.items()},
        "auroc": {k: w[0] for k, w in within.items()},
        "auroc_by_family": {k: w[1] for k, w in within.items()},
        "auroc_pooled": {k: auroc(v, labels) for k, v in scores.items()},
        "ece": {"pyes": ece(pyes, labels), "verbal": ece(verbal, labels)},
        "unparsed_verbal_rate": unparsed,
    }
    rows = [{"family": t.family, "question": t.question, "answer": t.answer, "text": s["text"],
             "correct": y, "pyes": a, "verbal": b, "logprob": s["logprob"]}
            for t, s, y, a, b in zip(tasks, samples, labels, pyes, verbal)]
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/selftrain.yaml")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    n = args.n or cfg["selftrain"]["n_tasks"]
    root = Path(args.out or cfg["run_root"])
    model, tokenizer = load_backbone(cfg)
    tag = cfg["model"]["base"].split("/")[-1]
    for seed in args.seeds:
        summary, rows = run_seed(model, tokenizer, cfg, seed, n)
        folder = ensure_dir(root / f"{tag}_n{n}_s{seed}")
        write_json(folder / "g1.summary.json", summary)
        write_jsonl(folder / "samples.jsonl", rows)
        print(f"{tag} seed {seed}: pass {summary['pass_rate']:.3f} auroc {summary['auroc']}",
              flush=True)


if __name__ == "__main__":
    main()
