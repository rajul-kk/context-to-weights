import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import (append_jsonl, ensure_dir, load_config, parse_overrides, read_jsonl,
                       set_seed, write_json)
from distill.gate import POLICIES, apply_gate, gate_stats
from kl_gate.skillset import SKILL_SYSTEM, load_all, load_index, with_skill, without_skill
from sleep.lm import chat_text, load_backbone
from sleep.trainer import attach_lora


class PairDataset(Dataset):
    def __init__(self, rows, docs, tokenizer, max_length):
        self.rows = []
        for r in rows:
            doc = docs[r["skill"]]
            t_prefix = chat_text(tokenizer, SKILL_SYSTEM, with_skill(doc, r["query"]))
            s_prefix = chat_text(tokenizer, SKILL_SYSTEM, without_skill(r["query"]))
            t_ids = tokenizer(t_prefix, add_special_tokens=False)["input_ids"][-max_length:]
            s_ids = tokenizer(s_prefix, add_special_tokens=False)["input_ids"][-max_length:]
            self.rows.append(
                {
                    "teacher_ids": t_ids,
                    "student_ids": s_ids,
                    "response_ids": r["token_ids"],
                    "weights": r["weights"],
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _pad_side(seqs, pad_id):
    width = max(len(s) for s in seqs)
    ids = [s + [pad_id] * (width - len(s)) for s in seqs]
    mask = [[1] * len(s) + [0] * (width - len(s)) for s in seqs]
    return torch.tensor(ids), torch.tensor(mask)


def collate(batch, pad_id):
    t_full = [b["teacher_ids"] + b["response_ids"] for b in batch]
    s_full = [b["student_ids"] + b["response_ids"] for b in batch]
    t_ids, t_mask = _pad_side(t_full, pad_id)
    s_ids, s_mask = _pad_side(s_full, pad_id)
    n_resp = max(len(b["response_ids"]) for b in batch)
    weights = torch.zeros(len(batch), n_resp)
    for i, b in enumerate(batch):
        w = b["weights"][: len(b["response_ids"])]
        weights[i, : len(w)] = torch.tensor(w, dtype=torch.float)
    return {
        "teacher_ids": t_ids,
        "teacher_mask": t_mask,
        "student_ids": s_ids,
        "student_mask": s_mask,
        "teacher_start": torch.tensor([len(b["teacher_ids"]) - 1 for b in batch]),
        "student_start": torch.tensor([len(b["student_ids"]) - 1 for b in batch]),
        "resp_len": torch.tensor([len(b["response_ids"]) for b in batch]),
        "weights": weights,
    }


def _gather_slice(logits, starts, n_resp):
    rows = []
    for i, start in enumerate(starts.tolist()):
        chunk = logits[i, start: start + n_resp]
        if chunk.shape[0] < n_resp:
            chunk = F.pad(chunk, (0, 0, 0, n_resp - chunk.shape[0]))
        rows.append(chunk)
    return torch.stack(rows)


def weighted_kl(teacher_logits, student_logits, weights, temperature, valid):
    t_lp = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_lp = F.log_softmax(student_logits / temperature, dim=-1)
    kl = (t_lp.exp() * (t_lp - s_lp)).sum(dim=-1)
    kl = torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0) * valid
    denom = (weights * valid).sum().clamp(min=1e-6)
    return (kl * weights * valid).sum() / denom, kl


def run_group(model, tokenizer, cfg, rows, docs, run_dir, label, log_every=25):
    ds = PairDataset(rows, docs, tokenizer, cfg["model"]["max_length"])
    dc = cfg["distill"]
    loader = DataLoader(ds, batch_size=dc["batch_size"], shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(dc["lr"]))
    device = next(model.parameters()).device
    accum = max(1, dc["grad_accum"])
    temperature = float(dc["temperature"])

    model.train()
    history = []
    step = 0
    t0 = time.time()

    while step < dc["steps"]:
        for batch in loader:
            if step >= dc["steps"]:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            n_resp = int(batch["resp_len"].max().item())

            with torch.no_grad():
                with model.disable_adapter():
                    t_out = model(input_ids=batch["teacher_ids"],
                                  attention_mask=batch["teacher_mask"]).logits.float()
                teacher = _gather_slice(t_out, batch["teacher_start"], n_resp)

            s_out = model(input_ids=batch["student_ids"],
                          attention_mask=batch["student_mask"]).logits.float()
            student = _gather_slice(s_out, batch["student_start"], n_resp)

            positions = torch.arange(n_resp, device=device).unsqueeze(0)
            valid = (positions < batch["resp_len"].unsqueeze(1)).float()
            loss, kl = weighted_kl(teacher, student, batch["weights"], temperature, valid)
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % log_every == 0 or step == dc["steps"]:
                kl_d = kl.detach()
                gated = float((kl_d * batch["weights"] * valid).sum()
                              / (batch["weights"] * valid).sum().clamp(min=1e-6))
                all_mean = float((kl_d * valid).sum() / valid.sum().clamp(min=1e-6))
                entry = {"step": step, "loss": float(loss.item()),
                         "kl_all_mean": all_mean, "kl_gated_mean": gated}
                history.append(entry)
                append_jsonl(Path(run_dir) / "metrics.jsonl", dict(entry, group=label))

    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return {"group": label, "steps": step, "n_examples": len(ds),
            "gpu_seconds": time.time() - t0, "history": history}


def group_rows(rows, grouping, index):
    cat = {r["name"]: r["category"] for r in index}
    groups = {}
    for r in rows:
        key = cat.get(r["skill"], "all") if grouping == "category" else r["skill"]
        groups.setdefault(key, []).append(r)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/skill_base.yaml")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--policy", default=None, choices=POLICIES)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    if args.policy:
        cfg["gate"]["policy"] = args.policy
    set_seed(cfg["seed"])

    policy = cfg["gate"]["policy"]
    tag = f"{policy}_{cfg['gate']['granularity']}_{cfg['gate']['top_frac']}"
    run_dir = ensure_dir(args.run_dir or Path(cfg["run_root"]) / f"distill_{tag}")

    rows = read_jsonl(args.scores)
    rows = apply_gate(rows, cfg, seed=cfg["seed"])
    stats = gate_stats(rows)
    write_json(run_dir / "config.json", {"cfg": cfg, "gate_stats": stats})
    print(f"gate {policy}: {stats['active_frac']:.3f} of {stats['tokens']} tokens active")

    root = cfg["skills"]["root"]
    index = load_index(root)
    docs = {s["name"]: s["doc"] for s in load_all(root)}
    groups = group_rows(rows, cfg["skills"]["grouping"], index)
    if args.groups:
        groups = {k: v for k, v in groups.items() if k in args.groups}

    model, tokenizer = load_backbone(cfg)
    results = []
    for label, group in sorted(groups.items()):
        adapter_dir = run_dir / label / "adapter"
        if args.resume and adapter_dir.exists():
            print(f"skip {label} (already trained)")
            continue
        peft_model = attach_lora(model, {"sleep": cfg["distill"]})
        result = run_group(peft_model, tokenizer, cfg, group, docs, run_dir, label)
        ensure_dir(adapter_dir.parent)
        peft_model.save_pretrained(str(adapter_dir))
        write_json(run_dir / label / "result.json", result)
        results.append(result)
        print(f"{label}: {result['n_examples']} demos, {result['steps']} steps, "
              f"{result['gpu_seconds']:.1f}s -> {adapter_dir}")
        model = peft_model.unload()

    write_json(run_dir / "summary.json", {"policy": policy, "gate_stats": stats,
                                          "groups": results})
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
