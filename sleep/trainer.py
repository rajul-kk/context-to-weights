import statistics
import time

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset

IGNORE = -100


class SpanDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.rows = []
        for ex in examples:
            p_ids = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
            t_ids = tokenizer(ex.target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids = (p_ids + t_ids)[:max_length]
            n_target = max(1, len(ids) - len(p_ids))
            labels = [IGNORE] * (len(ids) - n_target) + ids[-n_target:]
            if not ex.kept:
                labels = [IGNORE] * len(ids)
            span_mask = [0] * (len(ids) - n_target) + [1] * n_target
            self.rows.append(
                {
                    "input_ids": ids,
                    "labels": labels,
                    "span_mask": span_mask,
                    "kept": float(ex.kept),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": [], "span_mask": [], "kept": []}
    for b in batch:
        pad = width - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * pad)
        out["labels"].append(b["labels"] + [IGNORE] * pad)
        out["span_mask"].append(b["span_mask"] + [0] * pad)
        out["kept"].append(b["kept"])
    return {
        "input_ids": torch.tensor(out["input_ids"]),
        "attention_mask": torch.tensor(out["attention_mask"]),
        "labels": torch.tensor(out["labels"]),
        "span_mask": torch.tensor(out["span_mask"]),
        "kept": torch.tensor(out["kept"]),
    }


class MaskHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden):
        return self.proj(hidden).squeeze(-1)


def attach_lora(model, cfg):
    lora = LoraConfig(
        r=cfg["sleep"]["lora_r"],
        lora_alpha=cfg["sleep"]["lora_alpha"],
        lora_dropout=cfg["sleep"]["lora_dropout"],
        target_modules=cfg["sleep"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def load_or_attach(model, cfg, adapter_dir=None):
    if adapter_dir is not None:
        peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
        return peft_model
    return attach_lora(model, cfg)


def per_token_ce(logits, labels):
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    losses = nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=IGNORE,
    )
    valid = (shift_labels.reshape(-1) != IGNORE)
    return losses[valid]


@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_ce = []
    for batch in loader:
        batch = _to_device(batch, model)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        ce = per_token_ce(out.logits, batch["labels"])
        all_ce.extend(ce.tolist())
    model.train()
    if not all_ce:
        return {"val_ce_median": float("nan"), "val_ce_mean": float("nan"), "val_tokens": 0}
    return {
        "val_ce_median": float(statistics.median(all_ce)),
        "val_ce_mean": float(sum(all_ce) / len(all_ce)),
        "val_tokens": len(all_ce),
    }


def _to_device(batch, model):
    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in batch.items()}


def train_sleep_phase(model, tokenizer, cfg, train_examples, val_examples=None,
                      mask_head=None, log_every=25, on_log=None):
    sc = cfg["sleep"]
    max_length = cfg["model"]["max_length"]
    train_ds = SpanDataset(train_examples, tokenizer, max_length)
    if len(train_ds) == 0:
        return {"skipped": True, "steps": 0}

    pad_id = tokenizer.pad_token_id
    loader = DataLoader(
        train_ds, batch_size=sc["batch_size"], shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )
    val_loader = None
    if val_examples:
        val_ds = SpanDataset(val_examples, tokenizer, max_length)
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds, batch_size=sc["batch_size"], shuffle=False,
                collate_fn=lambda b: collate(b, pad_id),
            )

    params = [p for p in model.parameters() if p.requires_grad]
    if mask_head is not None:
        params += list(mask_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(sc["lr"]))

    model.train()
    history = []
    step = 0
    accum = max(1, sc["grad_accum"])
    t0 = time.time()
    bce = nn.BCEWithLogitsLoss(reduction="none")

    while step < sc["steps"]:
        for batch in loader:
            if step >= sc["steps"]:
                break
            batch = _to_device(batch, model)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_hidden_states=mask_head is not None,
            )
            loss = out.loss if out.loss is not None else torch.zeros((), device=batch["input_ids"].device)
            aux = torch.zeros((), device=loss.device)
            if mask_head is not None:
                hidden = out.hidden_states[-1]
                logits = mask_head(hidden.float())
                target = batch["kept"].unsqueeze(1).expand_as(logits)
                weight = batch["span_mask"].float()
                aux = (bce(logits, target) * weight).sum() / weight.sum().clamp(min=1.0)
                loss = loss + float(cfg["sleep"]["mask_head"]["weight"]) * aux

            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % log_every == 0 or step == sc["steps"]:
                entry = {"step": step, "train_loss": float(loss.item()), "aux_loss": float(aux.item())}
                if val_loader is not None:
                    entry.update(validate(model, val_loader))
                history.append(entry)
                if on_log:
                    on_log(entry)

    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return {
        "skipped": False,
        "steps": step,
        "n_train_examples": len(train_ds),
        "n_val_examples": len(val_examples) if val_examples else 0,
        "gpu_seconds": time.time() - t0,
        "history": history,
    }
