import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

transformers.logging.set_verbosity_error()

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(cfg, device=None):
    name = cfg["model"]["base"]
    device = device or pick_device()
    dtype = DTYPES[cfg["model"]["dtype"]]
    if device == "cpu":
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def chat_text(tokenizer, system, user):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prefix = f"{system}\n\n" if system else ""
        return f"{prefix}{user}\n\nAnswer:"


def make_generator(model, tokenizer):
    def generate(system, user, max_new_tokens=128):
        return batch_generate(model, tokenizer, [(system, user)], max_new_tokens)[0]

    return generate


@torch.no_grad()
def batch_generate(model, tokenizer, pairs, max_new_tokens=64, batch_size=8):
    device = next(model.parameters()).device
    outputs = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i: i + batch_size]
        texts = [chat_text(tokenizer, s, u) for s, u in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=model.config.max_position_embeddings).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for j in range(len(chunk)):
            new_tokens = gen[j][enc["input_ids"].shape[1]:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


@torch.no_grad()
def token_ce(model, tokenizer, prompt, target, max_length=1024):
    device = next(model.parameters()).device
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not t_ids:
        return []
    ids = (p_ids + t_ids)[-max_length:]
    n_target = min(len(t_ids), len(ids))
    input_ids = torch.tensor([ids], device=device)
    logits = model(input_ids=input_ids).logits[0].float()
    shift_logits = logits[:-1]
    shift_labels = input_ids[0][1:]
    losses = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="none")
    return losses[-n_target:].tolist()
