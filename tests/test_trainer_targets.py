from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from sleep.trainer import IGNORE, SpanDataset, collate, target_forward


class CharTokenizer:
    eos_token = "#"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 97 for c in text]}


def example(prompt, target, kept=True):
    return SimpleNamespace(prompt=prompt, target=target, kept=kept)


def test_long_prompt_keeps_the_target():
    ds = SpanDataset([example("p" * 30, "answer")], CharTokenizer(), max_length=16)
    row = ds[0]
    target = CharTokenizer()("answer#")["input_ids"]
    assert len(row["input_ids"]) == 16
    assert row["input_ids"][-len(target):] == target
    assert [t for t in row["labels"] if t != IGNORE] == target
    assert sum(row["span_mask"]) == len(target)


def test_short_prompt_unchanged():
    row = SpanDataset([example("ab", "cd")], CharTokenizer(), max_length=16)[0]
    assert row["input_ids"] == CharTokenizer()("abcd#")["input_ids"]
    assert row["labels"] == [IGNORE, IGNORE] + CharTokenizer()("cd#")["input_ids"]


def tiny_model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return Qwen2ForCausalLM(cfg).eval()


def test_target_forward_matches_full_logits_loss():
    tok = CharTokenizer()
    ds = SpanDataset([example("hello there", "yes"), example("a much longer prompt here", "no thanks"),
                      example("dropped", "zzz", kept=False)], tok, max_length=20)
    batch = collate([ds[i] for i in range(len(ds))], pad_id=0)
    model = tiny_model()
    with torch.no_grad():
        ref = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"]).loss
        _, ce = target_forward(model, batch)
    assert ce.numel() == len(tok("yes#")["input_ids"]) + len(tok("no thanks#")["input_ids"])
    assert torch.allclose(ce.mean(), ref, atol=1e-5)


def test_target_forward_with_nothing_labelled():
    ds = SpanDataset([example("only", "mask", kept=False)], CharTokenizer(), max_length=20)
    _, ce = target_forward(tiny_model(), collate([ds[0]], pad_id=0))
    assert ce.numel() == 0


def test_target_forward_through_lora_backpropagates():
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(tiny_model().train(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                             task_type="CAUSAL_LM"))
    ds = SpanDataset([example("hello there", "yes")], CharTokenizer(), max_length=20)
    _, ce = target_forward(model, collate([ds[0]], pad_id=0))
    ce.mean().backward()
    grads = [p.grad for n, p in model.named_parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
