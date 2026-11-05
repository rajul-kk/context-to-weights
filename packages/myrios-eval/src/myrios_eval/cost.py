FLOPS_METHOD = ("2N per token inference, 4N+6N_lora per token LoRA training, 6N per token full "
                "training; attention term quadratic in length ignored")


def lora_params(shapes, rank, n_layers=1):
    if rank < 0 or n_layers < 0:
        raise ValueError("rank and n_layers must be non-negative")
    return int(n_layers * sum(rank * (d_in + d_out) for d_in, d_out in shapes))


def qwen_attention_shapes(hidden, n_heads, n_kv_heads, targets=("q", "k", "v", "o")):
    head = hidden // n_heads
    kv = head * n_kv_heads
    table = {"q": (hidden, hidden), "k": (hidden, kv), "v": (hidden, kv), "o": (hidden, hidden)}
    return [table[t] for t in targets]


def inference_flops(n_params, tokens):
    return 2.0 * n_params * tokens


def lora_train_flops(n_params, tokens, n_lora_params):
    return 4.0 * n_params * tokens + 6.0 * n_lora_params * tokens


def full_train_flops(n_params, tokens):
    return 6.0 * n_params * tokens
