import importlib.util
import os
import sys
from pathlib import Path

import pytest
from helpers import ROOT
from myrios_eval.cost import lora_params, lora_train_flops, qwen_attention_shapes
from myrios_eval.stats import bootstrap_ci

SKILL_LIBRARY = Path(os.environ.get("SKILL_LIBRARY_ROOT", ROOT.parent / "skill-library"))
PACKAGE = SKILL_LIBRARY / "skillmetrics"


@pytest.fixture(scope="module")
def skillmetrics():
    if not (PACKAGE / "__init__.py").exists():
        pytest.skip(f"{PACKAGE} not present")
    name = "_skill_library_skillmetrics"
    spec = importlib.util.spec_from_file_location(name, PACKAGE / "__init__.py",
                                                  submodule_search_locations=[str(PACKAGE)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]


def value(attr):
    return attr() if callable(attr) else attr


def test_lora_parameter_counts_agree(skillmetrics):
    for name, shape in skillmetrics.SHAPES.items():
        shapes = qwen_attention_shapes(shape.d_model, shape.n_heads, shape.n_kv_heads)
        for rank in (8, 16):
            assert lora_params(shapes, rank, n_layers=shape.n_layer) == shape.lora_params(rank), name


def test_lora_training_flops_agree_once_the_attention_term_is_removed(skillmetrics):
    flops = sys.modules["_skill_library_skillmetrics.flops"]
    shape = skillmetrics.SHAPES["Qwen/Qwen2.5-1.5B-Instruct"]
    n = value(shape.non_embedding_params) + shape.head_params()
    tokens, tokens_sq, n_lora = 120_000.0, 120_000.0 * 300, shape.lora_params(16)
    theirs = flops.lora_train_flops(shape, tokens, tokens_sq, n_lora)
    attention = 2.0 * flops._attn(shape, (tokens_sq + tokens) / 2.0)
    assert lora_train_flops(n, tokens, n_lora) == pytest.approx(theirs - attention, rel=1e-12)


def test_bootstrap_intervals_agree(skillmetrics):
    for values in ([0.70, 0.74, 0.73, 0.75, 0.71, 0.72],
                   [0.1, 0.9, 0.4, 0.4, 0.6, 0.2, 0.8, 0.5, 0.3, 0.7]):
        theirs = skillmetrics.mean_ci(values, n_boot=2000, seed=0)
        lo, hi = bootstrap_ci(values, n_resamples=2000, seed=0)
        assert (lo, hi) == pytest.approx((theirs["lo"], theirs["hi"]), abs=1e-12)
