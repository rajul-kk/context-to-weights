import json
import subprocess
import sys

import pytest
from helpers import ROOT

from eval import kvquant_report


def summary(model, scheme, bits, seed, hit, sigma=6.0, content=0.2):
    return {"model": model, "label": "read", "n": 384, "kv_scheme": scheme, "kv_bits": bits,
            "seed": seed, "hit_rate": hit, "content_dependence": content,
            "sigma_over_constant": sigma}


def rows_for(model, scheme_hits, baseline=(0.40, 0.42, 0.41)):
    rows = [summary(model, "none", 0, s, h) for s, h in enumerate(baseline)]
    for scheme, by_bits in scheme_hits.items():
        for bits, hits in by_bits.items():
            rows += [summary(model, scheme, bits, s, h) for s, h in enumerate(hits)]
    return rows


def test_report_finds_a_cliff_between_held_and_collapsed_bits():
    rows = rows_for("org/M", {"token": {8: (0.40, 0.42, 0.41), 4: (0.40, 0.41, 0.41),
                                        2: (0.10, 0.12, 0.11)}})
    text, verdicts = kvquant_report.render(rows)
    assert verdicts["org/M"]["schemes"]["token"] == "cliff at 2 bits, holds at 4"
    assert "| token | 2 |" in text


def test_report_says_nothing_to_guide_when_every_bit_width_holds():
    rows = rows_for("org/M", {"token": {8: (0.41, 0.42, 0.40), 2: (0.40, 0.42, 0.42)}})
    _, verdicts = kvquant_report.render(rows)
    assert "nothing to guide" in verdicts["org/M"]["schemes"]["token"]


def test_report_flags_collapse_already_at_the_highest_bit_width():
    rows = rows_for("org/M", {"token": {8: (0.10, 0.12, 0.11), 4: (0.10, 0.10, 0.10)}})
    _, verdicts = kvquant_report.render(rows)
    assert verdicts["org/M"]["schemes"]["token"] == "already degraded at 8 bits"


def test_report_requires_two_paired_seeds():
    rows = [summary("org/M", "none", 0, 0, 0.4), summary("org/M", "token", 4, 0, 0.3)]
    _, verdicts = kvquant_report.render(rows)
    assert verdicts["org/M"]["schemes"]["token"] == "not enough paired seeds"


def test_report_invalidates_a_baseline_at_chance():
    rows = [summary("org/M", "none", 0, s, 0.13, sigma=0.5) for s in range(3)]
    text, verdicts = kvquant_report.render(rows)
    assert not verdicts["org/M"]["valid"] and "INVALID" in text


def test_report_handles_a_missing_baseline():
    rows = [summary("org/M", "token", 4, s, 0.3) for s in range(3)]
    text, verdicts = kvquant_report.render(rows)
    assert not verdicts["org/M"]["valid"] and "No full-precision baseline" in text


def test_report_separates_schemes():
    rows = rows_for("org/M", {"token": {4: (0.10, 0.12, 0.11)},
                              "channel_keys": {4: (0.40, 0.41, 0.42)}})
    _, verdicts = kvquant_report.render(rows)
    assert verdicts["org/M"]["schemes"]["token"] == "already degraded at 4 bits"
    assert "nothing to guide" in verdicts["org/M"]["schemes"]["channel_keys"]


def test_report_cli_writes_markdown_and_fails_on_empty(tmp_path):
    for i, row in enumerate(rows_for("org/M", {"token": {4: (0.4, 0.41, 0.42)}})):
        folder = tmp_path / "runs" / f"r{i}"
        folder.mkdir(parents=True)
        (folder / "read.summary.json").write_text(json.dumps(row), encoding="utf-8")
    out = tmp_path / "report.md"
    done = subprocess.run([sys.executable, "-m", "eval.kvquant_report", "--run-root",
                           str(tmp_path / "runs"), "--out", str(out)], cwd=ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "KV quantization gate" in out.read_text(encoding="utf-8")
    empty = subprocess.run([sys.executable, "-m", "eval.kvquant_report", "--run-root",
                            str(tmp_path / "none")], cwd=ROOT, capture_output=True, text=True)
    assert empty.returncode != 0


torch = pytest.importorskip("torch")

from declare.elicit import (fake_quantize, key_validity, quantize_per_channel,
                            quantize_per_token, register_quant)


def test_zero_bits_is_the_identity():
    x = torch.randn(1, 2, 8, 64)
    assert fake_quantize(x, 0, 32) is x


def test_per_token_error_shrinks_with_more_bits():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, 64)
    errors = [(quantize_per_token(x, b, 32) - x).abs().mean().item() for b in (2, 4, 8)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.01


def test_per_token_uses_at_most_two_to_the_bits_levels_per_group():
    x = torch.randn(1, 1, 4, 64)
    q = quantize_per_token(x, 2, 32).reshape(1, 1, 4, 2, 32)
    for token in range(4):
        for g in range(2):
            assert q[0, 0, token, g].unique().numel() <= 4


def test_per_token_is_independent_across_tokens():
    x = torch.randn(1, 1, 6, 64)
    full = quantize_per_token(x, 3, 32)
    alone = quantize_per_token(x[:, :, 2:3], 3, 32)
    assert torch.equal(full[:, :, 2:3], alone)


def test_per_channel_keeps_the_trailing_residual_unquantized():
    x = torch.randn(1, 2, 70, 64)
    q = quantize_per_channel(x, 2, 32)
    assert torch.equal(q[:, :, 64:], x[:, :, 64:])
    assert not torch.equal(q[:, :, :64], x[:, :, :64])


def test_per_channel_returns_input_when_shorter_than_one_group():
    x = torch.randn(1, 2, 10, 64)
    assert quantize_per_channel(x, 2, 32) is x


def test_per_channel_ignores_padded_positions_when_choosing_the_grid():
    torch.manual_seed(1)
    x = torch.randn(1, 1, 32, 8)
    padded = x.clone()
    padded[:, :, :8] = 1000.0
    valid = torch.ones(1, 32, dtype=torch.bool)
    valid[:, :8] = False
    masked = quantize_per_channel(padded, 3, 32, valid)
    unmasked = quantize_per_channel(padded, 3, 32)
    assert (masked[:, :, 8:] - x[:, :, 8:]).abs().mean() < 0.5
    assert (unmasked[:, :, 8:] - x[:, :, 8:]).abs().mean() > 1.5


def test_key_validity_reads_the_last_query_row():
    mask = torch.tensor([[[[False, True, True], [False, True, True], [False, True, True]]]])
    assert key_validity(mask, 3).tolist() == [[False, True, True]]
    additive = torch.tensor([[[[0.0, 0.0, 0.0], [-1e9, 0.0, 0.0], [-1e9, 0.0, 0.0]]]])
    assert key_validity(additive, 3).tolist() == [[False, True, True]]
    assert key_validity(None, 3) is None


def test_register_quant_rejects_unknown_schemes():
    with pytest.raises(ValueError):
        register_quant(4, scheme="nonsense")
