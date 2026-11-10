import pytest

torch = pytest.importorskip("torch")

from declare import elicit
from declare.elicit import capturing, probe_attention, quant_attention_for, register_quant
from sleep.lm import left_truncation


def qkv():
    torch.manual_seed(0)
    return torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)


def test_probe_captures_nothing_outside_a_capture():
    q, k, v = qkv()
    probe_attention(None, q, k, v)
    assert elicit._capture is None


def test_probe_captures_one_row_per_call_inside_a_capture():
    q, k, v = qkv()
    with capturing() as rows:
        probe_attention(None, q, k, v)
        probe_attention(None, q, k, v)
    assert len(rows) == 2
    assert rows[0].shape == (1, 2, 5)
    assert torch.allclose(rows[0].sum(-1), torch.ones(1, 2), atol=1e-4)


def test_calls_outside_a_capture_do_not_leak_into_the_next_one():
    q, k, v = qkv()
    probe_attention(None, q, k, v)
    with capturing() as rows:
        pass
    assert rows == []


def test_capture_closes_after_an_exception():
    with pytest.raises(ValueError):
        with capturing():
            raise ValueError("boom")
    assert elicit._capture is None


def test_captures_do_not_nest():
    with capturing():
        with pytest.raises(RuntimeError, match="already active"):
            with capturing():
                pass


def test_quant_functions_are_independent_of_registration_order():
    q, k, v = qkv()
    coarse = quant_attention_for(2, 4, "token")
    fine = quant_attention_for(8, 4, "token")
    before = coarse(None, q, k, v)[0]
    fine(None, q, k, v)
    assert torch.equal(coarse(None, q, k, v)[0], before)
    assert not torch.equal(coarse(None, q, k, v)[0], fine(None, q, k, v)[0])


def test_register_quant_names_encode_the_configuration():
    assert register_quant(4, scheme="token") != register_quant(2, scheme="token")
    assert register_quant(4, scheme="token") != register_quant(4, scheme="channel_keys")
    assert register_quant(4, scheme="token") == register_quant(4, scheme="token")


def test_register_quant_rejects_non_positive_bits():
    with pytest.raises(ValueError, match="at least 1"):
        register_quant(0)


class StubTokenizer:
    truncation_side = "right"


def test_left_truncation_restores_the_side():
    tok = StubTokenizer()
    with left_truncation(tok):
        assert tok.truncation_side == "left"
    assert tok.truncation_side == "right"


def test_left_truncation_restores_the_side_after_an_exception():
    tok = StubTokenizer()
    with pytest.raises(ValueError):
        with left_truncation(tok):
            raise ValueError("boom")
    assert tok.truncation_side == "right"
