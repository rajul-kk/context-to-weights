import pytest

pytest.importorskip("torch")

from sleep.lm import LOADERS, chat_text


class StubTokenizer:
    def __init__(self, chat_kwargs=None):
        if chat_kwargs is not None:
            self.chat_kwargs = chat_kwargs
        self.seen = None

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kwargs):
        self.seen = kwargs
        return "rendered"


class BrokenTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("no template")


def test_chat_text_passes_template_kwargs_when_configured():
    tok = StubTokenizer({"enable_thinking": False})
    assert chat_text(tok, "sys", "hi") == "rendered"
    assert tok.seen == {"enable_thinking": False}


def test_chat_text_passes_nothing_by_default():
    tok = StubTokenizer()
    chat_text(tok, None, "hi")
    assert tok.seen == {}


def test_chat_text_falls_back_when_the_template_fails():
    assert chat_text(BrokenTokenizer(), "sys", "hi") == "sys\n\nhi\n\nAnswer:"


def test_loader_names_cover_causal_and_image_text():
    assert set(LOADERS) == {"causal", "image_text"}
