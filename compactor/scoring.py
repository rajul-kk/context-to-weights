import torch

from compactor.base import Compactor, _extractive_summary

SCORING_SYSTEM = (
    "You judge whether a single line from an engineering conversation must be preserved "
    "verbatim so a later question about the conversation stays answerable."
)

SCORING_TEMPLATE = """Line from the conversation:
{text}

If this line were deleted, would a later question about the conversation become
unanswerable? Concrete decisions, names, versions, identifiers, numeric limits, ownership
and commitments matter. Chit-chat, restatements and generic advice do not.

Answer Yes or No."""


class ScoringCompactor(Compactor):
    name = "scoring"

    def __init__(self, model, tokenizer, batch_size=16, max_length=512):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.calls = 0
        self.yes_ids = self._variants(["Yes", " Yes", "yes", " yes"])
        self.no_ids = self._variants(["No", " No", "no", " no"])

    def _variants(self, words):
        ids = set()
        for w in words:
            enc = self.tokenizer.encode(w, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    @torch.no_grad()
    def score_spans(self, texts):
        from sleep.lm import chat_text

        device = next(self.model.parameters()).device
        scores = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i: i + self.batch_size]
            prompts = [chat_text(self.tokenizer, SCORING_SYSTEM,
                                 SCORING_TEMPLATE.format(text=t)) for t in chunk]
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_length).to(device)
            logits = self.model(**enc).logits[:, -1].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            yes = torch.logsumexp(logprobs[:, self.yes_ids], dim=-1)
            no = torch.logsumexp(logprobs[:, self.no_ids], dim=-1)
            scores.extend((yes - no).tolist())
        return scores

    def select(self, spans, budget):
        self.calls += 1
        self.last_decided_by = "scoring"
        scores = self.score_spans([s["text"] for s in spans])
        ranked = sorted(range(len(spans)), key=lambda i: -scores[i])
        kept = ranked[:budget]
        dropped = [spans[i]["text"] for i in ranked[budget:]]
        return kept, _extractive_summary(dropped)
