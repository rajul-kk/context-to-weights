# Context-Gap Distillation

*Workshop draft. Numbers marked TBD are filled from `docs/results_skills.md`.*

## Abstract

Skill documents work, and they are expensive: every invocation pays for the document in
context. Distilling a skill into weights removes that cost, but existing recipes distil
uniformly — every token of every demonstration is trained on with equal weight, whether or
not the skill document had anything to do with it. We propose gating skill distillation on
the model's own context gap: the KL divergence, per token, between the frozen backbone's
predictions with the skill document in context and without it. High divergence marks content
the document is actually responsible for; low divergence marks content the model would have
produced regardless. The same divergence serves as the training loss, so importance scoring
and distillation objective are one quantity computed from two forward passes of one frozen
model. On synthetic tool-use skills, KL-gated distillation reaches TBD% task pass rate
against TBD% for uniform distillation at equal steps and TBD% for a random-span control
matched on active-token budget, while removing TBD% of runtime prompt tokens.

## 1. Introduction

A skill document — an API reference, a house style, a runbook — is a prompt-time solution to
a weights-time problem. It works immediately and costs tokens forever. Skill-to-LoRA [1]
shows the trade is worth reversing: distil the skill into an adapter offline, drop the
document from the prompt, save 4.89% of tokens and gain 10.2% aggregate pass rate on
SWE-Skills-Bench.

But S2L distils uniformly. A demonstration response is mostly ordinary language — framing
sentences, code fences, restatements — that a competent base model produces whether or not
it has read the skill document. Spending gradient on that content is at best wasted and at
worst dilutes the parts that matter.

There is a signal available for free that says which is which. Run the frozen backbone
twice over the same demonstration, once with the skill document in context and once without,
and take the per-token KL between the two next-token distributions. That divergence *is* the
document's causal contribution to the model's behaviour, measured in the only units that
matter. Gate on it, and train on it.

Contributions:

1. Context-gap KL as a self-supervised importance signal for skill distillation, computed
   from two forward passes of one frozen backbone — no gradient norms, no RL, no judge.
2. Weighted KL distillation that uses the same divergence as importance score and as loss.
3. A random-span control matched on active-token budget, isolating the gate from the
   confound of simply training on less data.

## 2. Related work

**Skill-to-LoRA [1]** is the closest prior work and our primary baseline. It distils each
skill into its own LoRA by offline self-distillation from synthetic demonstrations, with no
importance gating: all content is weighted equally. We keep its setup and change exactly one
thing — the per-span loss weight — so any difference is attributable to the gate. We also
adopt its retrieval-mismatch evaluation, since robustness to a slightly wrong retrieved
document is a documented benefit of internalisation and worth replicating.

**ThinkSwitch [2]** performs context distillation by removing reasoning traces. Like S2L it
applies no importance or divergence gate; the removal is structural rather than scored.

**PEAM [3]** gates contrastive internalisation on a "parameterization-worthiness" score.
This is the nearest thing in the literature to what we do, and it is a different signal on
different objects: PEAM scores episodic trajectory *pairs*, we score the with-vs-without
context gap on skill *documents*.

**Titans [4]** defines surprise as the gradient of an associative-memory loss with respect
to the input, and uses it to decide what a test-time memory module writes. Ours is a
surprise signal in the same spirit and a different realisation: KL between two forward
passes rather than a gradient norm, and applied to offline skill distillation rather than
online memory writing. We take the framing seriously — a gradient-based and a
divergence-based surprise are not interchangeable, and which one a setting wants is an
empirical question.

## 3. Method

### 3.1 Context-gap scoring

Let `d` be a skill document, `q` a demonstration query and `r` its response. Form two
prefixes: `T = chat(system, d ++ q)` and `S = chat(system, q)`. Run the frozen backbone on
`T ++ r` and on `S ++ r`, and read off the next-token distributions at the positions
predicting each token of `r`. For token `i`,

```
kl_i = KL( p_teacher(· | T, r_<i) || p_student(· | S, r_<i) )
```

Tokens are grouped into spans by sentence and code-line boundaries via the tokenizer's
offset mapping, and a span's score is the mean of its token scores.

### 3.2 Gating

Order spans by score, take the top ρ fraction, and set weight 1 on their tokens and a floor
weight (0 by default) elsewhere. Token-level gating is available as an ablation.

### 3.3 Weighted distillation

Attach a LoRA adapter to the same backbone. The teacher is the *same model with the adapter
disabled* and the document in context; the student is the adapter-enabled model without it.
The loss is the importance-weighted KL,

```
L = sum_i w_i * KL( p_teacher_i || p_student_i ) / sum_i w_i
```

One model in memory, two forward passes per step, one of them under `no_grad`.

### 3.4 Adapter granularity

One adapter per skill *category*. Per-skill adapters make cross-skill forgetting
unmeasurable by construction; one shared adapter across all skills confounds the gate with
multi-skill interference. Categories put two to three related skills behind each adapter, so
interference is measurable and comparable across arms.

## 4. Experimental setup

**Skills.** Eight synthetic tool-use skills across three categories, each a fictional API
with a `SKILL.md` document, nine demonstrations and twelve held-out tasks. The APIs are
invented so the base model cannot know them, which is what makes the context gap large on
identifiers and small on prose. SWE-Skills-Bench-style data is the planned extension.

**Models.** Qwen2.5-1.5B-Instruct primary, TinyLlama-1.1B-Chat for a cross-family check.
LoRA rank 16 on attention and MLP projections. Single T4.

**Arms.** (a) full document in prompt; (b) S2L-style uniform distillation; (c) no document;
(d) random-span control at matched active-token budget; ours, KL-gated.

**Metrics.** Task pass rate — every required string present in the output. Mean prompt
tokens. Out-of-category pass rate, as interference. Pass rate under a mismatched retrieved
document.

## 5. Gate verification

Before anything trains on the gate, it is checked by hand. On two skills and 1050 response
tokens, token KL is heavily right-skewed (median 1.02, mean 2.13, p90 6.06). The
highest-scoring spans are the API rule clauses, the literal call expressions and the error
codes; the lowest are markdown scaffolding (` ```python `, ` ```\n\nNote: `). At the token
level, the subword pieces of `vx_stash`, `vx_fetch` and `VX_TTL_MISSING` are the top scorers
in their demonstrations.

Half of the strings a task requires fall inside the top 25% of spans. The misses are
informative rather than a failure: the required set includes the first positional argument
value, which appears in the query itself, so the student predicts it without the document
and the gate correctly scores it low. The gate measures *what the document adds*, not *what
the answer contains*. Full trace in `docs/kl_gate_check.md`.

## 6. Results

TBD — see `docs/results_skills.md`.

Planned figures: pass rate and token cost across the five arms, with the random-span control
adjacent to ours as the proof-of-mechanism; the KL-threshold sweep over ρ ∈ {0.1, 0.25, 0.5}
at span and token granularity.

## 7. Relation to Project A

This project and Compaction-Supervised Sleep Consolidation are two answers to the same
question — what should a consolidation pass internalise — differing in where the salience
label comes from. Project A takes it from an external decision an agent already makes: the
compactor's keep/drop choice. This project computes it internally from the model's own
behaviour under context ablation. Both are supervised, both avoid RL, both run the same
LoRA machinery. A combined paper comparing an *externally supervised* against a
*self-supervised* consolidation signal is the natural merge, if the results support it.

## 8. Limitations

Synthetic skills with invented identifiers maximise the context gap by construction; real
skill documents overlap more with the model's priors and the gate should be expected to
separate less cleanly. Pass rate is string containment, not execution. One seed per
configuration. The teacher is the same backbone the student adapts, so teacher quality
caps the method.

## References

[1] Skill-to-LoRA. arXiv:2606.16769, Jun 2026.
[2] ThinkSwitch. arXiv:2606.01080, Jun 2026.
[3] PEAM. arXiv:2605.27762, May 2026.
[4] Titans: Learning to Memorize at Test Time. Behrouz, Zhong, Mirrokni. arXiv:2501.00663.
