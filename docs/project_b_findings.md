# Project B results

Two questions, answered separately.

1. **Does the with/without-context KL carry a usable importance signal?** Weakly, yes — but
   only when scored per token. Pooled over spans it inverts.
2. **Does gating distillation on it beat uniform distillation?** No. The first run was
   refuted, and the diagnosis is that the gate was selecting the wrong tokens.

## The run

`Qwen2.5-1.5B-Instruct`, 8 toy skills grouped into 3 categories, 300 steps per group,
`gate.granularity: span`, `top_frac: 0.25`, `floor_weight: 0.0`.

| method | in-group pass |
|---|---|
| (a) full skill text in prompt (ceiling) | 0.708 |
| (b) S2L-style uniform distillation | **0.708** |
| (d) random-span control | 0.583 |
| ours: KL-gated distillation | **0.510** |
| (c) no skill (floor) | 0.000 |

Uniform distillation matches the full-prompt ceiling: the skill *is* fully internalisable at
this scale, at 73.5% fewer runtime tokens (234 -> 62). That is a clean replication of the S2L
result and the one positive number in the project.

The gate then loses to both uniform and its own random control. One secondary result
survives: **ours-mismatched 0.281 vs prompt-mismatched 0.000** — an internalised skill keeps
working when the retrieved document is the wrong one, where a prompt-based skill collapses.

## Why the gate failed

Not because gating is a bad idea. Because span-mean pooling inverted the signal.

`inspect_gate.py` measures **required-token coverage**: the fraction of each demo's
`required` strings — the API identifiers the answer must contain — that fall inside the
gate's selection, against a **matched-budget random control** over 20 permutations.

| granularity | top_frac | gate coverage | control | lift | margin | verdict |
|---|---|---|---|---|---|---|
| **span** (as run) | 0.25 | **0.118** | 0.307 | 0.38x | **-5.14σ** | below control |
| span | 0.50 | 0.634 | 0.561 | 1.13x | +1.37σ | indistinguishable |
| **token** | 0.25 | **0.299** | 0.252 | 1.19x | **+2.62σ** | clears control |
| token | 0.50 | 0.557 | 0.496 | 1.12x | +3.67σ | clears control |

The configuration that was trained selects required content at **a third of chance rate**.
It is not weakly informative; it is anti-correlated.

The mechanism is visible in the spans. Highest mean-KL:

```
5.290  [harrowdb]      'every append must carry an actor; scans without a checkpoint are refused.'
4.256  [quarrybuild]   'sandbox must be strict for anything published; loose sandboxes are local-only.'
4.102  [obsidian-flags] 'every flip needs a reason string; Obsidian writes it to the audit trail.'
```

These are the prose rule-restatements at the end of each demo. Every token is moderately
surprising, so the mean is high. A code span — `vx_stash("batch", key=key, ttl_s=3600)` — is
three genuinely high-KL identifier tokens buried in a dozen near-zero syntax tokens, so its
mean is low. The gate bought prose and skipped code, and the required content is in the code.

The signal itself was never the problem. At token level the ranking is exactly right:

```
[veltrix-cache d1 read] 'rejects':7.94, 'V':7.17, 'fetch':7.00, 'x':6.99, 'write':5.79
```

**Mean-pooling a spiky signal destroys it.** `kl_p90 / kl_median` is 7.23 / 0.88, about 8x —
that ratio alone is a warning not to average.

Coverage also predicts the downstream result monotonically across the three arms actually
trained:

| | required coverage | in-group pass |
|---|---|---|
| ours (span) | 0.118 | 0.510 |
| random control | ~0.31 | 0.583 |
| uniform | 1.000 | 0.708 |

Three points is not a fit, but it is consistent, and it is the simplest explanation available.

A second, smaller contributor: `floor_weight: 0.0` gives non-selected tokens zero gradient.
Training saw the high-surprise tokens with the low-surprise prefix that *conditions* them
down-weighted to nothing, so the autoregressive path to those tokens was never trained. This
is the same shape as Project A's failure — see
[project_a_findings.md](project_a_findings.md), where the adapter was trained on
`cue -> declarative sentence` and evaluated on `question -> answer`. Both projects broke the
*path* to the knowledge rather than the knowledge itself.

## The methodological failure

The gate check existed and was run. `gate_report.json` recorded
`required_in_top_frac: 0.125` beside `top_frac: 0.25`, and nothing compared the two. There
was no control, so a number that should have stopped the run read as unremarkable.

Worse, the check had *passed* earlier at 0.50 on a two-skill debug slice with SmolLM2-360M
([kl_gate_check.md](kl_gate_check.md)), and that document contains a written rationalisation
for why 0.50 was acceptable. The check was then re-run on the real eight-skill scoring set,
returned 0.125, and the rationalisation carried over unexamined.

Project A had a positional control for its salience signal from early on. Project B had no
equivalent for its gate until after the run. That asymmetry is the whole story.

**Fixed.** `inspect_gate.py` now reports coverage against a matched-budget random control
with a σ margin and a `clears control` / `indistinguishable` / `below control` verdict, and
prints a warning when the gate does not clear. `run_skills.py` scopes every artifact by
granularity so arms cannot overwrite each other.

## The token-granularity arm

One flag, roughly 40 minutes:

```bash
python scripts/run_skills.py --config configs/kaggle_skills.yaml \
  --granularity token --stages score,distill,eval,report
```

Writes `scores_token.jsonl`, `gate_report_token.json`, `distill_*_token_0.25/`,
`report_token/` and `docs/results_skills_token.md`, leaving the span arm intact. Enabled in
[b1_skills.ipynb](../notebooks/b1_skills.ipynb) via `RUN_TOKEN_ARM`.

**Prediction, recorded before the run:** ours lands between the random control and uniform,
and still below uniform. Uniform has coverage 1.0 by construction; a 1.19x gate over a 25%
budget cannot make up that deficit. If that holds, the claim is not "the gate is broken" but
the stronger and more useful:

> Importance gating cannot beat uniform distillation at this scale, because uniform already
> achieves perfect coverage of the required content and the corpus is small enough to train
> on in full. A gate is a coverage sacrifice, and it buys nothing until the corpus is large
> enough that full coverage is unaffordable.

That is a falsifiable statement about *when* gating should start to pay, rather than a null.

## Consequence for the writeup

The headline claim — KL-gated distillation beats uniform — is **refuted**. What survives:

- **Uniform self-distillation reaches the full-prompt ceiling** at 73.5% fewer runtime
  tokens, and the internalised skill degrades gracefully under a mismatched retrieved
  document (0.281 vs 0.000) where prompting does not.
- **Aggregation can invert a signal.** A real per-token signal, mean-pooled over spans,
  scored -5.14σ against its own control. This is artifact #8 for the paper's §6 table and
  the first one that hurt the result rather than flattering it.
- **Precondition tests need controls, not thresholds.** 0.125 looked like a low number.
  0.38x-of-chance was the number that mattered, and nothing computed it.

See [paper/draft_combined.md](../paper/draft_combined.md).
