# Project B results

1. **Does the with/without-context KL carry a usable importance signal?** Yes. On the current
   environment the span gate selects required content at **+6.40σ** over a matched control.
2. **Does gating distillation on it beat uniform distillation?** Unmeasured. The one
   comparison run trained on a gate scored under an older tokenizer, which inverted it. A
   re-run against the corrected gate is pending.

## The run

`Qwen2.5-1.5B-Instruct`, 8 toy skills in 3 categories, 300 steps per group,
`gate.granularity: span`, `top_frac: 0.25`, `floor_weight: 0.0`.

| method | in-group pass |
|---|---|
| (a) full skill text in prompt (ceiling) | 0.708 |
| (b) S2L-style uniform distillation | **0.708** |
| (d) random-span control | 0.583 |
| ours: KL-gated distillation (stale gate) | 0.510 |
| (c) no skill (floor) | 0.000 |

**Uniform distillation matches the full-prompt ceiling** at 73.5% fewer runtime tokens
(234 -> 62), a clean replication of S2L. An internalised skill also keeps working under a
mismatched retrieved document (**0.281 vs 0.000** for the prompted skill).

The `ours` row is not a verdict on gating: it trained on the stale gate described below.

## The gate, before and after the tokenizer change

`inspect_gate.py` measures **required-token coverage** — the fraction of each demo's
`required` API identifiers inside the gate's selection — against a **matched-budget random
control** over 20 permutations.

| score file | granularity | top_frac | coverage | control | margin | verdict |
|---|---|---|---|---|---|---|
| **current** (transformers 5.0, 3,326 tokens) | span | 0.25 | **0.586** | 0.392 | **+6.40σ** | clears control |
| stale (older tokenizer, 3,871 tokens) | span | 0.25 | 0.118 | 0.307 | -5.14σ | below control |
| stale | span | 0.50 | 0.634 | 0.561 | +1.37σ | indistinguishable |
| stale | token | 0.25 | 0.299 | 0.252 | +2.62σ | clears control |
| stale | token | 0.50 | 0.557 | 0.496 | +3.67σ | clears control |

The stale rows come from a local `scores_span.jsonl` scored under an older tokenizer that
split identifiers such as `vx_stash` into five pieces. A code span then became a few high-KL
identifier fragments among many near-zero syntax tokens, so its mean was low, while prose
rule-restatements scored uniformly moderate and won:

```
5.290  [harrowdb]      'every append must carry an actor; scans without a checkpoint are refused.'
4.256  [quarrybuild]   'sandbox must be strict for anything published; loose sandboxes are local-only.'
```

Under that tokenization span-mean pooling bought prose and skipped code. Under the current
one it clears its control. **The -5.14σ is retracted**; it was a property of the tokenizer,
not of the gate. The token-granularity rows share the stale file and need re-scoring.

One contributor independent of the tokenizer: `floor_weight: 0.0` gives non-selected tokens
zero gradient, so the low-surprise prefix that conditions a high-surprise token is never
trained. This breaks the *path* to the knowledge, the same shape as Project A's
`cue -> sentence` vs `question -> answer` mismatch
([project_a_findings.md](project_a_findings.md)).

## Methodological record

- **The first gate check had no control.** `gate_report.json` recorded
  `required_in_top_frac: 0.125` beside `top_frac: 0.25` and nothing compared them. An earlier
  pass at 0.50 on a two-skill debug slice ([kl_gate_check.md](kl_gate_check.md)) had been
  rationalised in writing. `inspect_gate.py` now reports a σ margin against a matched-budget
  control with a verdict and a warning.
- **The same gate scored -5.14σ and +6.40σ on two tokenizers.** Nothing in the code changed.
  A score file is only valid in the environment that produced it; re-score before trusting a
  cached one.

## Pending

Re-run the downstream comparison on the corrected gate (about 40 min):

```bash
python scripts/run_skills.py --config configs/kaggle_skills.yaml \
  --granularity span --stages score,distill,eval,report
```

Add `--granularity token` for the token arm; artifacts are scoped by granularity
(`scores_token.jsonl`, `report_token/`, `docs/results_skills_token.md`). Both are wired into
[b1_skills.ipynb](../notebooks/b1_skills.ipynb).

**Prediction, recorded before the run:** gated distillation lands between the random control
and uniform, still below uniform. Uniform has coverage 1.0 by construction, and at this
corpus size full coverage is affordable, so a gate is a coverage sacrifice that buys nothing.
It should start to pay only when the corpus is too large to train on in full.

See [paper/draft_combined.md](../paper/draft_combined.md).
