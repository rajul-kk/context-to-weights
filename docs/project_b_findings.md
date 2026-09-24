# Project B results

1. **Does the with/without-context KL carry a usable importance signal?** Yes. The span gate
   selects required content at **+6.77σ** over a matched-budget control.
2. **Does gating distillation on it beat uniform distillation?** **No.** At the original
   `floor_weight: 0` it loses (0.510 vs 0.708). With a 0.1 floor it ties uniform (0.729) — and
   so does random-span selection at the same floor (0.729). The signal is real at selection
   time and confers no downstream advantage.

## The run

`Qwen2.5-1.5B-Instruct`, 8 toy skills in 3 categories, 300 steps per group,
`gate.granularity: span`, `top_frac: 0.25`, `floor_weight: 0.0`. Batch job on a Kaggle T4,
2026-09-24, source `c37bcd7`, fresh scoring and no restored artifacts.

| method | in-group pass (n=96) |
|---|---|
| (a) full skill text in prompt (ceiling) | 0.708 |
| (b) S2L-style uniform distillation | **0.708** |
| (d) random-span control | 0.583 |
| ours: KL-gated distillation | **0.510** |
| (c) no skill (floor) | 0.000 |

| comparison | difference | margin |
|---|---|---|
| ours vs uniform | -0.198 | about -2.9σ |
| ours vs random-span control | -0.073 | about -1.0σ, not significant |
| uniform vs random-span control | +0.125 | about +1.8σ |

**Uniform distillation matches the full-prompt ceiling** at 69% fewer runtime tokens
(201 -> 62), a clean replication of S2L. An internalised skill also keeps partial function
under a mismatched retrieved document (**0.292 vs 0.000** for the prompted skill).

These numbers reproduce an earlier run exactly (0.510 / 0.708 / 0.583), so that run also used
a correctly scored gate.

## The gate works; training on it does not

`inspect_gate.py` measures **required-token coverage** — the fraction of each demo's
`required` API identifiers inside the gate's selection — against a **matched-budget random
control** over 20 permutations.

| score file | coverage | control | margin | verdict |
|---|---|---|---|---|
| **batch run** (2026-09-24, transformers 5.0) | **0.597** | 0.392 | **+6.77σ** | clears control |
| earlier Kaggle scoring | 0.586 | 0.392 | +6.40σ | clears control |
| stale local file (older tokenizer, 3,871 tokens) | 0.118 | 0.307 | -5.14σ | retracted |

The stale file split identifiers such as `vx_stash` into five pieces, which drowned code spans
under span-mean pooling. That number is retracted; it never reflected the gate.

So selection is not the problem: the gate covers required content at 1.5x its control, and
the gated adapter still trains worse than one fed random spans.

## Floor-weight ablation

At `floor_weight: 0` every non-selected token gets zero gradient, so the low-surprise prefix
that conditions each selected identifier is never trained. Same gate (+6.77σ), same 300
steps, non-selected tokens given a small weight instead (batch job, 2026-09-24):

| floor weight | KL-gated | random-span | uniform |
|---|---|---|---|
| 0 | 0.510 | 0.583 | 0.708 |
| **0.1** | **0.729** | **0.729** | 0.708 |
| 0.3 | 0.708 | 0.656 | 0.708 |

In-group pass, n=96 each.

- **The zero floor was the defect.** A 0.1 floor lifts gated distillation by +0.219 (about
  3.2σ), to uniform's level. The gradient-path account is confirmed.
- **The gate then adds nothing over random selection.** At 0.1 both reach 0.729. At 0.3 the
  gate leads by 0.052 (about 0.8σ, not significant).
- **So the loss to uniform at floor 0 was an artifact of the weighting, not evidence that
  gating hurts.** What survives is weaker and cleaner: a verified importance signal (+6.77σ at
  selection) buys no downstream advantage over random spans or uniform training.

**Recorded prediction, scored.** We predicted gated distillation would land between random
and uniform. At floor 0 it landed below random; with a floor it ties both. Neither matches.

## Methodological record

- **The first gate check had no control.** `gate_report.json` recorded
  `required_in_top_frac: 0.125` beside `top_frac: 0.25` and nothing compared them.
  `inspect_gate.py` now reports a σ margin against a matched-budget control.
- **A cached score file is only valid in the environment that produced it.** The same gate
  read -5.14σ from a stale local file and +6.77σ when re-scored, with no code change.
- **We briefly withdrew a correct result.** Seeing the stale inspection number, we assumed the
  downstream run had trained on it too and marked Project B's comparison confounded. Re-running
  reproduced it exactly. A retraction needs the same evidence as a claim.

## Next

- Seeds on the floor-0.1 comparison: gated, random and uniform are within 0.02, so a claim of
  equality needs more than one run.
- Token-granularity arm: `scripts/run_skills.py --granularity token --stages
  score,distill,eval,report`, wired into [b1_skills.ipynb](../notebooks/b1_skills.ipynb) via
  `RUN_TOKEN_ARM`.

See [paper/draft_combined.md](../paper/draft_combined.md).
