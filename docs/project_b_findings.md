# Project B results

1. **Does the with/without-context KL carry a usable importance signal?** Yes. The span gate
   selects required content at **+6.77σ** over a matched-budget control.
2. **Does gating distillation on it beat uniform distillation?** At `floor_weight: 0` it
   loses (0.510 vs 0.708). Fixing the floor to 0.1 flips this: across three seeds, gated
   distillation numerically leads uniform on all three and random-span on two of three, though
   the margin is not significant at this seed count (t(2)=2.78 vs uniform, 2.0 vs random).

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

| floor weight | seed | KL-gated | random-span | uniform |
|---|---|---|---|---|
| 0 | 0 | 0.510 | 0.583 | 0.708 |
| 0.1 | 0 | 0.729 | 0.729 | 0.708 |
| 0.1 | 1 | 0.823 | 0.750 | 0.729 |
| 0.1 | 2 | 0.802 | 0.719 | 0.698 |
| 0.3 | 0 | 0.708 | 0.656 | 0.708 |

In-group pass, n=96 each.

- **The zero floor was the defect.** A 0.1 floor lifts gated distillation by +0.219 (about
  3.2σ) at seed 0. Confirmed on two more seeds: floor 0.1 never scores below 0.708.
- **With more seeds, the gate looks ahead rather than tied.** Seed 0 was an exact tie with
  random-span (0.729 each); seeds 1 and 2 both put the gate ahead of random by 0.07-0.08 and
  ahead of uniform by 0.09-0.10. Paired across all three seeds: **kl_top - uniform mean
  +0.073, t(2) = 2.78**; **kl_top - random mean +0.052, t(2) = 2.0**. Neither clears the
  t(2) critical value of 4.30, so this is not yet a significant result, but the earlier
  "ties, adds nothing" reading was one seed away from "leads, not yet significant" — the
  honest statement is that three seeds point the same direction and need a fourth and fifth
  to settle it.
- `random - uniform` is +0.021 on all three seeds exactly, which is small enough to be a
  fixed rounding artifact of `n=96` rather than a real, reseedable effect; not interpreted
  further here.

**Recorded prediction, scored.** We predicted gated distillation would land between random
and uniform. At floor 0 it landed below both. With the floor fixed and pooled across three
seeds it nominally leads both, not significantly. Revise, don't discard, on new evidence.

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

## More seeds needed before this is a claim

n=3 seeds is what decided "tied" was actually "leads, not yet significant" above. Two more
seeds at floor 0.1 (about 45 min) would either confirm the lead or fold it back into noise.

See [paper/draft_combined.md](../paper/draft_combined.md).
