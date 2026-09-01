# Context-Gap Distillation

Internalising a skill document into LoRA weights, using the KL divergence between the
model's predictions *with* the skill text in context and *without* it as both the importance
signal and the training loss. Self-supervised: no RL, no judge, no second model.

## Mechanism

For a skill (a `SKILL.md` document plus worked demonstrations), each demonstration response
is scored twice through the same frozen backbone:

- **teacher** — skill document present in context
- **student reference** — skill document absent

Per response token, `KL(teacher || student_ref)` measures how much the skill document
changes what the model would predict. High KL means the content is behaviourally
load-bearing. Low KL means the model would have produced it anyway.

Training then distils the teacher into a LoRA student that does *not* see the skill
document, weighting the loss by that importance score. At inference the adapted model
reproduces with-skill behaviour with the document out of the prompt.

## Positioning

| Prior work | Signal | How this differs |
|---|---|---|
| Skill-to-LoRA (2606.16769) | none — uniform over all content | We gate on context-gap KL; S2L uniform is our baseline (b) |
| ThinkSwitch (2606.01080) | none — reasoning-trace removal | No importance gate at all |
| PEAM (2605.27762) | "parameterization-worthiness" over episodic trajectory pairs | A different gating signal, on trajectories not skill documents |
| Titans (2501.00663) | surprise as gradient of an associative-memory loss w.r.t. input | Gradient-based, applied to test-time memory writing; ours is KL-based and offline |

The contribution is narrow: no prior work gates skill distillation on a self-supervised
with-vs-without-context KL divergence. It is a distinct realisation of a "surprise"-style
importance signal, computed from two forward passes of one frozen backbone.

## Layout

```
skills/     toy skill specs, SKILL.md + demos + held-out tasks generator (CPU-only)
kl_gate/    dual-forward-pass scoring, span aggregation, manual inspection CLI
distill/    gate policies and weighted KL LoRA training
eval/       skill_eval.py, skill_report.py
configs/    skill_base, skill_cpu_debug, skill_tinyllama
```

Shared with Project A: `common/`, `sleep/lm.py` (backbone, chat templating, generation),
`sleep/trainer.py` (`attach_lora`), and the config/override machinery.

## Adapter granularity

**One LoRA per skill category**, not per skill and not one shared adapter. Per-skill
adapters make cross-skill forgetting unmeasurable by construction; a single shared adapter
across all eight skills confounds the gate with multi-skill interference on a first pass.
Categories (`storage`, `network`, `tooling`) sit in between: two to three related skills per
adapter, so interference is measurable within a group and comparable across arms. Set by
`skills.grouping` in the config.

## Baselines

| Tag | Description | Command |
|---|---|---|
| (a) | full skill text in prompt, no adapter | `eval/skill_eval.py --doc-mode correct` |
| (b) | S2L-style uniform distillation | `distill/train.py --policy uniform` |
| (c) | no skill, no adapter | `eval/skill_eval.py --doc-mode none` |
| (d) | random-span control | `distill/train.py --policy random` |
| ours | KL-gated distillation | `distill/train.py --policy kl_top` |

Control (d) is the one that has to be right. It trains on the same *number of active tokens*
as the gate selects — not merely the same number of spans, since high-KL spans run longer
than average. Any gap between (d) and ours is the gate doing something smarter than training
on less data.

## Metrics

- **Task pass rate** on held-out tasks per skill: every required string (API function name,
  error code, first positional argument) must appear in the output.
- **Runtime tokens saved** — mean prompt tokens with the document internalised vs in-prompt.
- **Forgetting / interference** — pass rate on skills *outside* the adapter's own category,
  reported as `pass_rate_out_group` beside `pass_rate_in_group`.
- **Retrieval-mismatch robustness** — pass rate when the retrieved document in context
  belongs to a different skill (`--doc-mode mismatched`). Arm (a) should degrade sharply;
  an internalised model should not.

## Ablations

| Ablation | How |
|---|---|
| KL-threshold sweep | `scripts/sweep_kl.py --fracs 0.1,0.25,0.5` |
| token vs span gating | `scripts/sweep_kl.py --granularities span,token` |

## Compute

Two forward passes per training step — teacher with the document, student without — so a
step costs roughly twice a plain SFT step. Demonstrations are kept to a few hundred response
tokens to hold that down. KL scoring is a one-off pass over all demonstrations and is cached
to `scores_<granularity>.jsonl`; the sweep reuses it.

| Stage | Cost |
|---|---|
| skill generation | CPU, seconds |
| KL scoring, 8 skills x 9 demos | ~0.2 GPU-h |
| one policy, 3 category adapters | ~1.2 GPU-h |
| one eval arm | ~0.3 GPU-h |
| main table | ~5 GPU-h |
| sweeps + second model | ~7 GPU-h |

Each category adapter is self-contained, so this project is more session-choppable than
Project A. `distill/train.py --resume` skips groups whose adapter already exists.
