# Benchmarks

## Status

| Benchmark | Project | State | Loader |
|---|---|---|---|
| Synthetic dev conversations | A | **working, all results so far** | `data/generate_synthetic.py` |
| HotpotQA (distractor) | A | **working, pipeline verified** | `data/load_hotpotqa.py` |
| LoCoMo | A | loader written, **needs a manual download** | `data/load_locomo.py` |
| Toy tool-use skills | B | **working, gate verified** | `skills/generate_toy_skills.py` |
| SWE-Skills-Bench | B | **not implemented** | - |

Every number reported anywhere in this repo so far comes from the **synthetic** set. Treat
the other rows as plumbing that runs, not as evidence.

## Synthetic dev conversations

CPU-generated multi-turn software-engineering dialogues. Six load-bearing facts per
trajectory — datastore choices, header names, rate limits, queue names, flag names, version
pins — planted in an early window, then buried under topically plausible filler. Each fact
carries a held-out probe with a gold answer.

Defaults: 48 train / 24 eval trajectories, 120 turns, 6 facts each.

The point of this set is control, not realism. Fact placement, compaction pressure and the
fact/filler ratio are all dials, which is what makes it useful for isolating whether the
compaction signal specifically helps.

**Known weakness, and what it actually is.** Facts are introduced with the phrase
`One thing to lock in:`. An unmarked variant drops it:

```bash
python data/generate_synthetic.py --unmarked --out artifacts/data/synthetic_hard
```

Removing the marker turns out to change almost nothing for the heuristic compactor: 4.43x
marked against 4.38x unmarked. The marker was never what it keyed on.

The real leak is vocabulary overlap between the heuristic's cue list and the fact templates.
Eight of the fourteen cues in `compactor/base.py` — `settled on`, `fixed at`, `agreed`,
`pins`, `owns`, `rejected`, `kill switch`, `nowhere else` — appear verbatim in
`data/banks.py`. The heuristic was written against the same vocabulary the generator emits,
so its lift measures that overlap and nothing else. This is why the heuristic backend is
debug-only and never reported.

The model compactor is a different story, and a worse one. Qwen2.5-1.5B with logit scoring
scores 2.56x on the marked set and **1.10x on the unmarked set**, where it no longer beats
its positional control. The heuristic ignored the marker and keyed on vocabulary; the model
ignored the vocabulary and keyed on the marker. Two backends, two different shortcuts.

Report both, and read the unmarked set as a **floor** rather than a neutral test. Facts and
filler there are generated from the same template bank in the same register, so they are
close to stylistically indistinguishable once the marker is gone — harsher than any real
conversation. HotpotQA, where the compactor reaches 1.61x against a 0.98x control, is the
more representative number.

## HotpotQA

```bash
python data/load_hotpotqa.py --n-train 32 --n-eval 16 --per-trajectory 5
python baselines/cascading.py --config configs/hotpotqa.yaml --split both
```

Multi-hop QA repackaged as a retention task. Each trajectory bundles several HotpotQA
examples: supporting paragraphs are scattered at random positions through the first
`--early-frac` of the trajectory, distractor paragraphs fill the rest, and each example's
question becomes a probe. Roughly 80 turns and 4 probes per trajectory at the defaults.

**The interleaving matters.** A first version placed every supporting paragraph in a
contiguous block at the front. That made position almost perfectly predictive: the
positional control scored 2.44x, higher than the compactor's own 1.76x, so the benchmark
was measuring layout rather than judgment. Facts still land early — they have to, or
compaction never evicts them — but scattered rather than stacked. Always read the positional
control that `eval/span_report.py` prints beside the lift; if the control is high, the
dataset construction is doing the work.

Fact spans are labelled from HotpotQA's **sentence-level** `supporting_facts`, not from
whether the span happens to contain the gold answer string. Multi-hop answers frequently do
not appear verbatim in every supporting sentence, so answer-substring labelling would
undercount badly.

Use `configs/hotpotqa.yaml`, not `configs/base.yaml`. HotpotQA paragraphs are long, and
under the default 1024-token budget the recent-turns window alone exceeds it — compaction
then re-triggers every turn and reduces nothing (103 events across 2 trajectories, ratio
0.036). The tuned config raises the budget to 3072 and drops `keep_recent_turns` to 4, which
gives 2 events per trajectory at ratio 0.61. `baselines/cascading.py` warns when a
configuration falls into the degenerate regime.

## LoCoMo

Not bundled — download `locomo10.json` from https://github.com/snap-research/locomo and
point the loader at it:

```bash
python data/load_locomo.py --input /path/to/locomo10.json
```

Sessions are flattened into one long trajectory and the dataset's own QA pairs become the
probes. An utterance is labelled as carrying a fact when it contains a gold answer string.

**Caveat worth checking before relying on this.** That answer-substring labelling is weaker
here than for HotpotQA, because LoCoMo answers are often paraphrases or aggregations rather
than spans lifted from the dialogue. The loader drops any trajectory where no QA answer
matched any utterance, so a low trajectory count after loading is the signal that this is
happening. Verify the fact/filler split by eye with `eval/span_report.py` before trusting a
salience-lift number from LoCoMo.

## Toy tool-use skills

Eight synthetic skills across three categories, each an invented API with a `SKILL.md`
document, nine demonstrations and twelve held-out tasks. The APIs are fictional so the base
model cannot know them, which is exactly what makes the context gap large on identifiers and
small on prose — see [kl_gate_check.md](kl_gate_check.md).

## SWE-Skills-Bench

Not implemented. Skill-to-LoRA (arXiv:2606.16769) reports on it, so a direct comparison
needs either their released artifacts or a reconstruction from their recipe. The toy skills
exist to debug the KL gate mechanism first; this is the follow-on once the gate is shown to
beat the random-span control.

## What to run first

**HotpotQA is now the primary set for the headline claim**, because it is the only one where
the compaction signal survives with the positional control at chance and no marker or shared
vocabulary to exploit (1.61x against 0.98x).

The synthetic sets are the mechanism-isolation tools around it: the marked variant for
debugging and for the compaction-ratio sweep, the unmarked variant as the floor case showing
what happens when salient content is stylistically identical to filler.
