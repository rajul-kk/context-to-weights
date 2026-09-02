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

The unmarked variant is still the better set for the write-up: it is closer to how facts
actually appear in a working conversation, and it is where the *model* compactor's lift
should be measured, since the model has no such vocabulary overlap to exploit.

## HotpotQA

```bash
python data/load_hotpotqa.py --n-train 32 --n-eval 16 --per-trajectory 5
python baselines/cascading.py --config configs/hotpotqa.yaml --split both
```

Multi-hop QA repackaged as a retention task. Each trajectory bundles several HotpotQA
examples: supporting paragraphs are placed early, distractor paragraphs are shuffled through
the rest as filler, and each example's question becomes a probe. Roughly 80 turns and 4
probes per trajectory at the defaults.

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

The synthetic set is the one that isolates the mechanism, so headline results should come
from it. HotpotQA is the external-validity check — if compaction-supervised consolidation
only works on conversations whose facts are marked with a stock phrase, that is worth
knowing, and HotpotQA is where it would show up.
