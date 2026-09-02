# Consolidation signals for LoRA

Two research prototypes asking the same question — *what should a consolidation pass
internalise into weights?* — and answering it with two different free supervision signals.

| | Project A | Project B |
|---|---|---|
| name | Compaction-Supervised Sleep Consolidation | [Context-Gap Distillation](docs/context_gap.md) |
| signal | the compactor's keep/drop decision | with-vs-without-context KL divergence |
| source | external, an agent already emits it | internal, two forward passes of one backbone |
| loss | cross-entropy on kept spans | importance-weighted KL distillation |
| paper | [paper/draft.md](paper/draft.md) | [paper/draft_b.md](paper/draft_b.md) |

They share `common/`, `sleep/lm.py`, `sleep/trainer.py` and the config machinery. Project A
is the reference implementation; Project B was built on top of it.

---

# Project A: Compaction-Supervised Sleep Consolidation

Using a context-compactor's keep/drop decision as a free supervised signal for what a LoRA
"sleep" pass should internalise into weights. Pure SFT — no RL, no reward model, no LLM judge.

## Idea

Every time an agent compacts its context it already produces a decision: this span stays
verbatim, that span gets dropped or summarised. That decision is a salience label, it is
produced for free, and nobody trains on it. This project treats it as the target of a
periodic offline consolidation pass:

1. Run a trajectory until the context budget trips.
2. A frozen compactor picks spans to keep, by reading a per-span keep probability off its
   own logits rather than generating a list.
3. Log `(span_text, kept: bool)` for every span — the free label.
4. Every `K` compaction events, run a **sleep phase**: LoRA SFT with cross-entropy on the
   kept spans, plus a small reservoir replay buffer of previously-kept spans.
5. Keep the adapter across phases, so consolidation is cumulative.

An auxiliary variant adds a binary mask-prediction head that learns to predict the
keep/drop decision itself, testing whether learning salience helps beyond training on
salient content.

## What is new here

| Prior work | What it does | How this differs |
|---|---|---|
| Beyond Inference-Only Deployment (2605.24657) | Nightly LoRA consolidation on *reflected/synthesised* facts written by an LLM | We consolidate on the compactor's raw keep/drop decision — no synthesis step, no second generation pass |
| SCoL (2605.07076) | Learns *where* to write consolidated knowledge via meta-RL | We answer the *what/when* question, and as ordinary cross-entropy rather than RL |
| What to Keep, What to Forget (2607.08032) | Frames compaction as rate-distortion | Proposes no downstream consolidation; we supply one and use their signal as the label |
| CompactionRL (2607.05378) | Trains the compactor itself with RL | We leave the compactor frozen and train the *backbone* from its decisions |

## Methods compared

| Tag | Description | Entry point |
|---|---|---|
| ours | LoRA SFT on compactor-kept spans | `sleep/loop.py --method compaction` |
| a | Uniform/reservoir replay over *all* spans | `sleep/loop.py --method uniform` |
| b | Reflection/synthesis consolidation | `baselines/reflection.py` then `--method reflection` |
| c | Cascading compaction, no weight update | `baselines/cascading.py` |
| d | Full-context ceiling | `baselines/full_context.py --mode full` |
| floor | No context at all | `baselines/full_context.py --mode none` |

## Precondition

The method only works if the compactor's keep/drop decision correlates with what a later
question needs. Measure that first:

```bash
python scripts/check_compactor.py --config configs/kaggle.yaml
```

It reports **salience lift** — fact-span keep rate over filler-span keep rate — against a
positional control. The compactor must beat the control, not merely 1.0x.

Measured, and the result is mixed. Only **Qwen2.5-1.5B with the `scoring` backend** clears
the control, at **2.56x** against 1.68x — but only on the synthetic set whose facts are
introduced by an explicit marker phrase. On the unmarked variant the same compactor drops to
**1.10x** and fails its control.

Two separate shortcuts were hiding here: the heuristic backend keyed on cue vocabulary it
shared with the generator, the model backend keyed on the marker phrase. Neither was visible
until tested against a set that removed them.

Every configuration that asks a model to *generate* a ranked index list lands at chance at
every scale — both Qwen sizes answer `0, 1, 2, ...` regardless of content. Read the keep
decision off the logits; do not ask for it. Full table, and the three retracted measurements
that preceded it, in [docs/protocol.md](docs/protocol.md).

The compactor need not be the consolidation target. The event log is plain text, so run
compaction under a larger model and consolidate into a smaller one:

```bash
python baselines/cascading.py --split both --set model.base=Qwen/Qwen2.5-1.5B-Instruct
python sleep/loop.py --method compaction \
  --events artifacts/runs/cascading/train_events.jsonl \
  --set model.base=Qwen/Qwen2.5-0.5B-Instruct
```

## Metrics

- **Retention accuracy** on held-out probes about *early* trajectory content, asked after
  that content has been compacted away.
- **Median per-token validation cross-entropy** — the trustworthy signal. Mean CE is logged
  too, but per Beyond Inference-Only median CE tracks judged accuracy at r=+0.99 while mean
  CE moves the wrong way (r=-0.51). Read the median.
- **Inference token cost** — prompt tokens needed to answer post-compaction.
- **Consolidation cost** — GPU-seconds per sleep phase, per method.

## Quick start

```bash
pip install -r requirements.txt

python data/generate_synthetic.py --n-train 48 --n-eval 24
python baselines/cascading.py --config configs/base.yaml --split both
python eval/span_report.py --events artifacts/runs/cascading/train_events.jsonl

python sleep/loop.py --config configs/base.yaml --method compaction \
  --events artifacts/runs/cascading/train_events.jsonl \
  --val-events artifacts/runs/cascading/eval_events.jsonl

python eval/retention.py --config configs/base.yaml \
  --contexts artifacts/runs/cascading/eval_contexts.jsonl \
  --adapter artifacts/runs/sleep_compaction/latest/adapter \
  --label ours --out artifacts/runs/report/ours
```

`configs/cpu_debug.yaml` swaps in SmolLM2-360M, fp32, a heuristic compactor and 20 training
steps so the whole pipeline runs on a laptop without a GPU.

See [docs/protocol.md](docs/protocol.md) for the experiment protocol,
[docs/benchmarks.md](docs/benchmarks.md) for what data exists and what does not, and
[docs/kaggle.md](docs/kaggle.md) for the notebooks and session/resume workflow.

---

# Project B: Context-Gap Distillation

Internalising skill documents into LoRA by using the KL divergence between the model's
predictions with the skill text in context (teacher) and without it (student) as both the
importance signal and the training loss. Self-supervised — no RL, no judge.

```bash
python skills/generate_toy_skills.py

python kl_gate/score.py --config configs/skill_base.yaml \
  --granularity span --out artifacts/runs_skill/scores_span.jsonl
python kl_gate/inspect_gate.py --scores artifacts/runs_skill/scores_span.jsonl

python scripts/run_skills.py --config configs/skill_base.yaml
python scripts/sweep_kl.py --config configs/skill_base.yaml
```

The gate is verified by hand before anything trains on it —
[docs/kl_gate_check.md](docs/kl_gate_check.md) records that check. Full design in
[docs/context_gap.md](docs/context_gap.md).

---

## Repository map

```
common/     config, IO, dataclasses shared by both projects
data/       A: synthetic generator (CPU-only), HotpotQA and LoCoMo loaders
compactor/  A: fixed-prompt compactor and cascading runner
sleep/      A: consolidation loop; shared LM layer and LoRA trainer
skills/     B: toy skill specs and generator (CPU-only)
kl_gate/    B: dual-forward-pass scoring and manual inspection
distill/    B: gate policies and weighted KL training
baselines/  A: cascading, full/no context, reflection
eval/       both: retention, skill eval, reports and figures
configs/    base and cpu_debug configs for both projects
notebooks/  Kaggle GPU notebooks: a0 precondition, a1 main, b1 skills
scripts/    orchestrators and ablation sweeps
docs/       protocol, Kaggle workflow, gate check, results
paper/      workshop drafts
```
