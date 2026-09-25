# Consolidation signals for LoRA

Three research arms asking the same question — *which free signal says what matters in
context?* — and testing whether each can drive consolidation into weights.

| | A | B | C |
|---|---|---|---|
| name | Compaction-Supervised Sleep Consolidation | [Context-Gap Distillation](docs/context_gap.md) | [Declarative Attention at small scale](docs/declarative.md) |
| signal | the compactor's keep/drop decision | with-vs-without-context KL divergence | which region the model says it must read |
| source | external, an agent already emits it | internal, two forward passes | external, the model states it |
| paper | [draft_combined.md](paper/draft_combined.md) §3 | §4 | §5 |

They share `common/`, `sleep/lm.py`, `sleep/trainer.py` and the config machinery.

**The finding that ties them together** ([paper/draft_combined.md](paper/draft_combined.md)):
signals the model is *asked to state* collapse at small scale — both Qwen sizes answer a
span-selection request with `0, 1, 2, ...` regardless of content, and SmolLM2-360M answers a
region-declaration request with `FOCUS: 0` every time. Signals *measured from behaviour*
survive: the same decision read off the logits reaches 2.56x salience lift at 1.5B and 2.17x
at 360M. Measure the model; do not ask it.

---

# Project A: Compaction-Supervised Sleep Consolidation

Every time an agent compacts its context it decides which spans stay verbatim. That decision
is a free salience label. A frozen compactor reads a per-span keep probability off its logits;
every `K` compaction events a **sleep phase** runs LoRA SFT on the kept spans plus a reservoir
replay buffer, and the adapter carries across phases. Pure SFT: no RL, no reward model.

| Tag | Description | Entry point |
|---|---|---|
| ours | LoRA SFT on compactor-kept spans | `sleep/loop.py --method compaction` |
| a | Uniform replay over all spans | `sleep/loop.py --method uniform` |
| b | Reflection/synthesis consolidation | `baselines/reflection.py`, then `--method reflection` |
| c | Cascading compaction, no weight update | `baselines/cascading.py` |
| d | Full-context ceiling | `baselines/full_context.py --mode full` |
| floor | No context | `baselines/full_context.py --mode none` |

**Result.** The keep decision is a strong salience signal (+15.85σ), yet uniform replay
recovers more evicted facts than compaction-supervised consolidation on all three synthetic
corpora (0.070 vs 0.010 mean; paired t(2)=1.81, not significant). The same mask
trained for abstention cuts hallucination on evicted facts by 72.8%.

```bash
pip install -r requirements.txt
python scripts/run_all.py --config configs/base.yaml
```

`configs/cpu_debug.yaml` runs the pipeline on a laptop (SmolLM2-360M, heuristic compactor).
Results: [docs/project_a_findings.md](docs/project_a_findings.md). Protocol and precondition
sweep: [docs/protocol.md](docs/protocol.md). Data: [docs/benchmarks.md](docs/benchmarks.md).
Notebooks: [docs/kaggle.md](docs/kaggle.md).

---

# Project B: Context-Gap Distillation

Internalising skill documents into LoRA, using the KL divergence between predictions with the
skill text in context (teacher) and without it (student) as both the importance signal and the
loss. Self-supervised: no RL, no judge.

```bash
python skills/generate_toy_skills.py
python scripts/run_skills.py --config configs/skill_base.yaml
```

Results: [docs/project_b_findings.md](docs/project_b_findings.md). Design:
[docs/context_gap.md](docs/context_gap.md).

---

# Project C: Declarative attention at small scale

Can a small model *state* which context region it needs, versus having it read off its logits
or its attention? `python declare/run.py --config configs/declare.yaml`. Results:
[docs/declarative.md](docs/declarative.md).

---

## Repository map

```
common/     config, IO, dataclasses shared by all projects
data/       A: synthetic generator, HotpotQA and LoCoMo loaders
compactor/  A: compactor backends and cascading runner
sleep/      A: consolidation loop; shared LM layer and LoRA trainer
baselines/  A: cascading, full/no context, reflection
skills/     B: toy skill specs and generator
kl_gate/    B: dual-forward-pass scoring and gate inspection
distill/    B: gate policies and weighted KL training
declare/    C: regions, elicitation modes, runner
eval/       retention, abstention, skill eval, reports and figures
configs/    per-project and per-environment configs
notebooks/  Kaggle notebooks: a0, a1 (A), b1 (B), c1, c2 (C)
scripts/    orchestrators, sweeps, Kaggle sync
docs/       findings, protocol, benchmarks, Kaggle workflow
paper/      combined workshop draft
```
