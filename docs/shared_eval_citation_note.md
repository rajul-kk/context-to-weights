# Where to cite "salient is not trainable" in skill-library's paper

For the owner of `D:\Work\skill-library` (`paper/paper.md` at commit `906d77a`). Nothing in that
repo was edited.

## What the Myrios result says

[paper/draft_combined.md](../paper/draft_combined.md) §5: a salience signal that clearly
carries information does not make a better consolidation target than uncurated coverage at the
same training budget. Compactor-gated LoRA consolidation loses to uniform replay on all five
synthetic corpora (0.017 vs 0.076 evicted facts recovered, paired t(4) = 3.10), because the
gate never shows the adapter whole fact types (§5.1). KL-gated skill distillation is level with
a random subset of the same size (t(4) = 0.74, §5.2). §5.3 states the common reading: coverage,
not selection, decides what the weights learn.

## Where it fits in skill-library's paper

1. **§4.2 Ablations, "Allocation under a scarce buffer" (main place).** skill-library finds that
   tier-balanced allocation beats recency and round-robin when the buffer is below one example
   per skill, and that retention saturates once the buffer covers the library. Both are coverage
   effects. Suggested sentence after "beats rotating through skills": *"This agrees with [Myrios],
   where replay selected by a salience signal lost to uniform replay at equal budget because the
   selection left whole fact types out of training: what a replay buffer covers matters more than
   how cleverly it is chosen."*
2. **§4.2, "Replay budget".** One clause after "it needs to cover it": *"(the same coverage
   effect reported for fact consolidation in [Myrios])."*
3. **§2.3 A night, last paragraph.** Pre-empts the reviewer question "why not prioritise replay
   by salience or loss?": *"We do not weight replay by salience: in a closely related setting,
   salience-gated consolidation was worse than uniform replay [Myrios]."*
4. **§5 Related work, "Continual learning".** One sentence placing both projects together as
   evidence against selective replay at small scale.
5. **§6 Limitations.** Say that salience-weighted replay was not tested here and that the
   expectation it would not help rests on [Myrios], a different task (fact recall and tool-use
   skill documents, Qwen2.5 0.5B to 1.5B) rather than skill programs.

## Cautions

- Cite it as consistent evidence, not proof: the tasks, models and signals differ, and the
  skill-library CPU numbers come from a simulated interference model, not gradients.
- The Myrios draft is an anonymous TMLR submission. If skill-library goes to a double-blind
  venue, cite it in the third person (or as "anonymous, under review") and do not say both works
  share an author. Add the entry to skill-library's references once the Myrios preprint has an
  arXiv identifier; until then cite the tagged Myrios repository.
- Quote numbers only as they appear in the Myrios paper, which a Myrios test keeps equal to
  `paper/figures/data.json`.
- When the library-size sweep writes `myrios_eval` result files, the comparison sentence in point
  1 can cite matched statistics (paired t and bootstrap CI over seeds) computed the same way as
  the Myrios figures; see [shared_eval_harness.md](shared_eval_harness.md).
