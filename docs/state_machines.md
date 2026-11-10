# Implicit state machines

An implicit state machine here is behaviour that depends on a hidden mode, a call-order
protocol or files that must agree, with no single place that names the states.

## Fixed

| Where | Hidden state | Failure | Now |
|---|---|---|---|
| `declare/elicit.py` `LAST_ROW` | global list, protocol clear, forward, read, clear | the probe appended on every forward, so `read` and `generate` in a run that also probes leaked one CPU tensor per layer per forward | `capturing()` context manager; the probe records only inside it, captures do not nest and close on exceptions |
| `declare/elicit.py` `KV_QUANT` | global dict written by `register_quant` | registering at 2 bits after 8 bits changed the behaviour of a model already loaded | `quant_attention_for(bits, group, scheme)` closes over its settings and registers under a name that encodes them |
| `declare/run.py` | attention implementation chosen by flag priority | `--kv-bits` with an attention mode was ignored while the summary recorded the bits | `declare/plan.py` resolves modes and flags into one `AttentionPlan`, rejects the conflict, and the summary records the plan |
| `sleep/checkpoint.py`, `sleep/loop.py` | `state.json`, `adapter/`, `mask_head.pt` must agree | a save interrupted after the adapter was removed left state without an adapter, and resume trained a fresh adapter at the old cursor; a missing or unexpected mask head was ignored | the adapter is staged and swapped, `state.json` is written last and atomically, each adapter carries a phase stamp, and `resume_run` refuses any disagreement |
| `sleep/lm.py` | `tokenizer.truncation_side` flipped and restored | an exception between the two left the tokenizer truncating on the left | `left_truncation()` restores it in a `finally` |
| `sleep/loop.py` | `sleep.mix_mode` string | any value other than `stratified` meant random | validated against `MIX_MODES` |
| `scripts/run_all.py`, `scripts/run_skills.py` | stage, method and policy names in comma lists | a typo selected nothing and the script exited cleanly | `choice_list` rejects unknown names |
| `common/io.py` | `write_json` truncated in place | an interrupted write left a half-written file, including `state.json` and gold references | written to a temporary file and replaced |

Tests are in `tests/test_state_machines.py` and `tests/test_attention_state.py`.

## Examined and left

- `tokenizer.chat_kwargs` is set once by `load_backbone` from the config and never changed, so it
  is configuration carried on the tokenizer, not a transition.
- `Compactor.last_decided_by` is assigned inside every `select` before the runner reads it.
- `ModelCompactor` counters only accumulate and are read after the run.
- `CompactionRunner.run` keeps `carry` and `recent` in local variables with one visible loop.
- `ReplayBuffer` exposes `state_dict` and `load_state_dict`.
- Notebook stage order is enforced by the preflight and `REQUIRE` checks in each notebook.

## Known gap

The random generator is not part of `state.json`, so a resumed sleep run draws different
examples than an uninterrupted one. The cursor, replay buffer and adapter are restored exactly.
