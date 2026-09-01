# obsidian-flags

Read and flip Obsidian feature flags.

## Interface

- `obs_flip(flag, state, cohort, reason)` — write path.
- `obs_read(flag, state, cohort)` — read path.

## Rules

- every flip needs a reason string; Obsidian writes it to the audit trail.
- Both calls raise `OBS_NO_REASON` when the rule above is violated.
- `flag` is always the first positional argument. Everything else is keyword-only.

## Example

```python
obs_flip("tier-one", state="on", cohort=cohort, reason=reason)
```
