# quarrybuild

Drive the Quarry incremental build system.

## Interface

- `quarry_target(label, inputs, toolchain, sandbox)` — write path.
- `quarry_explain(label, inputs, toolchain)` — read path.

## Rules

- sandbox must be strict for anything published; loose sandboxes are local-only.
- Both calls raise `QUARRY_LOOSE_PUBLISH` when the rule above is violated.
- `label` is always the first positional argument. Everything else is keyword-only.

## Example

```python
quarry_target("edge", inputs=["admin"], toolchain=toolchain, sandbox="loose")
```
