# harrowdb

Query the HarrowDB append-only ledger.

## Interface

- `harrow_append(ledger, record, actor, checkpoint)` — write path.
- `harrow_scan(ledger, record, actor)` — read path.

## Rules

- every append must carry an actor; scans without a checkpoint are refused.
- Both calls raise `HARROW_NO_ACTOR` when the rule above is violated.
- `ledger` is always the first positional argument. Everything else is keyword-only.

## Example

```python
harrow_append("edge", record=record, actor=actor, checkpoint=checkpoint)
```
