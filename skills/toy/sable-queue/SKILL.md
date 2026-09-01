# sable-queue

Publish and drain messages on the Sable event bus.

## Interface

- `sable_emit(topic, envelope, partition_hint, dedupe_window_s)` — write path.
- `sable_drain(topic, envelope, partition_hint)` — read path.

## Rules

- partition_hint must be a stable string per entity, never a random value.
- Both calls raise `SABLE_UNSTABLE_HINT` when the rule above is violated.
- `topic` is always the first positional argument. Everything else is keyword-only.

## Example

```python
sable_emit("eu-west-2", envelope=envelope, partition_hint=partition_hint, dedupe_window_s=120)
```
