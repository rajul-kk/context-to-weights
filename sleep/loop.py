import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import (append_jsonl, ensure_dir, load_config, parse_overrides, read_jsonl,
                       set_seed, write_json)
from common.schema import CompactionEvent
from sleep.checkpoint import load_state, resume_adapter, save_phase
from sleep.examples import ReplayBuffer, from_compaction, from_reflection, from_uniform
from sleep.lm import load_backbone
from sleep.trainer import MaskHead, load_or_attach, train_sleep_phase

METHODS = ["compaction", "uniform", "reflection"]


def build_examples(method, events, rng, reflections, include_dropped):
    out = []
    for ev in events:
        if method == "compaction":
            out.extend(from_compaction(ev, include_dropped=include_dropped))
        elif method == "uniform":
            out.extend(from_uniform(ev, rng))
        elif method == "reflection":
            text = reflections.get((ev.traj_id, ev.event_id))
            if text:
                out.extend(from_reflection(ev, text))
    return out


def load_reflections(path):
    if not path:
        return {}
    return {(r["traj_id"], r["event_id"]): r["reflection"] for r in read_jsonl(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--method", default="compaction", choices=METHODS)
    ap.add_argument("--events", required=True)
    ap.add_argument("--val-events", default=None)
    ap.add_argument("--reflections", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--mask-head", action="store_true")
    ap.add_argument("--max-phases", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=96)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])

    use_replay = cfg["sleep"]["replay"]["enabled"] and not args.no_replay
    use_mask = cfg["sleep"]["mask_head"]["enabled"] or args.mask_head
    tag = args.method + ("+mask" if use_mask else "") + ("" if use_replay else "-noreplay")
    run_dir = ensure_dir(args.run_dir or Path(cfg["run_root"]) / f"sleep_{tag}")
    write_json(run_dir / "config.json", {"cfg": cfg, "method": args.method,
                                         "replay": use_replay, "mask_head": use_mask})

    events = [CompactionEvent.from_dict(d) for d in read_jsonl(args.events)]
    reflections = load_reflections(args.reflections)
    val_examples = []
    if args.val_events:
        val_events = [CompactionEvent.from_dict(d) for d in read_jsonl(args.val_events)]
        for ev in val_events:
            val_examples.extend(from_compaction(ev))
        if args.val_limit:
            rng.shuffle(val_examples)
            val_examples = val_examples[: args.val_limit]

    replay = ReplayBuffer(capacity=cfg["sleep"]["replay"]["capacity"], seed=cfg["seed"])
    cursor = 0
    phase = 0
    state = load_state(run_dir) if args.resume else None
    adapter = resume_adapter(run_dir) if args.resume else None
    if state:
        cursor = state["cursor"]
        phase = state["phase"]
        replay.load_state_dict(state["replay"])
        print(f"resuming at phase {phase}, event cursor {cursor}")

    model, tokenizer = load_backbone(cfg)
    model = load_or_attach(model, cfg, adapter)
    model.print_trainable_parameters()

    mask_head = None
    if use_mask:
        mask_head = MaskHead(model.config.hidden_size).to(next(model.parameters()).device)

    k = cfg["sleep"]["every_k_events"]
    metrics_path = run_dir / "metrics.jsonl"

    while cursor < len(events):
        window = events[cursor: cursor + k]
        cursor += len(window)
        train_examples = build_examples(args.method, window, rng, reflections,
                                        include_dropped=use_mask)
        if use_replay:
            train_examples = train_examples + replay.sample(cfg["sleep"]["replay"]["capacity"] // 2)
        if not train_examples:
            continue

        result = train_sleep_phase(model, tokenizer, cfg, train_examples, val_examples,
                                   mask_head, log_every=args.log_every)
        replay.add_all([e for e in build_examples(args.method, window, rng, reflections, False)])
        phase += 1

        record = {
            "phase": phase,
            "method": args.method,
            "cursor": cursor,
            "n_events_seen": cursor,
            "replay_size": len(replay.items),
            **{key: val for key, val in result.items() if key != "history"},
        }
        if result.get("history"):
            record["final"] = result["history"][-1]
        append_jsonl(metrics_path, record)
        print(
            f"phase {phase:>3} events {cursor:>4}/{len(events)} "
            f"examples {result.get('n_train_examples', 0):>4} "
            f"gpu_s {result.get('gpu_seconds', 0):.1f} "
            f"val_ce_median {record.get('final', {}).get('val_ce_median', float('nan')):.4f}"
        )

        save_phase(run_dir, phase, model,
                   {"phase": phase, "cursor": cursor, "method": args.method,
                    "replay": replay.state_dict()})

        if args.max_phases and phase >= args.max_phases:
            print("hit --max-phases, stopping")
            break

    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
