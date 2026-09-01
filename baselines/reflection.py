import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import load_config, read_jsonl, set_seed, write_jsonl
from common.schema import CompactionEvent
from sleep.lm import batch_generate, load_backbone

REFLECTION_SYSTEM = (
    "You are reviewing a slice of an engineering conversation at the end of a working session."
)

REFLECTION_INSTRUCTION = """Write what you learned from this slice, as durable notes for your future self.
State concrete decisions, names, versions, identifiers and limits. Do not speculate.
Three sentences at most.

Conversation slice:
{body}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    events = [CompactionEvent.from_dict(d) for d in read_jsonl(args.events)]
    model, tokenizer = load_backbone(cfg)

    pairs = []
    for ev in events:
        body = "\n".join(s.text for s in ev.spans)
        pairs.append((REFLECTION_SYSTEM, REFLECTION_INSTRUCTION.format(body=body)))

    texts = batch_generate(model, tokenizer, pairs,
                           max_new_tokens=args.max_new_tokens,
                           batch_size=cfg["eval"]["batch_size"])

    rows = [
        {"traj_id": ev.traj_id, "event_id": ev.event_id, "reflection": t.strip()}
        for ev, t in zip(events, texts)
    ]
    write_jsonl(args.out, rows)
    print(f"{len(rows)} reflections -> {args.out}")


if __name__ == "__main__":
    main()
