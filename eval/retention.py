import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import load_config, parse_overrides, read_jsonl, set_seed, write_json, write_jsonl
from eval.metrics import aggregate, answer_match
from sleep.lm import batch_generate, chat_text, load_backbone, token_ce

QA_SYSTEM = (
    "You answer questions about an earlier engineering conversation. "
    "Reply with the answer only, no explanation."
)


def build_prompt(context, question):
    body = "\n".join(context).strip()
    if body:
        return f"Retained notes from the conversation:\n{body}\n\nQuestion: {question}"
    return f"Question: {question}"


def evaluate(model, tokenizer, contexts, cfg, adapter_label="none", compute_ce=True):
    pairs = []
    index = []
    for row in contexts:
        for probe in row["probes"]:
            prompt = build_prompt(row["context"], probe["question"])
            pairs.append((QA_SYSTEM, prompt))
            index.append((row["traj_id"], probe, prompt))

    preds = batch_generate(
        model, tokenizer, pairs,
        max_new_tokens=cfg["eval"]["max_new_tokens"],
        batch_size=cfg["eval"]["batch_size"],
    )

    records = []
    for (traj_id, probe, prompt), pred in zip(index, preds):
        chat = chat_text(tokenizer, QA_SYSTEM, prompt)
        ce = token_ce(model, tokenizer, chat, probe["answer"], cfg["model"]["max_length"]) if compute_ce else []
        records.append(
            {
                "traj_id": traj_id,
                "adapter": adapter_label,
                "fact_key": probe["fact_key"],
                "question": probe["question"],
                "gold": probe["answer"],
                "prediction": pred,
                "correct": answer_match(pred, probe["answer"], probe.get("aliases")),
                "fact_in_context": probe["answer"].lower() in prompt.lower(),
                "prompt_tokens": len(tokenizer.encode(chat, add_special_tokens=False)),
                "ce": ce,
            }
        )
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", default="method")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-ce", action="store_true")
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    contexts = read_jsonl(args.contexts)
    if args.limit:
        contexts = contexts[: args.limit]

    model, tokenizer = load_backbone(cfg)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    records = evaluate(model, tokenizer, contexts, cfg, args.label, compute_ce=not args.no_ce)
    summary = aggregate(records)
    summary["label"] = args.label
    summary["model"] = cfg["model"]["base"]
    summary["contexts"] = args.contexts
    summary["adapter"] = args.adapter

    out = Path(args.out)
    write_jsonl(out.with_suffix(".records.jsonl"), records)
    write_json(out.with_suffix(".summary.json"), summary)
    for k, v in summary.items():
        print(f"{k:<24} {v}")
    if summary["n_evicted"] == 0:
        print("\nWARNING: no probe was evicted from context, so evicted_accuracy is undefined.\n"
              "The compactor kept every fact. Lower compaction.keep_frac or lengthen "
              "trajectories before comparing methods.")


if __name__ == "__main__":
    main()
