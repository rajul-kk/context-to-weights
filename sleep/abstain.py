import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import (ensure_dir, load_config, parse_overrides, read_jsonl, set_seed,
                       write_json, write_jsonl)
from eval.retention import QA_SYSTEM, build_prompt
from sleep.examples import SleepExample
from sleep.lm import chat_text, load_backbone
from sleep.trainer import attach_lora, train_sleep_phase

ABSTAIN_TEXT = "I do not have that information in the retained notes."


def survives(context, probe):
    return probe["answer"].lower() in " ".join(context).lower()


def build_examples(contexts, tokenizer, abstain_text=ABSTAIN_TEXT):
    out = []
    for row in contexts:
        for probe in row["probes"]:
            kept = survives(row["context"], probe)
            prompt = chat_text(tokenizer, QA_SYSTEM,
                               build_prompt(row["context"], probe["question"]))
            out.append(
                SleepExample(
                    prompt=prompt,
                    target=probe["answer"] if kept else abstain_text,
                    kept=True,
                    source="answer" if kept else "abstain",
                    traj_id=row["traj_id"],
                    fact_key=probe["fact_key"],
                )
            )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Train the compaction mask as an abstention signal rather than a "
                    "recall signal: answer when the fact survived, refuse when it did not.")
    ap.add_argument("--config", default="configs/kaggle_synthetic.yaml")
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--val-contexts", default=None)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    if args.steps:
        cfg["sleep"]["steps"] = args.steps

    model, tokenizer = load_backbone(cfg)
    train = build_examples(read_jsonl(args.contexts), tokenizer)
    val = build_examples(read_jsonl(args.val_contexts), tokenizer) if args.val_contexts else None

    n_abstain = sum(1 for e in train if e.source == "abstain")
    print(f"{len(train)} examples: {n_abstain} abstain, {len(train) - n_abstain} answer")
    if not n_abstain or n_abstain == len(train):
        raise SystemExit(
            "the training split is one-sided, so the adapter would learn a constant policy. "
            "Check that compaction actually evicts some probed facts.")

    peft_model = attach_lora(model, cfg)
    result = train_sleep_phase(peft_model, tokenizer, cfg, train, val)

    run_dir = ensure_dir(args.run_dir)
    peft_model.save_pretrained(str(ensure_dir(run_dir / "adapter")))
    result["n_abstain"] = n_abstain
    result["n_answer"] = len(train) - n_abstain
    result["abstain_text"] = ABSTAIN_TEXT
    result["model"] = cfg["model"]["base"]
    write_json(run_dir / "result.json", result)
    write_jsonl(run_dir / "examples.jsonl", [e.to_dict() for e in train])
    print(f"{result['steps']} steps, {result['gpu_seconds']:.1f}s -> {run_dir}/adapter")


if __name__ == "__main__":
    main()
