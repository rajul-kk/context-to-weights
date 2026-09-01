import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import (ensure_dir, load_config, parse_overrides, set_seed, write_json,
                       write_jsonl)
from kl_gate.skillset import SKILL_SYSTEM, load_all, with_skill, without_skill
from sleep.lm import batch_generate, chat_text, load_backbone

DOC_MODES = ["none", "correct", "mismatched"]


def build_prompt(skill, other_doc, query, doc_mode):
    if doc_mode == "correct":
        return with_skill(skill["doc"], query)
    if doc_mode == "mismatched":
        return with_skill(other_doc, query)
    return without_skill(query)


def passed(prediction, required):
    return all(tok in prediction for tok in required)


def eval_arm(model, tokenizer, skills, cfg, doc_mode, label, group_of, adapter_group,
             limit_tasks=0):
    pairs = []
    index = []
    for si, skill in enumerate(skills):
        other = skills[(si + 1) % len(skills)]["doc"]
        tasks = skill["tasks"][:limit_tasks] if limit_tasks else skill["tasks"]
        for task in tasks:
            prompt = build_prompt(skill, other, task["query"], doc_mode)
            pairs.append((SKILL_SYSTEM, prompt))
            index.append((skill, task, prompt))

    preds = batch_generate(model, tokenizer, pairs,
                           max_new_tokens=cfg["eval"]["max_new_tokens"],
                           batch_size=cfg["eval"]["batch_size"])

    records = []
    for (skill, task, prompt), pred in zip(index, preds):
        chat = chat_text(tokenizer, SKILL_SYSTEM, prompt)
        group = group_of.get(skill["name"], "all")
        records.append(
            {
                "label": label,
                "doc_mode": doc_mode,
                "skill": skill["name"],
                "group": group,
                "adapter_group": adapter_group,
                "in_group": adapter_group is None or group == adapter_group,
                "kind": task["kind"],
                "required": task["required"],
                "prediction": pred,
                "passed": passed(pred, task["required"]),
                "prompt_tokens": len(tokenizer.encode(chat, add_special_tokens=False)),
            }
        )
    return records


def summarize(records, label):
    def rate(rows):
        return sum(int(r["passed"]) for r in rows) / len(rows) if rows else float("nan")

    in_group = [r for r in records if r["in_group"]]
    out_group = [r for r in records if not r["in_group"]]
    return {
        "label": label,
        "n": len(records),
        "pass_rate": rate(records),
        "pass_rate_in_group": rate(in_group),
        "pass_rate_out_group": rate(out_group),
        "n_out_group": len(out_group),
        "mean_prompt_tokens": (sum(r["prompt_tokens"] for r in records) / len(records)
                               if records else 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/skill_base.yaml")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--doc-mode", default="none", choices=DOC_MODES)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    root = cfg["skills"]["root"]
    skills = load_all(root)
    group_of = {s["name"]: (s["category"] if cfg["skills"]["grouping"] == "category" else s["name"])
                for s in skills}

    model, tokenizer = load_backbone(cfg)
    records = []

    if args.run_dir is None:
        records = eval_arm(model, tokenizer, skills, cfg, args.doc_mode, args.label,
                           group_of, None, args.limit_tasks)
    else:
        from peft import PeftModel

        run_dir = Path(args.run_dir)
        adapters = sorted(p.parent.name for p in run_dir.glob("*/adapter/adapter_config.json"))
        if not adapters:
            raise SystemExit(f"no adapters under {run_dir}")
        for group in adapters:
            peft_model = PeftModel.from_pretrained(model, str(run_dir / group / "adapter"))
            peft_model.eval()
            records.extend(eval_arm(peft_model, tokenizer, skills, cfg, args.doc_mode,
                                    args.label, group_of, group, args.limit_tasks))
            model = peft_model.unload()

    summary = summarize(records, args.label)
    summary["doc_mode"] = args.doc_mode
    summary["run_dir"] = args.run_dir

    out = Path(args.out)
    ensure_dir(out.parent)
    write_jsonl(out.with_suffix(".records.jsonl"), records)
    write_json(out.with_suffix(".summary.json"), summary)
    for k, v in summary.items():
        print(f"{k:<24} {v}")


if __name__ == "__main__":
    main()
