import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import (ensure_dir, load_config, parse_overrides, read_jsonl, set_seed,
                       write_json, write_jsonl)
from common.schema import Trajectory
from eval.retention import QA_SYSTEM, build_prompt
from sleep.examples import SleepExample
from sleep.lm import chat_text, load_backbone
from sleep.trainer import attach_lora, train_sleep_phase

VALUE_FORMS = [
    "In {project}, what is the {kp}?",
    "For {project}, which {kp} was chosen?",
    "{project}: {kp}?",
    "State the {kp} used by {project}.",
    "What did the {project} team settle on for {kp}?",
    "In {project}, give the {kp} in one phrase.",
    "Which {kp} does {project} use?",
    "I forgot the {kp} for {project}. What was it?",
    "Report the {kp} agreed for {project}.",
    "For the {project} project, the {kp} is what?",
    "Name the {kp} for {project}.",
    "{kp} for {project}?",
    "Look up the {kp} on {project}.",
    "{project} configuration, {kp}:",
]

STATEMENT_FORMS = [
    "Recall the {kp} decision for {project}.",
    "Summarise the {kp} for {project}.",
    "What is recorded about {kp} in {project}?",
    "Remind me what was agreed about {kp} in {project}.",
]

SENTENCE_FORM = "Confirm the {kp} for {project}."


def humanise(key):
    return key.replace("_", " ")


def cloze(statement, value):
    if value and value in statement:
        return statement.replace(value, "____", 1)
    return None


def forms_for(project, fact, n_forms):
    kp = humanise(fact["key"])
    out = []
    for t in VALUE_FORMS:
        out.append((t.format(project=project, kp=kp), fact["value"]))
    for t in STATEMENT_FORMS:
        out.append((t.format(project=project, kp=kp), fact["statement"]))
    out.append((SENTENCE_FORM.format(project=project, kp=kp),
                f"The {kp} for {project} is {fact['value']}."))
    blank = cloze(fact["statement"], fact["value"])
    if blank:
        out.append((f"Fill in the blank for {project}: {blank}", fact["value"]))
    return out[:n_forms]


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def build_examples(rows, tokenizer, n_forms):
    held_out = {norm(p["question"]) for r in rows for p in r["probes"]}
    examples, dropped, seen = [], 0, set()
    for row in rows:
        traj = Trajectory.from_dict(row)
        project = traj.meta.get("project") or traj.traj_id
        probed = {p.fact_key for p in traj.probes}
        for fact in traj.facts:
            if fact.key not in probed:
                continue
            for question, target in forms_for(project, fact.to_dict(), n_forms):
                key = (project, fact.key, norm(question))
                if norm(question) in held_out:
                    dropped += 1
                    continue
                if key in seen:
                    continue
                seen.add(key)
                examples.append(
                    SleepExample(
                        prompt=chat_text(tokenizer, QA_SYSTEM, build_prompt([], question)),
                        target=target,
                        kept=True,
                        source="oracle",
                        traj_id=traj.traj_id,
                        fact_key=fact.key,
                    )
                )
    return examples, dropped


def main():
    ap = argparse.ArgumentParser(
        description="Positive control: augment the ground-truth facts into many surface forms "
                    "and consolidate those. If this cannot beat the no-adapter baseline, the "
                    "setup cannot detect a consolidation effect at all.")
    ap.add_argument("--config", default="configs/kaggle_synthetic.yaml")
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-forms", type=int, default=20)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    if args.steps:
        cfg["sleep"]["steps"] = args.steps

    rows = read_jsonl(args.trajectories)
    model, tokenizer = load_backbone(cfg)
    examples, dropped = build_examples(rows, tokenizer, args.n_forms)
    if not examples:
        raise SystemExit("no augmented examples were built; check the trajectories file")

    facts = len({(e.traj_id, e.fact_key) for e in examples})
    print(f"{len(examples)} augmented examples over {facts} facts "
          f"({len(examples) / facts:.1f} forms each)")
    print(f"dropped {dropped} forms that matched a held-out eval question verbatim")

    peft_model = attach_lora(model, cfg)
    result = train_sleep_phase(peft_model, tokenizer, cfg, examples)

    run_dir = ensure_dir(args.run_dir)
    peft_model.save_pretrained(str(ensure_dir(run_dir / "adapter")))
    result.update({"n_facts": facts, "n_forms": args.n_forms, "dropped_held_out": dropped,
                   "model": cfg["model"]["base"]})
    write_json(run_dir / "result.json", result)
    write_jsonl(run_dir / "examples.jsonl", [e.to_dict() for e in examples])
    print(f"{result['steps']} steps, {result['gpu_seconds']:.1f}s -> {run_dir}/adapter")


if __name__ == "__main__":
    main()
