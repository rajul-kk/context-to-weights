import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir, read_json, set_seed, write_jsonl
from common.schema import Fact, Probe, Trajectory, Turn

SOURCE = "https://github.com/snap-research/locomo"


def _sessions(sample):
    conv = sample.get("conversation", sample)
    keys = sorted((k for k in conv if k.startswith("session_") and k[8:].isdigit()),
                  key=lambda k: int(k[8:]))
    for k in keys:
        turns = conv[k]
        if isinstance(turns, list):
            yield k, turns


def utterance_texts(sample):
    return [(utt.get("text") or utt.get("clean_text") or "").strip()
            for _, session in _sessions(sample) for utt in session if isinstance(utt, dict)]


def eligible_answers(sample, verbatim_only=False, min_answer_chars=1):
    texts = [t.lower() for t in utterance_texts(sample) if t]
    out = []
    for item in sample.get("qa", []) or []:
        ans = item.get("answer")
        if ans is None or not str(ans).strip():
            continue
        ans = str(ans).strip()
        if len(ans) < min_answer_chars:
            continue
        if verbatim_only and not any(ans.lower() in t for t in texts):
            continue
        out.append((str(item["question"]), ans))
    return out


def fold_split(n, n_folds, fold):
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold {fold} not in [0, {n_folds})")
    eval_idx = [i for i in range(n) if i % n_folds == fold]
    return eval_idx, [i for i in range(n) if i % n_folds != fold]


def build_trajectory(traj_id, sample, max_probes, verbatim_only=False, min_answer_chars=1):
    turns = []
    facts = []
    probes = []
    idx = 0

    answers = eligible_answers(sample, verbatim_only, min_answer_chars)[:max_probes]

    markers = [a for _, a in answers]

    for _, session in _sessions(sample):
        for utt in session:
            if not isinstance(utt, dict):
                continue
            text = (utt.get("text") or utt.get("clean_text") or "").strip()
            if not text:
                continue
            speaker = (utt.get("speaker") or "speaker").strip()
            hits = [m for m in markers if m and m.lower() in text.lower()]
            tags = ["fact", hits[0]] + hits if hits else ["filler"]
            turns.append(Turn(idx=idx, role="user" if idx % 2 == 0 else "assistant",
                              content=f"{speaker}: {text}", tags=tags))
            if hits:
                facts.append(Fact(key=hits[0], statement=text, value=hits[0], turn_idx=idx))
            idx += 1

    for q, a in answers:
        probes.append(Probe(fact_key=a, question=q, answer=a, aliases=[]))

    return Trajectory(
        traj_id=traj_id, domain="locomo", turns=turns, facts=facts, probes=probes,
        meta={"n_turns": len(turns), "n_probes": len(probes), "source": "locomo"},
    )


def main():
    ap = argparse.ArgumentParser(
        description=f"Build trajectories from a local LoCoMo json. Download it from {SOURCE}")
    ap.add_argument("--input", required=True, help="path to locomo10.json or equivalent")
    ap.add_argument("--out", default="artifacts/data/locomo")
    ap.add_argument("--eval-frac", type=float, default=0.34)
    ap.add_argument("--max-probes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbatim-only", action="store_true")
    ap.add_argument("--min-answer-chars", type=int, default=1)
    ap.add_argument("--n-folds", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"{src} not found. Download LoCoMo from {SOURCE} and pass --input.")

    set_seed(args.seed)
    rng = random.Random(args.seed)
    raw = read_json(src)
    samples = raw if isinstance(raw, list) else raw.get("data", [])
    if not samples:
        raise SystemExit("no samples found in the input file")

    trajectories = []
    for i, sample in enumerate(samples):
        traj = build_trajectory(f"locomo-{i:04d}", sample, args.max_probes, args.verbatim_only, args.min_answer_chars)
        if traj.probes and traj.turns:
            trajectories.append(traj)

    if not trajectories:
        raise SystemExit("no usable trajectories: no QA answer matched any utterance text")

    if args.n_folds:
        eval_idx, train_idx = fold_split(len(trajectories), args.n_folds, args.fold)
        splits = (("eval", [trajectories[i] for i in eval_idx]), ("train", [trajectories[i] for i in train_idx]))
    else:
        rng.shuffle(trajectories)
        n_eval = max(1, int(len(trajectories) * args.eval_frac))
        splits = (("eval", trajectories[:n_eval]), ("train", trajectories[n_eval:]))
    out = ensure_dir(args.out)
    for split, rows in splits:
        write_jsonl(out / f"{split}.jsonl", [t.to_dict() for t in rows])
        turns = sum(t.meta["n_turns"] for t in rows) / max(1, len(rows))
        probes = sum(len(t.probes) for t in rows)
        print(f"{split}: {len(rows)} trajectories, {turns:.0f} turns avg, {probes} probes")


if __name__ == "__main__":
    main()
