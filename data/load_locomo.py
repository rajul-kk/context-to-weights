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


def build_trajectory(traj_id, sample, max_probes):
    turns = []
    facts = []
    probes = []
    idx = 0

    qa = sample.get("qa", []) or []
    answers = []
    for item in qa[:max_probes]:
        ans = item.get("answer")
        if ans is None or not str(ans).strip():
            continue
        answers.append((str(item["question"]), str(ans)))

    markers = [a for _, a in answers]

    for _, session in _sessions(sample):
        for utt in session:
            if not isinstance(utt, dict):
                continue
            text = (utt.get("text") or utt.get("clean_text") or "").strip()
            if not text:
                continue
            speaker = (utt.get("speaker") or "speaker").strip()
            hits = [m for m in markers if m and m in text]
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
        traj = build_trajectory(f"locomo-{i:04d}", sample, args.max_probes)
        if traj.probes and traj.turns:
            trajectories.append(traj)

    if not trajectories:
        raise SystemExit("no usable trajectories: no QA answer matched any utterance text")

    rng.shuffle(trajectories)
    n_eval = max(1, int(len(trajectories) * args.eval_frac))
    out = ensure_dir(args.out)
    for split, rows in (("eval", trajectories[:n_eval]), ("train", trajectories[n_eval:])):
        write_jsonl(out / f"{split}.jsonl", [t.to_dict() for t in rows])
        turns = sum(t.meta["n_turns"] for t in rows) / max(1, len(rows))
        probes = sum(len(t.probes) for t in rows)
        print(f"{split}: {len(rows)} trajectories, {turns:.0f} turns avg, {probes} probes")


if __name__ == "__main__":
    main()
