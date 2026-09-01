import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import load_config, parse_overrides, write_json
from compactor.scoring import ScoringCompactor
from sleep.lm import load_backbone

SALIENT = [
    "One thing to lock in: the Data pod owns user-profile and signs off on every schema change.",
    "The public rate limit for billing is fixed at 500 req/min.",
    "quartz-ingest pins its client library to version 3.11.2 until the migration lands.",
    "The auth token travels in the X-Session-Key header, not Authorization.",
    "Events for search-index land on the queue named evt.harrow.v1.",
    "The kill switch is the flag WRITE_FUSE_OPEN; flipping it disables writes.",
]

FILLER = [
    "Give each test its own temp directory and it settles.",
    "Right now it is still readable.",
    "Makes sense, thanks.",
    "The docs for search-index are stale, want me to rewrite them?",
    "Cold start warms the pool lazily, so the first request pays the connection cost.",
    "Structured logging is fine. Keep the field names consistent.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    model, tokenizer = load_backbone(cfg)
    scorer = ScoringCompactor(model, tokenizer, batch_size=8,
                              max_length=cfg["model"]["max_length"])

    texts = SALIENT + FILLER
    scores = scorer.score_spans(texts)
    salient = scores[: len(SALIENT)]
    filler = scores[len(SALIENT):]

    print(f"model {cfg['model']['base']}\n")
    for label, group, vals in (("salient", SALIENT, salient), ("filler", FILLER, filler)):
        print(f"{label}:")
        for v, t in sorted(zip(vals, group), reverse=True):
            print(f"  {v:+7.3f}  {t[:78]}")
        print()

    mean_s = sum(salient) / len(salient)
    mean_f = sum(filler) / len(filler)
    ranked = sorted(zip(scores, [1] * len(salient) + [0] * len(filler)), reverse=True)
    top_half = sum(lab for _, lab in ranked[: len(salient)])
    report = {
        "model": cfg["model"]["base"],
        "mean_salient": mean_s,
        "mean_filler": mean_f,
        "separation": mean_s - mean_f,
        "range": max(scores) - min(scores),
        "salient_in_top_half": top_half / len(salient),
    }
    for k, v in report.items():
        print(f"{k:<22} {v}")

    if report["separation"] <= 0:
        print("\nWARNING: salient spans do not score above filler. This scorer cannot rank "
              "salience and\nany compaction built on it selects no better than chance.")
    elif report["range"] < 2.0:
        print("\nWARNING: the score range is narrow. Ranking may be dominated by surface "
              "features\nrather than content.")

    if args.out:
        write_json(args.out, report)


if __name__ == "__main__":
    main()
