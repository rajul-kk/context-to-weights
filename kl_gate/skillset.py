import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import read_json, read_jsonl

SKILL_SYSTEM = "You are a precise engineering assistant. Answer with the exact API usage."


def load_skill(root, name):
    d = Path(root) / name
    return {
        "name": name,
        "doc": (d / "SKILL.md").read_text(encoding="utf-8"),
        "demos": read_jsonl(d / "demos.jsonl"),
        "tasks": read_jsonl(d / "tasks.jsonl"),
    }


def load_index(root):
    return read_json(Path(root) / "index.json")


def load_all(root, names=None):
    index = load_index(root)
    if names:
        index = [r for r in index if r["name"] in names]
    return [dict(load_skill(root, r["name"]), category=r["category"]) for r in index]


def by_category(root):
    groups = {}
    for row in load_index(root):
        groups.setdefault(row["category"], []).append(row["name"])
    return groups


def with_skill(doc, query):
    return f"{doc}\n\nTask: {query}"


def without_skill(query):
    return f"Task: {query}"
