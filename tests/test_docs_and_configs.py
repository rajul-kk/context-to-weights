import re

import pytest
from helpers import ROOT, repo_files

from common.io import load_config

REPO_PATH = re.compile(
    r"`((?:configs|scripts|eval|data|sleep|declare|distill|kl_gate|compactor|baselines|common|"
    r"notebooks|skills|docs|paper|tests)/[A-Za-z0-9_./-]+\.(?:py|yaml|md|ipynb|json|jsonl))`")
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


@pytest.mark.parametrize("path", repo_files("*.md"), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_markdown_links_and_paths_resolve(path):
    text = path.read_text(encoding="utf-8")
    broken = [t for t in LINK.findall(text)
              if not t.startswith(("http", "mailto")) and not (path.parent / t).exists()]
    missing = [p for p in REPO_PATH.findall(text) if not (ROOT / p).exists()]
    assert broken == [] and missing == []


CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_loads_with_its_base(path):
    cfg = load_config(path)
    assert isinstance(cfg, dict)
    assert "_base_" not in cfg


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_data_sources_have_a_loader(path):
    source = load_config(path).get("data", {}).get("source")
    assert source in (None, "synthetic", "hotpotqa")
