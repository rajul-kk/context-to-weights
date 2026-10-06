import ast
import io
import json
import re
import runpy
import tokenize

import pytest
from helpers import ROOT, repo_files

from scripts.check_notebooks import python_only


def python_files():
    return repo_files("*.py")


def notebooks():
    return repo_files("notebooks/*.ipynb")


def comment_lines(source):
    found = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT and not (tok.start[0] == 1 and tok.string.startswith("#!")):
            found.append(tok.start[0])
    return found


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_python_has_no_comments_or_docstrings(path):
    source = path.read_text(encoding="utf-8")
    assert comment_lines(source) == []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            assert ast.get_docstring(node) is None, f"{path.name}: docstring on {getattr(node, 'name', 'module')}"


@pytest.mark.parametrize("path", notebooks(), ids=lambda p: p.name)
def test_notebooks_compile_and_have_no_code_comments(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = python_only(cell["source"])
        compile(source, f"{path.name}:cell{i}", "exec")
        assert comment_lines(source) == [], f"{path.name} cell {i}"


@pytest.mark.parametrize("path", notebooks(), ids=lambda p: p.name)
def test_notebooks_are_committed_without_outputs(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    assert all(not c.get("outputs") for c in cells if c["cell_type"] == "code")


def test_paper_has_no_em_dashes():
    for path in (ROOT / "paper" / "draft_combined.md", ROOT / "paper" / "tmlr" / "main.tex"):
        assert "—" not in path.read_text(encoding="utf-8"), path.name


def test_paper_citations_resolve():
    text = (ROOT / "paper" / "draft_combined.md").read_text(encoding="utf-8")
    body, refs = text.split("\n## References\n", 1)
    defined = [int(n) for n in re.findall(r"(?m)^\[(\d+)\] ", refs)]
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    assert defined == list(range(1, len(defined) + 1))
    assert cited <= set(defined)
    assert set(defined) <= cited


def test_every_citation_key_has_a_bibtex_entry():
    keys_in_build = runpy.run_path(str(ROOT / "paper" / "tmlr" / "build.py"))["KEYS"]
    bib = (ROOT / "paper" / "tmlr" / "main.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    assert set(keys_in_build.values()) == keys
