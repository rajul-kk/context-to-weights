import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "draft_combined.md"

KEYS = {1: "dennis2026beyond", 2: "wang2026scol", 3: "colaco2026keep", 4: "li2026compactionrl",
        5: "ho2026attention", 6: "zhang2026skill", 7: "saini2026thinkswitch", 8: "guo2026peam",
        9: "behrouz2024titans", 10: "zhang2025sentinel", 11: "wu2024retrieval",
        12: "khodabandehlou2026abstention", 13: "xu2026tip", 14: "lin2024rho", 15: "hu2021lora",
        16: "qwen2024qwen25"}
TEXTUAL = ("following", "Following", "of", "in", "accuracy")

PREAMBLE = r"""\documentclass[10pt]{article}
\pdfinfoomitdate=1
\pdftrailerid{}
\pdfsuppressptexinfo=-1
\usepackage{tmlr}
\input{math_commands.tex}
\usepackage{hyperref}
\usepackage{url}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{calc}
\usepackage{newunicodechar}
\usepackage{graphicx}
\graphicspath{{../}}
\providecommand{\pandocbounded}[1]{#1}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{Σ}{\ensuremath{\Sigma}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≈}{\ensuremath{\approx}}
"""


CITE = re.compile(r"([A-Za-z'-]*)(\s*)\[(\d{1,2})\]")


def cite(m):
    word, space, num = m.group(1), m.group(2), int(m.group(3))
    cmd = "citet" if word in TEXTUAL else "citep"
    return f"{word}{space or ' '}\\{cmd}{{{KEYS[num]}}}" if word else f"\\{cmd}{{{KEYS[num]}}}"


def section(md, title):
    m = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", md, re.S | re.M)
    return m.group(1).strip()


def main():
    md = SRC.read_text(encoding="utf-8")
    title = md.splitlines()[0].lstrip("# ").strip()
    abstract = section(md, "Abstract")
    body = md.split("## 1. Introduction", 1)[1]
    body = "## 1. Introduction" + body.split("## References", 1)[0]
    appendix = md.split("## Appendix A.", 1)[1]
    body = body.split("## Appendix A.", 1)[0]
    body += "\n\\appendix\n\n## Appendix A." + appendix
    body = re.sub(r"^(#{2,3}) (?:Appendix [A-Z]\.|\d+(?:\.\d+)?\.?) ", r"\1 ", body, flags=re.M)
    body = CITE.sub(cite, body)
    abstract = CITE.sub(cite, abstract)

    def pandoc(text, *extra):
        r = subprocess.run(["pandoc", "-f", "markdown+raw_tex-auto_identifiers", "-t", "latex",
                            "--wrap=preserve", "--columns=100", "--no-highlight", *extra],
                           input=text, capture_output=True, text=True, encoding="utf-8", check=True)
        return r.stdout

    tex_body = pandoc(body, "--shift-heading-level-by=-1")
    tex_abstract = pandoc(abstract)
    tex_title = pandoc(title).strip()
    doc = (PREAMBLE + f"\n\\title{{{tex_title}}}\n\\author{{Anonymous authors}}\n\n\\begin{{document}}\n"
           "\\maketitle\n\n\\begin{abstract}\n" + tex_abstract + "\\end{abstract}\n\n" + tex_body +
           "\n\\end{document}\n")
    doc = doc.replace("\\appendix\n\n\\section{", "\\bibliography{main}\n\\bibliographystyle{tmlr}\n\n"
                      "\\newpage\n\\appendix\n\n\\section{")
    (HERE / "main.tex").write_text(doc, encoding="utf-8")
    if "--no-pdf" not in sys.argv:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", "main.tex"],
                       cwd=HERE, check=True)


if __name__ == "__main__":
    main()
