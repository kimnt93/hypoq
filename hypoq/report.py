"""
LaTeX report helpers for hypoq research folders.

Every number in a manuscript must be formatted from a computed result dictionary, never typed by hand.
"""

import os
import shutil
import subprocess
from typing import Dict, Optional

TECTONIC = os.environ.get("TECTONIC_BIN") or shutil.which("tectonic") or "/home/kim/.local/bin/tectonic"

_PREAMBLE = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{authblk}
\usepackage{setspace}
\usepackage[margin=1.25in]{geometry}
\usepackage{graphicx}
\graphicspath{ {./figures/} }
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{natbib}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}
"""


def latex_escape(text: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        text = text.replace(a, b)
    return text


def write_paper(
    research_dir: str,
    title: str,
    abstract: Dict[str, str],
    body: str,
    bib: str,
) -> str:
    """Writes main.tex and references.bib. `abstract` maps bold label -> LaTeX text; `body` is LaTeX."""
    abs_tex = " \\\\\n".join(f"\\textbf{{{k}:}} {v}" for k, v in abstract.items())
    tex = (
        _PREAMBLE
        + f"\n\\title{{\\textbf{{{title}}}}}\n"
        + "\\author[1*]{hypoq autonomous research harness}\n"
        + "\\affil[1]{Hypothesis $\\times$ Quant}\n\\date{\\today}\n\\onehalfspacing\n\n"
        + "\\begin{document}\n\\maketitle\n\n"
        + f"\\begin{{abstract}}\n\\noindent {abs_tex}\n\\end{{abstract}}\n\n"
        + body
        + "\n\n\\bibliographystyle{plainnat}\n\\bibliography{references}\n\n\\end{document}\n"
    )
    with open(os.path.join(research_dir, "main.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    with open(os.path.join(research_dir, "references.bib"), "w", encoding="utf-8") as f:
        f.write(bib)
    return os.path.join(research_dir, "main.tex")


def compile_pdf(research_dir: str) -> Optional[str]:
    """Compiles main.tex with Tectonic; returns the PDF path or None if compilation failed."""
    try:
        proc = subprocess.run(
            [TECTONIC, "main.tex"], cwd=research_dir, capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[report] tectonic unavailable: {exc}")
        return None
    if proc.returncode != 0:
        print(proc.stderr[-1500:])
        return None
    return os.path.join(research_dir, "main.pdf")
