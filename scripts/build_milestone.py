# CS5491: Reasoning Augmented FunSearch
# Author: 王宇浩 (72655080), 朱泳颐 (72657023), 李逸洋 (72655555)
# Responsibility: Shared project setup, analysis and report writing

"""Compile the milestone report from LaTeX."""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "milestone/midterm_milestone.tex"
    build = source.parent / "rendered/latex"
    build.mkdir(parents=True, exist_ok=True)
    command = [
        "xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={build}",
        source.name,
    ]
    for _ in range(2):
        subprocess.run(command, cwd=source.parent, check=True)
    output = source.with_suffix(".pdf")
    shutil.copy2(build / output.name, output)
    print(f"Built {output}")


if __name__ == "__main__":
    main()
