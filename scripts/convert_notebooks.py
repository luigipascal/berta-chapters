#!/usr/bin/env python3
"""
Convert Jupyter notebooks to web-readable Markdown for the MkDocs site.

Reads all .ipynb files from chapters/ and writes clean .md files into
docs/chapters/content/ with proper formatting, code blocks, and links
to the playground.

Usage:
    python scripts/convert_notebooks.py
"""

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHAPTERS_DIR = ROOT / "chapters"
OUTPUT_DIR = ROOT / "docs" / "chapters" / "content"


CHAPTER_META = {
    "chapter-01-python-fundamentals": ("Ch 1", "Python Fundamentals for AI", "Foundation"),
    "chapter-02-data-structures": ("Ch 2", "Data Structures & Algorithms", "Foundation"),
    "chapter-03-linear-algebra": ("Ch 3", "Linear Algebra & Calculus", "Foundation"),
    "chapter-04-probability-statistics": ("Ch 4", "Probability & Statistics", "Foundation"),
    "chapter-05-software-design": ("Ch 5", "Software Design & Best Practices", "Foundation"),
    "chapter-06-intro-machine-learning": ("Ch 6", "Introduction to Machine Learning", "Practitioner"),
    "chapter-07-supervised-learning": ("Ch 7", "Supervised Learning", "Practitioner"),
}

NOTEBOOK_LABELS = {
    "01_introduction": "Introduction",
    "02_intermediate": "Intermediate",
    "03_advanced": "Advanced",
}


def notebook_to_markdown(nb_path, chapter_slug):
    """Convert a single notebook to clean markdown."""
    with open(nb_path) as f:
        nb = json.load(f)

    ch_short, ch_title, track = CHAPTER_META.get(chapter_slug, ("?", "?", "?"))
    nb_name = nb_path.stem
    nb_label = NOTEBOOK_LABELS.get(nb_name, nb_name)

    lines = []
    lines.append(f"# {ch_short}: {ch_title} - {nb_label}\n")
    lines.append(f"**Track**: {track} | "
                 f"[Try code in Playground](../../playground.md) | "
                 f"[Back to chapter overview](../chapter-{chapter_slug.split('-')[1]}.md)\n")
    lines.append("")
    lines.append("!!! tip \"Read online or run locally\"")
    lines.append("    You can read this content here on the web. To run the code interactively,")
    lines.append("    either use the [Playground](../../playground.md) or clone the repo and open")
    lines.append(f"    `chapters/{chapter_slug}/notebooks/{nb_path.name}` in Jupyter.\n")
    lines.append("---\n")

    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "")
        source_lines = cell.get("source", [])
        if isinstance(source_lines, list):
            source = "".join(source_lines)
        else:
            source = source_lines

        source = source.strip()
        if not source:
            continue

        if cell_type == "markdown":
            lines.append(source)
            lines.append("")
        elif cell_type == "code":
            lines.append("```python")
            lines.append(source)
            lines.append("```\n")

    lines.append("---\n")
    lines.append(f"*[Back to {ch_short} overview](../chapter-{chapter_slug.split('-')[1]}.md) | "
                 f"[Try in Playground](../../playground.md) | "
                 f"[View on GitHub](https://github.com/luigipascal/berta-chapters/tree/main/chapters/{chapter_slug}/notebooks/{nb_path.name})*")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    converted = 0
    for chapter_dir in sorted(CHAPTERS_DIR.iterdir()):
        if not chapter_dir.is_dir() or not chapter_dir.name.startswith("chapter-"):
            continue

        nb_dir = chapter_dir / "notebooks"
        if not nb_dir.exists():
            continue

        slug = chapter_dir.name
        ch_num = slug.split("-")[1]

        for nb_path in sorted(nb_dir.glob("*.ipynb")):
            md_content = notebook_to_markdown(nb_path, slug)
            out_name = f"ch{ch_num}-{nb_path.stem}.md"
            out_path = OUTPUT_DIR / out_name
            out_path.write_text(md_content)
            converted += 1
            print(f"  {nb_path.name} -> {out_name}")

    print(f"\nConverted {converted} notebooks to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
