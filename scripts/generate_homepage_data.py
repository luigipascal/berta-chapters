#!/usr/bin/env python3
"""
Generate homepage stats and chapter tables from chapters/chapter-*/.
Run from repo root. Updates docs/index.md in place by replacing the
content between <!-- AUTO_HOMEPAGE_DATA --> and <!-- /AUTO_HOMEPAGE_DATA -->.
This keeps all links relative to index.md so MkDocs resolves them correctly.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTERS_DIR = REPO_ROOT / "chapters"
INDEX_PATH = REPO_ROOT / "docs" / "index.md"
AUTO_START = "<!-- AUTO_HOMEPAGE_DATA -->"
AUTO_END = "<!-- /AUTO_HOMEPAGE_DATA -->"


def parse_readme(chapter_path: Path) -> dict:
    """Extract title, time, track from README.md."""
    readme = chapter_path / "README.md"
    out = {"title": "", "hours": "", "track": "Practitioner"}
    if not readme.exists():
        return out
    text = readme.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^#\s+Chapter\s+\d+:\s*(.+?)(?:\s*\||$)", text, re.MULTILINE)
    if m:
        out["title"] = m.group(1).strip()
    m = re.search(r"\*\*Time\*\*:\s*([\d–\-]+(?:\s*hours?)?)", text)
    if m:
        out["hours"] = m.group(1).strip()
    else:
        m = re.search(r"\|\s*\*\*Total\*\*\s*\|\s*\*\*([\d–\-]+(?:\s*hours?)?)\*\*", text)
        if m:
            out["hours"] = m.group(1).strip()
    if not out["hours"]:
        m = re.search(r"Time\s*\|\s*([\d–\-]+h)", text)
        if m:
            out["hours"] = m.group(1).strip()
    if "**Track**: Foundation" in text or "Track**: Foundation" in text:
        out["track"] = "Foundation"
    return out


def count_notebooks(chapter_path: Path) -> int:
    nb_dir = chapter_path / "notebooks"
    return len(list(nb_dir.glob("*.ipynb"))) if nb_dir.is_dir() else 0


def count_exercises(chapter_path: Path) -> int:
    ex_dir = chapter_path / "exercises"
    if not ex_dir.is_dir():
        return 0
    ipynbs = list(ex_dir.glob("*.ipynb"))
    solution_dir = ex_dir / "solutions"
    solution_ipynbs = list(solution_dir.glob("*.ipynb")) if solution_dir.is_dir() else []
    has_exercises_py = (ex_dir / "exercises.py").exists()
    has_solutions_py = (solution_dir / "solutions.py").exists()
    n = len(ipynbs)
    if n == 0 and has_exercises_py:
        n = 5
    if n == 0 and (solution_ipynbs or has_solutions_py):
        n = max(len(solution_ipynbs), 1)
    return n if n > 0 else (len(solution_ipynbs) or (1 if has_solutions_py else 0))


def count_diagrams(chapter_path: Path) -> int:
    d_dir = chapter_path / "assets" / "diagrams"
    return len(list(d_dir.iterdir())) if d_dir.is_dir() else 0


def get_chapter_number(chapter_path: Path) -> int:
    name = chapter_path.name
    m = re.match(r"chapter-(\d+)-", name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def collect_chapters() -> list[dict]:
    chapters = []
    for path in sorted(CHAPTERS_DIR.iterdir()):
        if not path.is_dir() or not path.name.startswith("chapter-"):
            continue
        num = get_chapter_number(path)
        if num <= 0:
            continue
        meta = parse_readme(path)
        doc_slug = f"chapter-{num:02d}"
        chapters.append({
            "num": num,
            "path": path,
            "title": meta["title"] or path.name.replace(f"chapter-{num:02d}-", "").replace("-", " ").title(),
            "hours": meta["hours"] or "—",
            "track": meta["track"],
            "notebooks": count_notebooks(path),
            "exercises": count_exercises(path),
            "diagrams": count_diagrams(path),
            "doc_link": f"chapters/{doc_slug}.md",
        })
    return sorted(chapters, key=lambda c: c["num"])


def format_includes(ch: dict) -> str:
    parts = []
    if ch["notebooks"]:
        parts.append(f"{ch['notebooks']} notebooks")
    if ch["exercises"]:
        parts.append(f"{ch['exercises']} exercises")
    if ch["diagrams"]:
        parts.append(f"{ch['diagrams']} diagrams")
    return ", ".join(parts) if parts else "—"


def main() -> None:
    chapters = collect_chapters()
    if not chapters:
        print("No chapters found.")
        return

    total_notebooks = sum(c["notebooks"] for c in chapters)
    total_diagrams = sum(c["diagrams"] for c in chapters)
    total_exercises = sum(c["exercises"] for c in chapters)
    total_hours = 0
    for c in chapters:
        h = c["hours"]
        if h and h != "—":
            nums = re.findall(r"\d+", h)
            if nums:
                total_hours += int(nums[0])

    # Build the block to inject (all links relative to docs/index.md)
    stats = f"""<div class="stats-grid" markdown>

<div class="stat-card" markdown>
<div class="number">{len(chapters)}</div>
<div class="label">Chapters</div>
</div>

<div class="stat-card" markdown>
<div class="number">{total_notebooks}</div>
<div class="label">Notebooks</div>
</div>

<div class="stat-card" markdown>
<div class="number">{total_diagrams}</div>
<div class="label">Diagrams</div>
</div>

<div class="stat-card" markdown>
<div class="number">{total_hours}h</div>
<div class="label">Content</div>
</div>

<div class="stat-card" markdown>
<div class="number">{total_exercises}</div>
<div class="label">Exercises</div>
</div>

<div class="stat-card" markdown>
<div class="number">$0</div>
<div class="label">Forever</div>
</div>

</div>"""

    foundation = [c for c in chapters if c["num"] <= 5]
    practitioner = [c for c in chapters if c["num"] > 5]
    table_lines = [
        "## Available Chapters",
        "",
        "### Foundation Track — Complete",
        "",
        "| # | Chapter | Hours | Includes |",
        "|---|---------|-------|----------|",
    ]
    for c in foundation:
        table_lines.append(f"| {c['num']} | [{c['title']}]({c['doc_link']}) | {c['hours']} | {format_includes(c)} |")
    table_lines.extend([
        "",
        "### Practitioner Track — In Progress",
        "",
        "| # | Chapter | Hours | Includes |",
        "|---|---------|-------|----------|",
    ])
    for c in practitioner:
        table_lines.append(f"| {c['num']} | [{c['title']}]({c['doc_link']}) | {c['hours']} | {format_includes(c)} |")
    next_num = max(c["num"] for c in chapters) + 1
    if next_num <= 25:
        table_lines.append(f"| {next_num}–25 | Coming soon | | [View roadmap](guides/roadmap.md) |")
    table_lines.append("")

    max_ch = max(c["num"] for c in chapters)
    playground_line = f"{total_exercises} pre-built exercises covering Chapters 1–{max_ch}. Load one and start coding."

    generated = (
        stats + "\n\n---\n\n" + "\n".join(table_lines)
        + "\n\n---\n\n## Online Playground\n\n"
        + "Practice Python directly in your browser. No installation required.\n"
        + "Errors are explained in plain English.\n\n"
        + "[Open the Playground](playground.md){ .md-button .md-button--primary }\n\n"
        + playground_line
    )

    # Replace placeholder in index.md
    if not INDEX_PATH.exists():
        print(f"Not found: {INDEX_PATH}")
        return
    content = INDEX_PATH.read_text(encoding="utf-8")
    if AUTO_START not in content or AUTO_END not in content:
        print("Placeholder not found in index.md. Add <!-- AUTO_HOMEPAGE_DATA --> ... <!-- /AUTO_HOMEPAGE_DATA -->")
        return
    pattern = re.compile(re.escape(AUTO_START) + r".*?" + re.escape(AUTO_END), re.DOTALL)
    new_content = pattern.sub(AUTO_START + "\n" + generated + "\n" + AUTO_END, content)
    INDEX_PATH.write_text(new_content, encoding="utf-8")

    print(f"Updated index.md: {len(chapters)} chapters, {total_notebooks} notebooks, {total_exercises} exercises, {total_diagrams} diagrams, {total_hours}h")


if __name__ == "__main__":
    main()
