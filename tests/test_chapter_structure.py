"""Tests for chapter directory structure validation."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

WORKSPACE = Path(__file__).parent.parent
CHAPTERS_DIR = WORKSPACE / "chapters"


class TestChapterStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chapter_dirs = sorted(CHAPTERS_DIR.iterdir()) if CHAPTERS_DIR.exists() else []
        cls.chapter_dirs = [d for d in cls.chapter_dirs if d.is_dir() and d.name.startswith("chapter-")]

    def test_each_chapter_has_readme(self):
        for chapter_dir in self.chapter_dirs:
            readme = chapter_dir / "README.md"
            self.assertTrue(readme.exists(), f"{chapter_dir.name} missing README.md")

    def test_each_chapter_has_requirements_txt(self):
        for chapter_dir in self.chapter_dirs:
            req = chapter_dir / "requirements.txt"
            self.assertTrue(req.exists(), f"{chapter_dir.name} missing requirements.txt")

    def test_each_chapter_has_notebooks_directory_with_ipynb(self):
        for chapter_dir in self.chapter_dirs:
            notebooks_dir = chapter_dir / "notebooks"
            self.assertTrue(notebooks_dir.is_dir(), f"{chapter_dir.name} missing notebooks/ directory")
            ipynb_files = list(notebooks_dir.glob("*.ipynb"))
            self.assertGreater(len(ipynb_files), 0, f"{chapter_dir.name} has no .ipynb files in notebooks/")

    def test_each_chapter_has_scripts_directory(self):
        for chapter_dir in self.chapter_dirs:
            scripts_dir = chapter_dir / "scripts"
            self.assertTrue(scripts_dir.is_dir(), f"{chapter_dir.name} missing scripts/ directory")

    def test_each_chapter_has_exercises_directory(self):
        for chapter_dir in self.chapter_dirs:
            exercises_dir = chapter_dir / "exercises"
            self.assertTrue(exercises_dir.is_dir(), f"{chapter_dir.name} missing exercises/ directory")

    def test_all_ipynb_files_valid_json_with_cells_and_nbformat(self):
        for chapter_dir in self.chapter_dirs:
            notebooks_dir = chapter_dir / "notebooks"
            if not notebooks_dir.exists():
                continue
            for ipynb in notebooks_dir.glob("*.ipynb"):
                with self.subTest(notebook=ipynb):
                    try:
                        with open(ipynb) as f:
                            nb = json.load(f)
                    except json.JSONDecodeError as e:
                        self.fail(f"{ipynb}: invalid JSON - {e}")
                    self.assertIn("cells", nb, f"{ipynb} missing 'cells' key")
                    self.assertIn("nbformat", nb, f"{ipynb} missing 'nbformat' key")


if __name__ == "__main__":
    unittest.main()
