"""Tests for interactive/berta.py"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from interactive import berta


class TestChaptersDict(unittest.TestCase):
    def test_chapters_has_25_entries(self):
        self.assertEqual(len(berta.CHAPTERS), 25)

    def test_all_learning_paths_reference_valid_chapters(self):
        valid_chapters = set(berta.CHAPTERS.keys())
        for path_id, path_data in berta.LEARNING_PATHS.items():
            for ch in path_data["chapters"]:
                self.assertIn(ch, valid_chapters, f"Path {path_id} references invalid chapter {ch}")

    def test_all_chapter_prerequisites_reference_valid_chapters(self):
        valid_chapters = set(berta.CHAPTERS.keys())
        for ch_num, ch_data in berta.CHAPTERS.items():
            for prereq in ch_data.get("prereqs", []):
                self.assertIn(prereq, valid_chapters, f"Chapter {ch_num} has invalid prereq {prereq}")


class TestProgress(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_progress = Path(self.temp_dir) / ".berta_progress.json"
        self.original_progress_file = berta.PROGRESS_FILE
        berta.PROGRESS_FILE = self.temp_progress

    def tearDown(self):
        berta.PROGRESS_FILE = self.original_progress_file
        if self.temp_progress.exists():
            self.temp_progress.unlink()
        Path(self.temp_dir).rmdir()

    def test_load_progress_returns_valid_structure_when_no_file_exists(self):
        progress = berta.load_progress()
        self.assertIn("learner_name", progress)
        self.assertIn("selected_path", progress)
        self.assertIn("chapters_completed", progress)
        self.assertIn("chapters_in_progress", progress)
        self.assertIn("quiz_scores", progress)
        self.assertIn("started_at", progress)
        self.assertIn("last_active", progress)
        self.assertIsNone(progress["learner_name"])
        self.assertIsNone(progress["selected_path"])
        self.assertEqual(progress["chapters_completed"], [])

    def test_save_progress_and_load_progress_roundtrip(self):
        original = {
            "learner_name": "Test User",
            "selected_path": "A",
            "chapters_completed": [1, 2, 3],
            "chapters_in_progress": [4],
            "quiz_scores": [80.0, 90.0],
            "started_at": "2025-01-01T00:00:00",
            "last_active": None,
        }
        berta.save_progress(original)
        loaded = berta.load_progress()
        self.assertEqual(loaded["learner_name"], original["learner_name"])
        self.assertEqual(loaded["selected_path"], original["selected_path"])
        self.assertEqual(loaded["chapters_completed"], original["chapters_completed"])
        self.assertEqual(loaded["chapters_in_progress"], original["chapters_in_progress"])
        self.assertEqual(loaded["quiz_scores"], original["quiz_scores"])
        self.assertIsNotNone(loaded["last_active"])


class TestLearningPaths(unittest.TestCase):
    def test_each_learning_path_has_correct_properties(self):
        required = {"name", "chapters", "hours", "description", "best_for"}
        for path_id, path_data in berta.LEARNING_PATHS.items():
            for key in required:
                self.assertIn(key, path_data, f"Path {path_id} missing property: {key}")
            self.assertIsInstance(path_data["name"], str)
            self.assertIsInstance(path_data["chapters"], list)
            self.assertIsInstance(path_data["hours"], (int, float))
            self.assertIsInstance(path_data["description"], str)
            self.assertIsInstance(path_data["best_for"], str)


class TestQuizQuestions(unittest.TestCase):
    def test_quiz_questions_have_valid_structure(self):
        required = {"question", "options", "answer", "chapter", "explanation"}
        for i, q in enumerate(berta.QUIZ_QUESTIONS):
            for key in required:
                self.assertIn(key, q, f"Question {i} missing key: {key}")
            self.assertIsInstance(q["options"], list)
            self.assertEqual(len(q["options"]), 4)
            self.assertIn(q["chapter"], berta.CHAPTERS)

    def test_quiz_answer_indices_within_valid_range(self):
        for i, q in enumerate(berta.QUIZ_QUESTIONS):
            self.assertGreaterEqual(q["answer"], 1, f"Question {i} answer {q['answer']} < 1")
            self.assertLessEqual(q["answer"], 4, f"Question {i} answer {q['answer']} > 4")
            self.assertLessEqual(q["answer"], len(q["options"]), f"Question {i} answer out of options")


if __name__ == "__main__":
    unittest.main()
