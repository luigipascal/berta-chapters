"""Tests for chapter-01-python-fundamentals/exercises/solutions/solutions.py"""

import sys
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "chapters" / "chapter-01-python-fundamentals" / "exercises" / "solutions"))

import solutions


class TestTypeDetective(unittest.TestCase):
    def test_type_detective_int(self):
        result = solutions.type_detective(42)
        self.assertIn("integer", result)
        self.assertIn("positive", result)
        self.assertIn("even", result)

    def test_type_detective_float(self):
        result = solutions.type_detective(-3.14)
        self.assertIn("float", result)
        self.assertIn("negative", result)

    def test_type_detective_str(self):
        result = solutions.type_detective("hello")
        self.assertIn("string", result)
        self.assertIn("length=5", result)
        self.assertIn("lowercase", result)

    def test_type_detective_list(self):
        result = solutions.type_detective([1, 2, 3])
        self.assertIn("list", result)
        self.assertIn("length=3", result)

    def test_type_detective_bool(self):
        result = solutions.type_detective(True)
        self.assertIn("boolean", result)
        self.assertIn("True", result)

    def test_type_detective_none(self):
        result = solutions.type_detective(None)
        self.assertEqual(result, "NoneType")


class TestBatchStats(unittest.TestCase):
    def test_batch_stats_output_format_and_values(self):
        numbers = list(range(1, 11))
        result = solutions.batch_stats(numbers, batch_size=3)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["batch"], 0)
        self.assertEqual(result[0]["min"], 1)
        self.assertEqual(result[0]["max"], 3)
        self.assertAlmostEqual(result[0]["mean"], 2.0)
        self.assertEqual(result[3]["batch"], 3)
        self.assertEqual(result[3]["min"], 10)
        self.assertEqual(result[3]["max"], 10)


class TestTokenize(unittest.TestCase):
    def test_tokenize_removes_stop_words_and_lowercases(self):
        tokens, vocab = solutions.tokenize("The quick brown fox jumps over the lazy dog.")
        self.assertNotIn("the", tokens)
        self.assertIn("quick", tokens)
        self.assertIn("brown", tokens)
        self.assertEqual(tokens[0], "quick")
        for t in tokens:
            self.assertEqual(t, t.lower())


class TestAnalyzeTraining(unittest.TestCase):
    def test_analyze_training_identifies_best_epoch(self):
        history = {
            "train_loss": [0.9, 0.5, 0.3, 0.15, 0.08],
            "val_loss": [0.95, 0.55, 0.35, 0.32, 0.40],
        }
        result = solutions.analyze_training(history)
        self.assertEqual(result["best_epoch"], 4)
        self.assertEqual(result["overfit_epoch"], 5)


class TestModelRegistry(unittest.TestCase):
    def test_register_and_get_best(self):
        reg = solutions.ModelRegistry()
        reg.register("model_a", "v1", {"accuracy": 0.85})
        reg.register("model_a", "v2", {"accuracy": 0.92})
        reg.register("model_b", "v1", {"accuracy": 0.88})
        best = reg.get_best(metric="accuracy", higher_is_better=True)
        self.assertEqual(best["name"], "model_a")
        self.assertEqual(best["version"], "v2")
        self.assertEqual(best["metrics"]["accuracy"], 0.92)

    def test_list_versions(self):
        reg = solutions.ModelRegistry()
        reg.register("bert", "v1", {"accuracy": 0.89})
        reg.register("bert", "v2", {"accuracy": 0.93})
        reg.register("gpt", "v1", {"accuracy": 0.91})
        versions = reg.list_versions("bert")
        self.assertEqual(len(versions), 2)
        names = {v["version"] for v in versions}
        self.assertEqual(names, {"v1", "v2"})


class TestPipeline(unittest.TestCase):
    def test_add_step_run_describe(self):
        pipe = solutions.Pipeline()
        pipe.add_step("lowercase", str.lower).add_step("strip", str.strip)
        result = pipe.run("  Hello World  ")
        self.assertEqual(result, "hello world")
        self.assertEqual(len(pipe.steps), 2)

    def test_pipeline_describe(self):
        pipe = solutions.Pipeline()
        pipe.add_step("step1", lambda x: x).add_step("step2", lambda x: x)
        out = StringIO()
        import sys
        old_stdout = sys.stdout
        sys.stdout = out
        try:
            pipe.describe()
            output = out.getvalue()
        finally:
            sys.stdout = old_stdout
        self.assertIn("step1", output)
        self.assertIn("step2", output)
        self.assertIn(" -> ", output)


if __name__ == "__main__":
    unittest.main()
