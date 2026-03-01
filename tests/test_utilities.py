"""Tests for chapter-01-python-fundamentals/scripts/utilities.py"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "chapters" / "chapter-01-python-fundamentals" / "scripts"))

import utilities


class TestNormalize(unittest.TestCase):
    def test_normalize_min_max_method(self):
        data = [1, 2, 3, 4, 5]
        result = utilities.normalize(data, method="min-max")
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[-1], 1.0)
        self.assertEqual(len(result), len(data))

    def test_normalize_z_score_method(self):
        data = [1, 2, 3, 4, 5]
        result = utilities.normalize(data, method="z-score")
        self.assertEqual(len(result), len(data))
        mean = sum(result) / len(result)
        self.assertAlmostEqual(mean, 0.0, places=10)

    def test_normalize_empty_list(self):
        result = utilities.normalize([], method="min-max")
        self.assertEqual(result, [])
        result = utilities.normalize([], method="z-score")
        self.assertEqual(result, [])


class TestTrainTestSplit(unittest.TestCase):
    def test_train_test_split_preserves_data(self):
        data = list(range(100))
        train, test = utilities.train_test_split(data, test_ratio=0.2, seed=42)
        combined = sorted(train + test)
        self.assertEqual(combined, data)
        self.assertEqual(len(train) + len(test), len(data))

    def test_train_test_split_seed_reproducibility(self):
        data = list(range(50))
        train1, test1 = utilities.train_test_split(data, test_ratio=0.3, seed=123)
        train2, test2 = utilities.train_test_split(data, test_ratio=0.3, seed=123)
        self.assertEqual(train1, train2)
        self.assertEqual(test1, test2)


class TestAccuracyScore(unittest.TestCase):
    def test_accuracy_score_perfect_predictions(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        self.assertEqual(utilities.accuracy_score(y_true, y_pred), 1.0)

    def test_accuracy_score_imperfect_predictions(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 99]
        self.assertEqual(utilities.accuracy_score(y_true, y_pred), 0.8)


class TestConfusionMatrix(unittest.TestCase):
    def test_confusion_matrix_produces_correct_counts(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 0, 0, 1]
        matrix = utilities.confusion_matrix(y_true, y_pred)
        self.assertEqual(matrix[0][0], 2)
        self.assertEqual(matrix[0][1], 0)
        self.assertEqual(matrix[1][0], 1)
        self.assertEqual(matrix[1][1], 2)


class TestBatchIterator(unittest.TestCase):
    def test_batch_iterator_yields_correct_number_of_batches(self):
        data = list(range(100))
        batches = list(utilities.batch_iterator(data, batch_size=32, shuffle=False, seed=42))
        self.assertEqual(len(batches), 4)
        self.assertEqual(len(batches[0]), 32)
        self.assertEqual(len(batches[1]), 32)
        self.assertEqual(len(batches[2]), 32)
        self.assertEqual(len(batches[3]), 4)


class TestWordFrequencies(unittest.TestCase):
    def test_word_frequencies_returns_correct_counts(self):
        text = "hello world hello test world hello"
        result = utilities.word_frequencies(text, top_n=5)
        self.assertEqual(result[0], ("hello", 3))
        self.assertEqual(result[1], ("world", 2))
        self.assertEqual(result[2], ("test", 1))


class TestCosineSimilarity(unittest.TestCase):
    def test_cosine_similarity_with_known_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(utilities.cosine_similarity(a, b), 1.0, places=6)

        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(utilities.cosine_similarity(a, b), 0.0, places=6)


class TestFormatMetrics(unittest.TestCase):
    def test_format_metrics_produces_formatted_string(self):
        metrics = {"accuracy": 0.923, "f1_score": 0.917, "loss": 0.184}
        result = utilities.format_metrics(metrics)
        self.assertIn("accuracy", result)
        self.assertIn("0.9230", result)
        self.assertIn("f1_score", result)


if __name__ == "__main__":
    unittest.main()
