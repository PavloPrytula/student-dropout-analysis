"""Small research checks independent of raw data, training runs and private files."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.optimize import check_grad

from student_dropout.data import FEATURES
from student_dropout.decision import allocation_metrics
from student_dropout.prediction import ProbabilityLogit

ROOT = Path(__file__).resolve().parents[1]


class MethodologyTests(unittest.TestCase):
    def test_frozen_nested_split_containment(self):
        split = json.loads((ROOT / "assets/results/split.json").read_text())
        folds = json.loads((ROOT / "assets/results/folds.json").read_text())
        train, test = set(split["train"]), set(split["test"])
        self.assertEqual((len(train), len(test)), (3539, 885))
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(range(4424)))
        seen = []
        for i in range(5):
            fold = folds[str(i)]
            a, b = set(fold["train"]), set(fold["valid"])
            self.assertFalse(a & b)
            self.assertEqual(a | b, train)
            seen.extend(b)
            for inner in fold["inner"]:
                c, d = set(inner["train"]), set(inner["valid"])
                self.assertFalse(c & d)
                self.assertEqual(c | d, a)
        self.assertEqual(sorted(seen), sorted(train))
        for fold in folds["final"]:
            a, b = set(fold["train"]), set(fold["valid"])
            self.assertFalse(a & b)
            self.assertEqual(a | b, train)

    def test_objective_gradients(self):
        rng = np.random.default_rng(23)
        x, y, w = rng.normal(size=(30, 5)), rng.integers(2, size=30), rng.normal(size=6)
        for criterion in ("log", "brier"):
            model = ProbabilityLogit(alpha=0.03, criterion=criterion)
            error = check_grad(
                lambda b: model.loss_grad(b, x, y)[0],
                lambda b: model.loss_grad(b, x, y)[1],
                w,
            )
            self.assertLess(error, 1e-6)

    def test_random_comparator_and_boundaries(self):
        result = allocation_metrics(100, 30, 10, 6)
        self.assertEqual(result["random_expected"], 3)
        self.assertEqual(result["extra_identified"], 3)
        self.assertEqual(result["lift"], 2)
        self.assertAlmostEqual(result["random_variance"], 10 * 0.3 * 0.7 * 90 / 99)
        self.assertEqual(allocation_metrics(100, 30, 100, 30)["extra_identified"], 0)
        self.assertEqual(allocation_metrics(0, 0, 0, 0)["random_expected"], 0)
        with self.assertRaises(ValueError):
            allocation_metrics(10, 3, 2, 3)

    def test_enrolment_schema(self):
        self.assertEqual(len(FEATURES), 18)
        self.assertFalse(any("sem" in name for name in FEATURES))
        for name in ("Target", "Debtor", "Scholarship holder", "GDP"):
            self.assertNotIn(name, FEATURES)
