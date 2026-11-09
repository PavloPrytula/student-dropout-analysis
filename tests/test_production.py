"""Portable product checks: synthetic data only; no reserved outcomes or training."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import numpy as np
import pandas as pd

from student_dropout import production as p
from student_dropout.decision import select_top


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(p.ROOT / "examples/synthetic_students.csv")

    def test_prediction_parity_and_dropout_class(self):
        model = joblib.load(p.ROOT / "models/P4.joblib")
        self.assertEqual(model.classes_.tolist(), [0, 1])
        expected = json.loads((p.ROOT / "assets/parity_expected.json").read_text())
        actual = p.predict(self.df)
        np.testing.assert_allclose(
            actual, expected["dropout_probability"], atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            actual, model.predict_proba(self.df[p.FEATURES])[:, 1], atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            actual + model.predict_proba(self.df[p.FEATURES])[:, 0],
            1,
            atol=1e-12,
            rtol=0,
        )

    def test_no_leakage_and_extra_fields(self):
        forbidden = [
            "Target",
            "Debtor",
            "Tuition fees up to date",
            "Scholarship holder",
            "GDP",
            "Inflation rate",
            "Unemployment rate",
        ] + [
            f"Curricular units {s} sem ({v})"
            for s in ["1st", "2nd"]
            for v in [
                "credited",
                "enrolled",
                "evaluations",
                "approved",
                "grade",
                "without evaluations",
            ]
        ]
        dirty = self.df.assign(**{c: "=UNTRUSTED()" for c in forbidden}, row_id=999)
        checked = p.validate(dirty)
        self.assertEqual(set(checked.ignored), set(forbidden + ["row_id"]))
        self.assertEqual(list(checked.inputs), p.FEATURES)
        pre = p.load_pipeline().named_steps["pre"]
        used = [
            c
            for name, trans, cols in pre.transformers_
            if name != "remainder"
            for c in cols
        ]
        self.assertEqual(set(used), set(p.FEATURES))
        self.assertTrue(set(used).isdisjoint(forbidden))
        np.testing.assert_array_equal(p.predict(self.df), p.predict(dirty))
        self.assertNotIn(b"UNTRUSTED", p.export_csv(p.review(checked, count=2)))

    def test_decision_parity_all_capacities_and_ties(self):
        for df in [self.df, pd.concat([self.df.iloc[:1]] * 21, ignore_index=True)]:
            v = p.validate(df)
            for share in [0, 0.05, 0.1, 0.2, 1]:
                result = p.review(v, share=share)
                k = int(np.floor(share * len(df)))
                ids = select_top(result, "dropout_probability", k)
                np.testing.assert_array_equal(
                    result.loc[result.selected_for_review, "row_id"], ids
                )
                self.assertEqual(result.selected_for_review.sum(), k)
                self.assertEqual(
                    result.row_id.tolist(),
                    select_top(result, "dropout_probability", len(df)).tolist(),
                )
        tied = pd.DataFrame({"row_id": [1, 2, 3, 4], "p": [0.2] * 4})
        self.assertEqual(
            select_top(tied, "p", 3).tolist(),
            select_top(tied.iloc[::-1], "p", 3).tolist(),
        )

    def test_capacity_edge_cases(self):
        self.assertEqual(p.capacity_size(19, share=0.05), 0)
        self.assertEqual(p.capacity_size(21, share=0.1), 2)
        for kwargs in [
            {"count": -1},
            {"count": 21},
            {"count": 1.1},
            {"share": 1.1},
            {"share": float("nan")},
            {"share": 0.1, "count": 2},
            {},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                p.capacity_size(20, **kwargs)

    def test_missing_columns_and_empty_frames(self):
        for df in [self.df.drop(columns=p.FEATURES[0]), self.df.iloc[:0]]:
            with self.assertRaises(ValueError):
                p.validate(df)

    def test_invalid_values_and_partial_acknowledgment(self):
        for col, value in [
            ("Age at enrollment", -1),
            ("Age at enrollment", 121),
            ("Age at enrollment", 19.5),
            ("Admission grade", 201),
            ("Admission grade", float("inf")),
            ("Admission grade", "no"),
            ("Admission grade", None),
            ("Application order", 10),
            ("Gender", 3),
            ("Course", 999999),
            ("Course", 9500.5),
        ]:
            with self.subTest(col=col, value=value):
                df = self.df.astype(object)
                df.loc[0, col] = value
                v = p.validate(df)
                self.assertEqual(v.rows.status.eq("rejected").sum(), 1)
                with self.assertRaises(ValueError):
                    p.review(v, share=0.1)
                result = p.review(v, share=0.1, accept_partial=True)
                self.assertEqual(len(result), 20)
                self.assertEqual(result.selected_for_review.eq(True).sum(), 1)
                self.assertTrue(
                    result.loc[result.row_id.eq(1), "dropout_probability"].isna().all()
                )
        invalid = self.df.copy()
        invalid["Age at enrollment"] = -1
        with self.assertRaises(ValueError):
            p.review(p.validate(invalid), count=0, accept_partial=True)

    def test_unusual_valid_values_warn_without_clipping(self):
        df = self.df.copy()
        df.loc[0, "Admission grade"] = 0
        df.loc[0, "Age at enrollment"] = 90
        v = p.validate(df)
        self.assertEqual(len(v.inputs), 20)
        self.assertIn("outside training range", v.rows.warnings.iloc[0])
        self.assertEqual(v.inputs.loc[1, "Admission grade"], 0)
        self.assertTrue(np.isfinite(p.predict(v.inputs)).all())

    def test_labels_and_documented_unseen_category(self):
        df = self.df.astype(object)
        for c in p.LABELS:
            df[c] = df[c].map(lambda x: p.LABELS[c][str(int(x))])
        np.testing.assert_array_equal(p.predict(df), p.predict(self.df))
        unseen = [
            (c, int(code))
            for c, d in p.LABELS.items()
            for code in d
            if int(code) not in p.MANIFEST["trained_categories"][c]
        ]
        self.assertTrue(unseen)
        c, code = unseen[0]
        df = self.df.copy()
        df.loc[0, c] = code
        v = p.validate(df)
        self.assertIn("all-zero block", v.rows.warnings.iloc[0])
        self.assertEqual(len(v.inputs), 20)
        self.assertTrue(np.isfinite(p.predict(v.inputs)).all())

    def test_csv_parser(self):
        for sep in [",", ";"]:
            raw = ("\ufeff" + self.df.to_csv(index=False, sep=sep)).encode()
            self.assertEqual(len(p.read_csv(raw)), 20)
        for raw in [
            b"",
            b"a,b\n",
            b"a,a\n1,2",
            b"a,b\n1",
            b"a,b\n\n",
            b"\xff",
            b"a" * 5_000_001,
            b'a,b\n"unterminated,2',
        ]:
            with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                p.read_csv(raw)
        dup = self.df.copy()
        dup[" Age at enrollment "] = 20
        with self.assertRaises(ValueError):
            p.validate(dup)

    def test_reproducibility_and_export(self):
        a = p.review(p.validate(self.df), share=0.2)
        p.load_pipeline.cache_clear()
        b = p.review(p.validate(self.df), share=0.2)
        self.assertEqual(p.export_csv(a), p.export_csv(b))
        restored = pd.read_csv(__import__("io").BytesIO(p.export_csv(a)))
        self.assertEqual(restored.selected_for_review.sum(), 4)
        self.assertEqual(set(restored.row_id), set(range(1, 21)))

    def test_model_integrity_guard(self):
        from unittest.mock import patch

        p.load_pipeline.cache_clear()
        with patch.dict(p.MANIFEST, {"sha256": "invalid"}):
            with self.assertRaises(ValueError):
                p.load_pipeline()
        p.load_pipeline.cache_clear()

    def test_streamlit_individual_and_demo(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(p.ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("12.7%", [m.value for m in app.metric])
        app.radio[0].set_value("Capacity-based review").run()
        app.checkbox[0].check().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[-1].value, "2")
        app.selectbox[0].set_value("20%").run()
        self.assertEqual(app.metric[-1].value, "4")
        app.radio[1].set_value("Number of students").run()
        app.number_input[0].set_value(3).run()
        self.assertEqual(app.metric[-1].value, "3")
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
