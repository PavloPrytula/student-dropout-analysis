"""Explicit full research reproduction; never called by the app or notebook viewer."""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from student_dropout.data import OUT
from student_dropout.decision import run_decision
from student_dropout.diagnostics import reproduce_historical_eda, run_diagnostics
from student_dropout.inference import run_inference
from student_dropout.review_diagnostics import run_review_diagnostics
from student_dropout.training import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Run expensive nested training explicitly",
    )
    args = parser.parse_args()
    if not args.retrain:
        parser.error(
            "Full reproduction requires --retrain; use notebooks to view results"
        )
    if OUT.exists() or any(Path("models").glob("*.joblib")):
        parser.error(
            "Use a separate copy; move its models/ and outputs/ aside before retraining. "
            "The frozen serving artifact must not be overwritten."
        )
    OUT.mkdir(parents=True)
    shutil.copyfile("assets/results/split.json", OUT / "split.json")
    run()
    run_inference()
    run_diagnostics()
    reproduce_historical_eda()
    run_decision()
    run_review_diagnostics()
    print("Full research reproduction finished in outputs/part2 and models/.")
    print("These research models do not replace the hash-pinned serving release.")


if __name__ == "__main__":
    main()
