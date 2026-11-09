"""Execute the documented short notebook path without rewriting saved notebooks."""

import hashlib
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "main.ipynb",
    "part1_candidates.ipynb",
    "part2_training.ipynb",
    "part3_production.ipynb",
)


def main():
    model = ROOT / "models/P4.joblib"
    before = hashlib.sha256(model.read_bytes()).hexdigest()
    for name in NOTEBOOKS:
        notebook = nbformat.read(ROOT / name, as_version=4)
        nbformat.validate(notebook)
        NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        print(f"Passed: {name}")
    assert hashlib.sha256(model.read_bytes()).hexdigest() == before
    print("Short path complete; frozen pipeline unchanged. No full training was run.")


if __name__ == "__main__":
    main()
