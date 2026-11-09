# Student dropout analysis and review

An academic solution for model development Parts I–III, with a local Streamlit
prototype. Q1 describes resolved cases; Q2 estimates three-class associations;
Q3 predicts recorded Dropout at normal course-end; Q4 compares risk-based review
with random selection at equal capacity (5%, 10%, 20%). Q4 is not causal inference.

## Install

Use **Python 3.14.2** (supported Python series: 3.14). From the repository root:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

On Windows, activate with `.venv\Scripts\activate`. Dependencies are pinned to
verified direct-package versions; transitive dependencies are resolved by pip.
The clean installation was checked on macOS; other platforms require validation.
No accounts, API keys, environment variables or external services are needed.

The original `data/data.csv` and the small frozen `models/P4.joblib` pipeline are
included. No download or training is needed. Keep the bundled scikit-learn version:
serialized pipelines are version dependent. Load only the trusted bundled artifact;
its SHA-256 is checked against `assets/model_manifest.json` before loading.

## Notebooks

```sh
python -m jupyterlab
# Execute the short path in fresh kernels without changing saved notebooks:
python scripts/check_notebooks.py
```

Open and run these notebooks in order, from the repository root:

| Notebook | Content |
|---|---|
| `main.ipynb` | Historical full-sample EDA and resolved-case Q1; exploratory population comparisons predate the current three-class Q2. |
| `part1_candidates.ipynb` | Part I: question definitions, B0/P1–P4 rationale, assumptions and expected ranking; narrative only. |
| `part2_training.ipynb` | Part II: frozen split, training-only Q1–Q4 tables, diagnostics and interpretation. Default execution reads saved results, without fitting or test scoring. |
| `part3_production.ipynb` | Part III: serving design, synthetic prediction parity, capacity selection and CSV export. |

Tables and plots are retained for reading without execution. The short path reruns
historical EDA and synthetic predictions, and displays the frozen research results;
it does **not** reproduce nested training, Q2 fitting/bootstrap or Q4 uncertainty.

For expensive full reproduction, use a **separate copy**, activate its environment,
and move its bundled `models/` directory aside (also move `outputs/` if present).
Then run `python scripts/run_part2.py --retrain`. The command refuses to overwrite
existing models/results, uses the frozen split, and executes the original nested
5 x 3 training, inference and diagnostics. Results go to `outputs/part2/` and new
models to `models/`; inspect those outputs directly. The viewing notebooks continue
to display the frozen `assets/results/` release. New models are research outputs,
not an automatic replacement for the hash-pinned serving model. Full training was
not rerun for repository preparation; no held-out performance was evaluated.

## Application and synthetic example

```sh
python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open http://127.0.0.1:8501. The individual example gives **12.7%**. In
**Capacity-based review**, upload `examples/synthetic_students.csv` or enable the
built-in demo. Its 20 invented records produce **1 / 2 / 4 selections** at
5% / 10% / 20%. Download the ranked CSV. `examples/input_template.csv` provides the
18 required headers. Category codes/labels follow the bundled source dictionary.
Missing, impossible or undocumented values are rejected; partial cohorts require
acknowledgment. Extrapolation and documented-but-untrained categories warn.
IDs are input positions; reordering tied records can change selection.

## Checks

```sh
python -m unittest discover -s tests -v
python -m ruff check src scripts tests app.py
python -m ruff format --check src scripts tests app.py
python scripts/check_notebooks.py
```

The public tests use synthetic data and bundled split indices, not the raw dataset.
They check gradients, fold containment, input timing, prediction parity (1e-12),
validation, capacities/ties, CSV exports and Streamlit interactions. CI runs style
and tests on push/pull request without secrets, training or the raw CSV. A workflow
file is included; no GitHub Actions run has been claimed.

Optional browser check, with the app already running:

```sh
python -m playwright install chromium
python scripts/smoke_production.py --output-dir outputs/browser
```

It tests actual individual submission, CSV upload, capacity changes, downloads and
partial-cohort acknowledgment. To use installed Chrome, append `--channel chrome`.

## Files

| Paths | Purpose |
|---|---|
| Four notebooks above | Assignment solutions and saved substantive results. |
| `app.py` | Streamlit interface. |
| `src/student_dropout/data.py`, `preprocessing.py`, `prediction.py`, `training.py` | Frozen schema/split, fold-local transformations, estimators and training. |
| `src/student_dropout/inference.py`, `diagnostics.py`, `review_diagnostics.py` | Q2 estimation, Q1/model diagnostics, exploratory list/subgroup interpretation. |
| `src/student_dropout/decision.py`, `production.py`, `part2.py` | Q4 selector, serving/validation/export, compatibility imports. |
| `models/P4.joblib` | Only the serving pipeline, including preprocessing. |
| `assets/model_manifest.json`, `category_labels.json`, `parity_expected.json` | Required serving contract/hash, source code dictionary and synthetic parity fixture. |
| `assets/results/summary.json`, `metrics.csv`, `split.json`, `folds.json` | Frozen candidate summaries, diagnostic metrics and exact original partition indices. |
| `assets/results/Q1_training*.csv`, `Q2_*.csv`, `Q2_diagnostics.json` | Observed descriptions, adjusted contrasts/coefficients and convergence evidence. |
| `assets/results/Q4_summary.csv`, `review_paired_summary.csv`, `review_subgroups.csv` | Frozen allocation, paired-list and subgroup results. |
| `assets/results/*.png` | Three original English diagnostic figures displayed by Part II. |
| `data/data.csv`, `data/README.md` | Immutable source dataset and attribution. |
| `examples/*.csv` | Invented students and empty input template. |
| `scripts/run_part2.py`, `check_notebooks.py`, `smoke_production.py` | Explicit full reproduction, short-path execution and real-browser checks. |
| `tests/test_production.py`, `test_methodology.py` | Portable product and research unit checks. |
| `requirements.txt`, `pyproject.toml`, `.github/workflows/ci.yml`, `.gitignore`, `.gitattributes` | Installation, style, CI, publication allowlist and preservation of raw CSV bytes. |

## Scientific and licensing limits

The 80/20 split (seed 481), nested folds, candidates B0/P1–P4 and sensitivity P1B
are preserved. All tuning uses training folds. P4 was selected by existing nested
OOF log loss; its small advantage over P2 does not establish superiority. Earlier
full-sample EDA indirectly explored the later reserved test set. Assure remains
unperformed. Enrolled is not eventual graduation and course horizons differ.
Q2 population coverage/dependence and future transport are unverified. Q4 bootstrap
intervals condition on fixed scores/folds and exclude refitting/selection uncertainty.
Identified cases are not prevented dropouts; high risk does not imply treatment
benefit. The original tutoring/advising effect is unidentified, not zero. The
numerical causal-estimate/standard-error assignment requirement remains unmet.
This is an academic review prototype, not a validated institutional decision system.

Dataset: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021).
[Predict Students' Dropout and Academic Success](https://doi.org/10.24432/C5MC89),
UCI Machine Learning Repository, **CC BY 4.0**. See `data/README.md` for details.
Dataset licensing does not license this project's code. **No software license has
been selected.**
