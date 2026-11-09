"""Local Streamlit entry point. Start from the project root using README commands."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import pandas as pd
import streamlit as st

from student_dropout.production import (
    FEATURES,
    LABELS,
    MANIFEST,
    ROOT,
    export_csv,
    load_pipeline,
    read_csv,
    review,
    validate,
)


def show_individual() -> None:
    """Render enrolment form and retain the last submitted prediction."""
    st.subheader("One student, information at enrolment")
    st.caption(
        "Initial values are an invented demonstration case. Replace every field for a new case."
    )
    example = pd.read_csv(ROOT / "examples/synthetic_students.csv").iloc[0]
    with st.form("individual"):
        values = {}
        cols = st.columns(3)
        for i, feature in enumerate(FEATURES):
            with cols[i % 3]:
                name = "Nationality" if feature == "Nacionality" else feature
                if feature in LABELS:
                    options = list(LABELS[feature])
                    default = str(int(example[feature]))
                    values[feature] = int(
                        st.selectbox(
                            name,
                            options,
                            index=options.index(default),
                            format_func=lambda code, feature=feature: (
                                f"{LABELS[feature][code]} [{code}]"
                            ),
                        )
                    )
                else:
                    values[feature] = st.number_input(
                        name,
                        value=float(example[feature]),
                        step=1.0
                        if feature in ["Age at enrollment", "Application order"]
                        else 0.1,
                    )
        submitted = st.form_submit_button("Estimate probability", type="primary")
    if submitted:
        validation = validate(pd.DataFrame([values]))
        st.session_state["individual_result"] = validation
    if "individual_result" in st.session_state:
        validation = st.session_state["individual_result"]
        if validation.rows.status.eq("rejected").any():
            st.error(validation.rows.errors.iloc[0])
        else:
            result = review(validation, count=0)
            if validation.rows.warnings.iloc[0]:
                st.warning(validation.rows.warnings.iloc[0])
            st.metric(
                "Estimated probability of recorded Dropout",
                f"{100 * result.dropout_probability.iloc[0]:.1f}%",
            )
            st.caption(
                "Result describes the last submitted form. Submit again after editing. No individual uncertainty interval is estimated."
            )


def show_cohort() -> None:
    """Render CSV validation, capacity controls and ranked downloads."""
    st.subheader("A review list within your capacity")
    a, b = st.columns(2)
    a.download_button(
        "Download CSV template",
        (ROOT / "examples/input_template.csv").read_bytes(),
        "input_template.csv",
        "text/csv",
    )
    b.download_button(
        "Download synthetic example",
        (ROOT / "examples/synthetic_students.csv").read_bytes(),
        "synthetic_students.csv",
        "text/csv",
    )
    st.caption(
        "UTF-8 CSV, comma or semicolon; 18 required fields; codes or exact dictionary labels. Up to 10,000 records / 5 MB. No names or student IDs needed."
    )
    upload = st.file_uploader("Upload student CSV", type=["csv"])
    demo = st.checkbox(
        "Use the synthetic demonstration (20 invented students)", value=False
    )
    payload = (
        upload.getvalue()
        if upload is not None
        else ((ROOT / "examples/synthetic_students.csv").read_bytes() if demo else None)
    )
    if payload is not None:
        if upload is None:
            st.warning(
                "SYNTHETIC DEMONSTRATION: these are invented cases, not real students."
            )
        try:
            validation = validate(read_csv(payload))
            if validation.ignored:
                st.warning(
                    "Ignored extra columns (never model inputs): "
                    + ", ".join(validation.ignored)
                )
            valid_count = len(validation.inputs)
            rejected = len(validation.rows) - valid_count
            a, b, feature = st.columns(3)
            a.metric("Loaded records", len(validation.rows))
            b.metric("Valid records", valid_count)
            feature.metric("Rejected records", rejected)
            st.caption(
                "row_id is the 1-based data-record position in this CSV, excluding its header. Keep the original file/order to match results. Row IDs are never predictors."
            )
            if rejected:
                st.error(
                    f"{rejected} records rejected. They are retained in the downloadable result without a probability or selection decision."
                )
                st.dataframe(
                    validation.rows[validation.rows.status.eq("rejected")],
                    hide_index=True,
                )
            warning_rows = validation.rows[validation.rows.warnings.ne("")]
            if len(warning_rows):
                st.warning("Review warnings before using these predictions.")
                st.dataframe(warning_rows, hide_index=True)
            accepted = (
                st.checkbox(
                    f"I acknowledge the exclusions; calculate capacity using only the {valid_count} valid records",
                    value=False,
                )
                if rejected
                else True
            )
            method = st.radio(
                "Capacity unit", ["Percentage", "Number of students"], horizontal=True
            )
            if method == "Percentage":
                preset = st.selectbox(
                    "Review capacity", ["5%", "10%", "20%", "Custom"], index=1
                )
                percent = (
                    st.number_input(
                        "Custom percentage", min_value=0.0, max_value=100.0, value=10.0
                    )
                    if preset == "Custom"
                    else float(preset.rstrip("%"))
                )
                kwargs = {"share": percent / 100}
            else:
                kwargs = {
                    "count": int(
                        st.number_input(
                            "Number to review",
                            min_value=0,
                            max_value=valid_count,
                            value=min(2, valid_count),
                            step=1,
                        )
                    )
                }
            st.caption(
                "Percentage capacity = floor(share × valid records). Equal risks: SHA256(1481:row_id), then row ID. Reordering an uploaded CSV can change tied selections."
            )
            if accepted and valid_count:
                result = review(validation, accept_partial=accepted, **kwargs)
                st.metric(
                    "Review list size", int(result.selected_for_review.eq(True).sum())
                )
                st.dataframe(
                    result,
                    hide_index=True,
                    column_config={
                        "dropout_probability": st.column_config.NumberColumn(
                            "Dropout probability", format="%.3f"
                        )
                    },
                )
                st.download_button(
                    "Download review results",
                    export_csv(result),
                    "review_results.csv",
                    "text/csv",
                    type="primary",
                )
                st.caption(
                    "Sorted by risk, with rejected records last. No observed outcomes are supplied: precision, recall, lift and identified Dropout counts cannot be calculated for this upload."
                )
            elif not valid_count:
                st.error("No valid records. Correct the CSV and upload again.")
        except ValueError as exc:
            st.error(str(exc))


def main() -> None:
    """Load the frozen model and display the selected local workflow."""
    st.set_page_config(
        page_title="Student Review Studio", page_icon="🎓", layout="wide"
    )
    st.title("Student Review Studio")
    st.caption("Enrolment-time estimates • Academic prototype • Human review required")
    st.info(
        "Estimate the probability of recorded Dropout by the end of normal course duration. This is a point estimate, not a confidence interval or a guarantee."
    )
    try:
        load_pipeline()
    except Exception as exc:
        st.error(f"Trusted model could not be loaded: {exc}")
        st.stop()
    st.caption(
        f"Frozen model: {MANIFEST['version']} | P4 shallow gradient boosting | Predictions made at enrolment"
    )
    with st.expander("Meaning, limitations and input policy"):
        st.write(
            "Dropout = 1; Graduate or Enrolled = 0 at the recorded horizon. Enrolled does not mean eventual graduation. Transfer to other cohorts or institutions is unvalidated. Historical full-sample EDA indirectly explored the reserved test cohort; no test performance is reported here."
        )
        st.write(
            "This model does not estimate tutoring/advising effects. Changing a form field does not establish causality. A review list is not a list of students guaranteed to benefit from support. Do not use this academic prototype for automatic decisions."
        )
        st.write(
            "All 18 fields must be known at enrolment. Missing or undocumented categories are rejected. Valid numbers outside the training range receive warnings; documented categories absent from training use the fitted all-zero encoding with a warning. Age 1–120 is a conservative input guard, not an eligibility rule. Grades must be 0–200, preference order 0–9. Extra columns are ignored explicitly and never enter the model."
        )
        st.markdown(
            "Category labels: [UCI dataset 697](https://doi.org/10.24432/C5MC89), CC BY 4.0. Use only the source coding scheme; do not guess mappings for another institution."
        )
    mode = st.radio(
        "Workflow", ["Individual prediction", "Capacity-based review"], horizontal=True
    )
    if mode == "Individual prediction":
        show_individual()
    else:
        show_cohort()


if __name__ == "__main__":
    main()
