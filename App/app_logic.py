"""
Pure, Streamlit-free logic for the Bank Marketing Response Predictor app.

Kept separate from app.py so it can be imported and unit-tested without
needing a running Streamlit script context (st.* calls outside
`streamlit run` don't have a ScriptRunContext and are awkward to test
around). app.py imports from this module; this module never imports
streamlit.
"""
import sys
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from predict import score_batch  # noqa: E402

# Display labels (full names, friendly for a form) mapped to the raw
# abbreviations the bank's export -- and therefore src.features.MONTH_MAP /
# DAY_MAP -- actually uses. This is a UI label lookup only; it is NOT a
# re-implementation of any feature-engineering math (that all lives in
# src/features.py, called by predict.score_batch).
MONTH_LABEL_TO_RAW = {
    "January": "jan", "February": "feb", "March": "mar", "April": "apr",
    "May": "may", "June": "jun", "July": "jul", "August": "aug",
    "September": "sep", "October": "oct", "November": "nov", "December": "dec",
}

DAY_LABEL_TO_RAW = {
    "Monday": "mon", "Tuesday": "tue", "Wednesday": "wed",
    "Thursday": "thu", "Friday": "fri",
}


def build_raw_customer_row(
    age: int,
    job: str,
    marital: str,
    education: str,
    default: str,
    housing: str,
    loan: str,
    contact: str,
    month_label: str,
    day_label: str,
    campaign: int,
    previous: int,
    pdays: int,
    poutcome: str,
    emp_var_rate: float,
    cons_price_idx: float,
    cons_conf_idx: float,
    euribor3m: float,
    nr_employed: float,
) -> pd.DataFrame:
    """
    Build a one-row DataFrame in the RAW schema `predict.score_batch`
    expects -- i.e. exactly what a fresh export from the bank's systems
    would look like (same column names/values as
    data/raw/bank-additional-full.csv, minus `duration` and `y`).

    No feature engineering happens here. `score_batch` calls
    `src.features.engineer_features` internally, which is the same
    function Notebook 03 used to build the training data.
    """
    row = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": MONTH_LABEL_TO_RAW[month_label],
        "day_of_week": DAY_LABEL_TO_RAW[day_label],
        "campaign": campaign,
        "previous": previous,
        "pdays": pdays,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
    }

    return pd.DataFrame([row])


def predict_customer(customer_df: pd.DataFrame) -> float:
    """Run the saved deployment pipeline and return probability."""

    result = score_batch(customer_df)

    if result.empty:
        raise ValueError("The model returned no prediction.")

    return float(result.iloc[0]["probability"])


def probability_label(probability: float) -> tuple[str, str]:
    """
    Business-friendly interpretation.

    NOTE:
    The model is not probability-calibrated, so this is presented as a
    predicted score/ranking signal rather than a guaranteed real-world
    chance.
    """

    if probability >= 0.70:
        return (
            "High predicted likelihood",
            "The model assigns this customer a relatively high response score.",
        )

    if probability >= 0.40:
        return (
            "Moderate predicted likelihood",
            "The model assigns this customer a moderate response score.",
        )

    if probability >= 0.20:
        return (
            "Lower predicted likelihood",
            "The model assigns this customer a relatively low response score.",
        )

    return (
        "Very low predicted likelihood",
        "The model assigns this customer a low response score.",
    )
