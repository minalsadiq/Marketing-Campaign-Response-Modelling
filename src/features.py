# src/features.py
"""
Feature engineering shared by notebook 03 (training) and models/predict.py (scoring).
Mirrors the transformations documented in 03_Data_Cleaning_FeatureEngineering.ipynb,
Sections 2.3 (pdays) and 3.1-3.3 (calendar, education, age).

Keeping this logic in one file means training and production scoring can never
silently drift apart -- both import and call the same function.
"""
import numpy as np
import pandas as pd

PDAYS_SENTINEL = 999

MONTH_MAP = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
             "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
DAY_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}

EDUCATION_ORDER = {
    "illiterate": 0, "basic.4y": 1, "basic.6y": 2, "basic.9y": 3,
    "high.school": 4, "professional.course": 5, "university.degree": 6,
}

AGE_BINS = [0, 30, 40, 50, 60, 100]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60+"]


def engineer_features(df: pd.DataFrame, leakage_cols=("duration",)) -> pd.DataFrame:
    """
    Apply the fixed, non-data-fitted transformations decided in Notebook 03.

    Deliberately excluded (these stay in notebook 03 only, since they're
    training-dataset-only steps that don't apply to scoring a single new
    customer): dropping duplicate rows, and the leakage-correlation sanity
    check. Anything data-fitted (imputation, one-hot vocab, scaling, PCA)
    is also excluded on purpose -- that's fit on the training split only,
    downstream of this function.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with the original column names (job, marital, education,
        pdays, month, day_of_week, age, previous, etc.).
    leakage_cols : iterable of str
        Columns known only after the outcome (e.g. "duration") to drop.
        Silently skips any not present, so calling this on already-cleaned
        data is safe.

    Returns
    -------
    pd.DataFrame
        Copy of df with engineered columns added and their raw source
        columns removed.
    """
    df = df.copy()

    # --- Drop leakage column(s) known only after the call ends ---
    present_leakage = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=present_leakage)

    # --- Section 2.3: pdays sentinel ---
    # previous>0 is the more reliable prior-contact signal (100% agreement
    # with poutcome; Notebook 01 v2 Section 7) -- more reliable than pdays.
    df["contacted_before"] = (df["previous"] > 0).astype(int)

    # Real pdays values are sparse (3.7% of rows); keep as a supplementary
    # numeric column with the 999 sentinel neutralised to NaN. Imputation
    # of the NaNs happens on the training split only, downstream.
    df["pdays_known_days"] = df["pdays"].replace(PDAYS_SENTINEL, np.nan)
    df = df.drop(columns=["pdays"])

    # --- Section 3.1: calendar cyclical encoding ---
    # month/day_of_week are points on a cycle (Dec is adjacent to Jan);
    # sin/cos is a fixed formula, safe to compute before any split.
    df["month_num"] = df["month"].map(MONTH_MAP)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    df["day_of_week_num"] = df["day_of_week"].map(DAY_MAP)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week_num"] / 5)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week_num"] / 5)

    df = df.drop(columns=["month_num", "day_of_week_num", "month", "day_of_week"])

    # --- Section 3.2: education ordinal encoding ---
    # Genuine order (illiterate < ... < university degree), unlike job/marital.
    # "unknown" -> NaN; imputed on the training split only, downstream.
    df["education_ordinal"] = df["education"].map(EDUCATION_ORDER)
    df = df.drop(columns=["education"])

    # --- Section 3.3: age buckets ---
    # Fixed banking-segment cutoffs decided in advance, offered as a
    # candidate feature for feature selection to keep or drop.
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)

    return df
