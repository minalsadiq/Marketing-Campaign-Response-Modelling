"""
Unit tests for src/features.py -- the shared feature-engineering function
used by both Notebook 03 (training) and models/predict.py (scoring).

Run with: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import engineer_features, PDAYS_SENTINEL, EDUCATION_ORDER


def _sample_row(**overrides):
    row = {
        "age": 41,
        "job": "technician",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "wed",
        "duration": 187,
        "campaign": 2,
        "pdays": PDAYS_SENTINEL,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 92.9,
        "cons.conf.idx": -33.0,
        "euribor3m": 1.0,
        "nr.employed": 5099.1,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_drops_leakage_column_by_default():
    out = engineer_features(_sample_row())
    assert "duration" not in out.columns


def test_leakage_drop_is_safe_when_column_absent():
    df = _sample_row().drop(columns=["duration"])
    out = engineer_features(df)  # should not raise
    assert "duration" not in out.columns


def test_pdays_sentinel_becomes_nan():
    out = engineer_features(_sample_row(pdays=PDAYS_SENTINEL))
    assert np.isnan(out["pdays_known_days"].iloc[0])
    assert "pdays" not in out.columns


def test_pdays_real_value_is_preserved():
    out = engineer_features(_sample_row(pdays=6))
    assert out["pdays_known_days"].iloc[0] == 6.0


def test_contacted_before_matches_previous():
    assert engineer_features(_sample_row(previous=0))["contacted_before"].iloc[0] == 0
    assert engineer_features(_sample_row(previous=3))["contacted_before"].iloc[0] == 1


def test_month_cyclical_encoding_wraps_dec_to_jan():
    dec = engineer_features(_sample_row(month="dec"))
    jan = engineer_features(_sample_row(month="jan"))
    # December and January should be close on the cycle (adjacent months),
    # much closer than e.g. December and June (opposite side of the cycle).
    dec_jan_dist = (dec["month_sin"].iloc[0] - jan["month_sin"].iloc[0]) ** 2 + \
                   (dec["month_cos"].iloc[0] - jan["month_cos"].iloc[0]) ** 2
    jun = engineer_features(_sample_row(month="jun"))
    dec_jun_dist = (dec["month_sin"].iloc[0] - jun["month_sin"].iloc[0]) ** 2 + \
                   (dec["month_cos"].iloc[0] - jun["month_cos"].iloc[0]) ** 2
    assert dec_jan_dist < dec_jun_dist


def test_education_ordinal_respects_declared_order():
    out = engineer_features(_sample_row(education="illiterate"))
    assert out["education_ordinal"].iloc[0] == EDUCATION_ORDER["illiterate"]
    out2 = engineer_features(_sample_row(education="university.degree"))
    assert out2["education_ordinal"].iloc[0] == EDUCATION_ORDER["university.degree"]
    assert out2["education_ordinal"].iloc[0] > out["education_ordinal"].iloc[0]


@pytest.mark.parametrize("age,expected_group", [
    (18, "<30"),
    (29, "<30"),
    (30, "30-39"),
    (45, "40-49"),
    (59, "50-59"),
    (60, "60+"),
    (90, "60+"),
])
def test_age_group_boundaries(age, expected_group):
    out = engineer_features(_sample_row(age=age))
    assert str(out["age_group"].iloc[0]) == expected_group


def test_output_has_no_raw_source_columns():
    out = engineer_features(_sample_row())
    for col in ("pdays", "month", "day_of_week", "education", "duration"):
        assert col not in out.columns
