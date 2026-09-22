"""
Guards against the exact train/serve-skew bug this project found and fixed:
App/app.py used to hand-roll its own copy of the feature engineering
instead of importing src/features.py, so the two could silently drift
apart. These tests fail loudly if that duplication ever comes back, and
confirm predict.py's internal pipeline matches calling the shared
building blocks directly.

Run with: pytest tests/
Requires model artifacts to be present in models/ (final_model.joblib,
final_preprocessor.joblib) -- skipped automatically if they're not (e.g.
in a checkout that hasn't downloaded/regenerated the model files).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "App"))

_MODEL_FILES_PRESENT = (
    (PROJECT_ROOT / "models" / "final_model.joblib").exists()
    and (PROJECT_ROOT / "models" / "final_preprocessor.joblib").exists()
)

pytestmark = pytest.mark.skipif(
    not _MODEL_FILES_PRESENT,
    reason="final_model.joblib / final_preprocessor.joblib not present in models/",
)


def _raw_row(**overrides):
    row = {
        "age": 41, "job": "technician", "marital": "married",
        "education": "university.degree", "default": "no", "housing": "yes",
        "loan": "no", "contact": "cellular", "month": "may", "day_of_week": "wed",
        "campaign": 2, "previous": 1, "pdays": 6, "poutcome": "success",
        "emp.var.rate": -1.8, "cons.price.idx": 92.9, "cons.conf.idx": -33.0,
        "euribor3m": 1.0, "nr.employed": 5099.1,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_predict_imports_shared_features_module():
    """predict.py must use src.features.engineer_features, not its own copy."""
    import predict
    assert predict.engineer_features.__module__ == "src.features"


def test_app_no_longer_defines_its_own_feature_engineering():
    """
    Regression guard for the original bug: neither app.py nor app_logic.py
    may re-implement engineer_features()-equivalent logic (cyclical
    month/day encoding, education ordinal mapping, age bucketing, etc.).
    If a function like the old `engineer_customer_features` reappears,
    it's a strong signal the duplication has come back.
    """
    for fname in ("app.py", "app_logic.py"):
        source = (PROJECT_ROOT / "App" / fname).read_text()
        assert "engineer_customer_features" not in source
        assert "month_sin" not in source, f"{fname} should not compute engineered features itself"
        assert "education_order" not in source.lower() or "EDUCATION_ORDER" in source, (
            f"{fname} should import education ordering from src.features, not redefine it"
        )


def test_score_batch_matches_manual_pipeline():
    """
    score_batch(raw_df) must give the same result as manually calling
    engineer_features -> preprocessor.transform -> model.predict_proba.
    This is the direct regression test for train/serve consistency.
    """
    import joblib
    from predict import score_batch
    from src.features import engineer_features

    raw = _raw_row()

    # Path A: through predict.score_batch (what the app and CLI use)
    via_score_batch = score_batch(raw.copy())["probability"].iloc[0]

    # Path B: manual, independent reconstruction of the pipeline
    prep = joblib.load(PROJECT_ROOT / "models" / "final_preprocessor.joblib")
    model = joblib.load(PROJECT_ROOT / "models" / "final_model.joblib")
    feature_names = [n.replace("<", "lt_").replace("[", "(").replace("]", ")")
                      for n in prep.get_feature_names_out()]
    engineered = engineer_features(raw.copy())
    encoded = pd.DataFrame(prep.transform(engineered), columns=feature_names)
    via_manual = model.predict_proba(encoded)[:, 1][0]

    assert via_score_batch == pytest.approx(via_manual, rel=1e-9)


def test_build_raw_customer_row_output_is_scoreable():
    """
    App/app.py's form-input builder must produce a DataFrame that
    predict.score_batch can score without error -- i.e. the app's raw
    schema and predict.py's expected raw schema haven't drifted apart.
    """
    from app_logic import build_raw_customer_row
    from predict import score_batch

    df = build_raw_customer_row(
        age=35, job="admin.", marital="single", education="high.school",
        default="no", housing="no", loan="no", contact="cellular",
        month_label="May", day_label="Wednesday", campaign=1, previous=0,
        pdays=999, poutcome="nonexistent", emp_var_rate=-1.8,
        cons_price_idx=93.4, cons_conf_idx=-36.1, euribor3m=1.3,
        nr_employed=5099.1,
    )
    result = score_batch(df)
    assert 0.0 <= result.iloc[0]["probability"] <= 1.0
