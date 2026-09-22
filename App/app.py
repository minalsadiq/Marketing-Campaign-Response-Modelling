"""
Streamlit frontend for the Bank Marketing Response Prediction Model.

Business workflow:
    1. Enter one customer's campaign/profile information.
    2. The app applies the same feature engineering used during training.
    3. The saved preprocessing pipeline transforms the customer.
    4. The final Random Forest model predicts subscription probability.
    5. The result is displayed as an easy-to-understand business decision-support output.

Model artifacts expected in ../models/:
    - final_model.joblib
    - final_preprocessor.joblib
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Make project root / models directory importable
# --------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from predict import score_batch

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Response Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Custom styling
# --------------------------------------------------------------------------

PRIMARY = "#0F3D5C"
ACCENT = "#C7962C"
SUCCESS = "#16803C"
DANGER = "#B42318"
BG = "#F7F8FA"
TEXT = "#243746"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #FFFFFF;
        }}

        h1, h2, h3 {{
            color: {PRIMARY};
            font-family: "Georgia", "Times New Roman", serif;
        }}

        .subtitle {{
            color: #5B6B79;
            font-size: 1rem;
            margin-top: -0.5rem;
            margin-bottom: 1.5rem;
        }}

        .result-card {{
            background: {BG};
            border: 1px solid #E1E6EA;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            margin-top: 1rem;
        }}

        .probability {{
            font-size: 3rem;
            font-weight: 700;
            color: {PRIMARY};
            line-height: 1.1;
        }}

        .result-title {{
            font-size: 1.35rem;
            font-weight: 700;
            color: {TEXT};
            margin-top: 0.5rem;
        }}

        .result-description {{
            color: #5B6B79;
            font-size: 0.95rem;
            margin-top: 0.5rem;
        }}

        .info-box {{
            background: #F7F8FA;
            border: 1px solid #E1E6EA;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.75rem 0;
        }}

        .warning-box {{
            background: #FFF8EC;
            border: 1px solid #EED9A8;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            color: #6B5A2A;
        }}

        div[data-testid="stMetric"] {{
            background-color: {BG};
            border: 1px solid #E1E6EA;
            border-left: 4px solid {ACCENT};
            border-radius: 8px;
            padding: 0.9rem;
        }}

        .stButton > button {{
            background-color: {PRIMARY};
            color: white;
            border-radius: 7px;
            border: none;
            font-weight: 600;
            padding: 0.65rem 1rem;
        }}

        .stButton > button:hover {{
            background-color: #0A2C42;
            color: white;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {BG};
        }}

        footer {{
            visibility: hidden;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

DAY_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
}


def engineer_customer_features(
    age: int,
    job: str,
    marital: str,
    education: str,
    default: str,
    housing: str,
    loan: str,
    contact: str,
    month: str,
    day_of_week: str,
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
    Create the exact raw feature-engineering contract expected by
    final_preprocessor.joblib.

    This reproduces the deterministic feature engineering from Notebook 03.
    """

    # ----------------------------------------------------------------------
    # Calendar features
    # ----------------------------------------------------------------------

    month_num = MONTH_MAP[month]

    month_sin = np.sin(2 * np.pi * month_num / 12)
    month_cos = np.cos(2 * np.pi * month_num / 12)

    day_num = DAY_MAP[day_of_week]

    day_of_week_sin = np.sin(2 * np.pi * day_num / 5)
    day_of_week_cos = np.cos(2 * np.pi * day_num / 5)

    # ----------------------------------------------------------------------
    # Education ordinal
    # ----------------------------------------------------------------------

    education_order = {
        "illiterate": 0,
        "basic.4y": 1,
        "basic.6y": 2,
        "basic.9y": 3,
        "high.school": 4,
        "professional.course": 5,
        "university.degree": 6,
        "unknown": np.nan,
    }

    education_ordinal = education_order[education]

    # ----------------------------------------------------------------------
    # Age group
    # Same bins as Notebook 03
    # [0, 30, 40, 50, 60, 100]
    # ----------------------------------------------------------------------

    if age < 30:
        age_group = "<30"
    elif age < 40:
        age_group = "30-39"
    elif age < 50:
        age_group = "40-49"
    elif age < 60:
        age_group = "50-59"
    else:
        age_group = "60+"

    # ----------------------------------------------------------------------
    # Previous contact features
    # ----------------------------------------------------------------------

    contacted_before = int(previous > 0)

    if pdays == 999:
        pdays_known_days = np.nan
    else:
        pdays_known_days = float(pdays)

    # ----------------------------------------------------------------------
    # Exact columns expected by Notebook 04 preprocessor
    #
    # IMPORTANT:
    # duration, month, day_of_week, education and pdays are NOT passed
    # because Notebook 03 engineered/dropped them.
    # ----------------------------------------------------------------------

    row = {
        "age": age,
        "campaign": campaign,
        "previous": previous,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_of_week_sin": day_of_week_sin,
        "day_of_week_cos": day_of_week_cos,
        "pdays_known_days": pdays_known_days,
        "education_ordinal": education_ordinal,
        "contacted_before": contacted_before,
        "job": job,
        "marital": marital,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "poutcome": poutcome,
        "age_group": age_group,
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
    predicted score/probability rather than a guaranteed real-world chance.
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


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown("# 🏦 Bank Marketing Response Predictor")

st.markdown(
    """
    <p class="subtitle">
    Enter a customer's profile and campaign information to estimate their
    predicted likelihood of subscribing to the term deposit.
    </p>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Model Information")

    st.write("**Model:** Random Forest")
    st.write("**Task:** Binary Classification")
    st.write("**Target:** Term Deposit Subscription")
    st.write("**Pipeline:** Feature Engineering → Preprocessing → Random Forest")

    st.divider()

    st.markdown("### About the prediction")

    st.info(
        "The displayed percentage is the model's predicted score for the "
        "entered customer. It should be used as decision support rather "
        "than as a guaranteed probability of future behaviour."
    )

    st.divider()

    st.caption(
        "Developed as a machine-learning decision-support application "
        "for the Bank Marketing Response Modeling project."
    )


# --------------------------------------------------------------------------
# Customer information
# --------------------------------------------------------------------------

st.markdown("## Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1,
        help="Customer age in years.",
    )

    job = st.selectbox(
        "Job",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown",
        ],
    )

    marital = st.selectbox(
        "Marital Status",
        ["married", "single", "divorced", "unknown"],
    )

    education = st.selectbox(
        "Education",
        [
            "university.degree",
            "high.school",
            "professional.course",
            "basic.9y",
            "basic.6y",
            "basic.4y",
            "illiterate",
            "unknown",
        ],
    )


with col2:
    default = st.selectbox(
        "Credit Default",
        ["no", "unknown", "yes"],
        help="Whether the customer has credit in default.",
    )

    housing = st.selectbox(
        "Housing Loan",
        ["no", "yes", "unknown"],
    )

    loan = st.selectbox(
        "Personal Loan",
        ["no", "yes", "unknown"],
    )

    contact = st.selectbox(
        "Contact Communication Type",
        ["cellular", "telephone"],
    )


with col3:
    campaign = st.number_input(
        "Contacts During This Campaign",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
        help="Number of contacts performed during the current campaign.",
    )

    previous = st.number_input(
        "Previous Contacts",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Number of contacts performed before this campaign.",
    )

    pdays = st.number_input(
        "Days Since Previous Contact",
        min_value=0,
        max_value=999,
        value=999,
        step=1,
        help="999 means the customer was not previously contacted.",
    )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["nonexistent", "failure", "success", "other"],
    )


# --------------------------------------------------------------------------
# Campaign / economic information
# --------------------------------------------------------------------------

st.markdown("## Campaign & Economic Information")

col1, col2, col3 = st.columns(3)

with col1:
    month = st.selectbox(
        "Campaign Month",
        list(MONTH_MAP.keys()),
        index=4,
    )

    day_of_week = st.selectbox(
        "Campaign Day",
        list(DAY_MAP.keys()),
        index=2,
    )

with col2:
    emp_var_rate = st.number_input(
        "Employment Variation Rate",
        value=-1.8,
        format="%.3f",
        help="emp.var.rate",
    )

    cons_price_idx = st.number_input(
        "Consumer Price Index",
        value=93.444,
        format="%.3f",
        help="cons.price.idx",
    )

    cons_conf_idx = st.number_input(
        "Consumer Confidence Index",
        value=-36.1,
        format="%.3f",
        help="cons.conf.idx",
    )

with col3:
    euribor3m = st.number_input(
        "Euribor 3 Month Rate",
        value=1.313,
        format="%.3f",
        help="euribor3m",
    )

    nr_employed = st.number_input(
        "Number of Employees",
        value=5099.1,
        format="%.1f",
        help="nr.employed",
    )


# --------------------------------------------------------------------------
# Prediction button
# --------------------------------------------------------------------------

st.write("")

predict_button = st.button(
    "🔮 Predict Customer Response",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

if predict_button:

    try:
        customer_df = engineer_customer_features(
            age=age,
            job=job,
            marital=marital,
            education=education,
            default=default,
            housing=housing,
            loan=loan,
            contact=contact,
            month=month,
            day_of_week=day_of_week,
            campaign=campaign,
            previous=previous,
            pdays=pdays,
            poutcome=poutcome,
            emp_var_rate=emp_var_rate,
            cons_price_idx=cons_price_idx,
            cons_conf_idx=cons_conf_idx,
            euribor3m=euribor3m,
            nr_employed=nr_employed,
        )

        with st.spinner("Analyzing customer with the trained model..."):
            probability = predict_customer(customer_df)

        percentage = probability * 100

        label, description = probability_label(probability)

        st.divider()

        st.markdown("## Prediction Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Predicted Response Score",
                f"{percentage:.1f}%",
            )

        with r2:
            st.metric(
                "Model Prediction",
                (
                    "Likely to Respond"
                    if probability >= 0.50
                    else "Less Likely to Respond"
                ),
            )

        with r3:
            st.metric(
                "Previous Contact",
                "Yes" if previous > 0 else "No",
            )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="probability">{percentage:.1f}%</div>
                <div class="result-title">{label}</div>
                <div class="result-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(max(probability, 0.0), 1.0))

        st.write("")

        # ------------------------------------------------------------------
        # Business interpretation
        # ------------------------------------------------------------------

        st.markdown("### Business Interpretation")

        if probability >= 0.70:
            st.success(
                "This customer receives a high predicted response score. "
                "The model ranks this customer relatively strongly for "
                "potential campaign response."
            )

        elif probability >= 0.40:
            st.warning(
                "This customer receives a moderate predicted response score. "
                "The model identifies some response potential, but the score "
                "is not in the highest range."
            )

        else:
            st.info(
                "This customer receives a lower predicted response score. "
                "The model ranks this customer below the higher-scoring group."
            )

        # ------------------------------------------------------------------
        # Generated feature summary
        # ------------------------------------------------------------------

        with st.expander("View engineered customer features"):
            engineered_display = customer_df.T.reset_index()
            engineered_display.columns = ["Feature", "Value"]

            st.dataframe(
                engineered_display,
                use_container_width=True,
                hide_index=True,
            )

        # ------------------------------------------------------------------
        # Model caveat
        # ------------------------------------------------------------------

        st.markdown(
            """
            <div class="warning-box">
            <strong>Important:</strong> This model was not probability-calibrated.
            Therefore, the percentage should primarily be interpreted as a
            model response score/ranking signal rather than a guaranteed
            real-world probability. The project's evaluation also documented
            performance variation across different time periods.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except FileNotFoundError as exc:
        st.error(
            "Model files could not be found. Make sure "
            "`final_model.joblib` and `final_preprocessor.joblib` are available "
            "in the configured models directory."
        )
        st.exception(exc)

    except Exception as exc:
        st.error("The customer could not be scored.")
        st.exception(exc)


# --------------------------------------------------------------------------
# Initial screen
# --------------------------------------------------------------------------

else:
    st.markdown(
        """
        <div class="info-box">
        <strong>How to use:</strong><br><br>
        1. Enter the customer's information above.<br>
        2. Enter the campaign and economic information.<br>
        3. Click <strong>Predict Customer Response</strong>.<br>
        4. The trained machine-learning model will return the customer's
        predicted response score.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    c1.metric("Prediction Type", "Binary Classification")
    c2.metric("Model", "Random Forest")
    c3.metric("Output", "Response Score")
