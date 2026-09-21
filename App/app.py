"""
Streamlit frontend for the bank marketing response-scoring model.

Run with:
    streamlit run app.py

Reads the same artifacts predict.py does (final_model.joblib,
final_preprocessor.joblib, final_model_config.json) -- this file adds a
UI on top of predict.py's score_batch()/get_model_info(); it does not
duplicate any scoring logic.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from predict import (
    MissingColumnsError,
    get_feature_importance,
    get_model_info,
    score_batch,
)

# --------------------------------------------------------------------------
# Page setup + styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Term Deposit Campaign · Response Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0F3D5C"       # deep navy
ACCENT = "#C7962C"        # muted gold
BG_CARD = "#F7F8FA"
TEXT_MUTED = "#5B6B79"

st.markdown(f"""
<style>
    .stApp {{ background-color: #FFFFFF; }}
    h1, h2, h3 {{ color: {PRIMARY}; font-family: "Georgia", "Times New Roman", serif; }}
    .subtitle {{ color: {TEXT_MUTED}; font-size: 0.95rem; margin-top: -0.6rem; }}
    div[data-testid="stMetric"] {{
        background-color: {BG_CARD};
        border: 1px solid #E4E8EC;
        border-left: 4px solid {ACCENT};
        border-radius: 6px;
        padding: 0.9rem 1rem 0.6rem 1rem;
    }}
    div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; font-weight: 500; }}
    div[data-testid="stMetricValue"] {{ color: {PRIMARY}; }}
    .stButton>button {{
        background-color: {PRIMARY}; color: white; border-radius: 4px; border: none;
        font-weight: 500;
    }}
    .stButton>button:hover {{ background-color: #0A2C42; color: white; }}
    .caveat-box {{
        background-color: #FFF8EC; border: 1px solid #EED9A8; border-radius: 6px;
        padding: 0.8rem 1rem; font-size: 0.88rem; color: #6B5A2A;
    }}
    section[data-testid="stSidebar"] {{ background-color: {BG_CARD}; }}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar -- data input + operating point
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<h3 style='margin-bottom:0'>Response Scoring</h3>", unsafe_allow_html=True)
    st.caption("Term Deposit Marketing Campaign")
    st.divider()

    info = get_model_info()
    st.markdown("**Model**")
    st.write(f"{info.get('model_type', 'Unknown')} "
             f"({info.get('n_estimators', '?')} trees, depth {info.get('max_depth', '?')})")
    if "validation_pr_auc" in info:
        st.write(f"Validation PR-AUC: **{info['validation_pr_auc']:.3f}**")
    if "validation_lift_at_10pct" in info:
        st.write(f"Lift@10%: **{info['validation_lift_at_10pct']:.2f}x**")
    st.divider()

    uploaded_file = st.file_uploader("Upload customer data (CSV)", type=["csv"])

    st.markdown("**Operating point**")
    mode = st.radio(
        "Cutoff mode",
        ["Call-centre capacity (top K%)", "Fixed probability threshold"],
        help="Capacity-based is the recommended default -- see the caveat below.",
    )
    if mode == "Call-centre capacity (top K%)":
        capacity_pct = st.slider("Capacity: top % of customers to call", 1, 100, 15)
        min_probability = None
    else:
        capacity_pct = None
        min_probability = st.slider("Minimum predicted probability", 0.0, 1.0, 0.30, 0.01)

    st.markdown(
        "<div class='caveat-box'>Capacity-based cutoffs are more stable across time periods "
        "than a fixed probability threshold -- the base subscription rate has been observed to "
        "shift substantially between periods in backtesting. Use a fixed threshold only once "
        "confirmed cost-per-call / value-per-subscription figures are available.</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    run_button = st.button("Score customers", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown("# Term Deposit Campaign — Response Scoring")
st.markdown("<p class='subtitle'>Rank customers by predicted subscription probability "
            "and generate a prioritised call list.</p>", unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------------------
# Main flow
# --------------------------------------------------------------------------
if uploaded_file is None:
    st.info("Upload a customer CSV in the sidebar to get started. The file must contain the "
            "same feature-engineered columns the model was trained on "
            "(see predict.py's module docstring for the exact input contract).")
    st.stop()

raw_df = pd.read_csv(uploaded_file)

if not run_button:
    st.write(f"Loaded **{len(raw_df):,}** rows. Set the operating point in the sidebar and "
             "click **Score customers**.")
    st.dataframe(raw_df.head(10), use_container_width=True)
    st.stop()

try:
    with st.spinner("Scoring customers..."):
        call_list = score_batch(raw_df, capacity_pct=capacity_pct, min_probability=min_probability)
        full_ranked = score_batch(raw_df)  # unfiltered, for the distribution chart
except MissingColumnsError as exc:
    st.error(f"**Input is missing required columns.**\n\n{exc}")
    st.stop()
except ValueError as exc:
    st.error(f"**Could not score this file.**\n\n{exc}")
    st.stop()

# --------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers scored", f"{len(raw_df):,}")
k2.metric("Recommended calls", f"{len(call_list):,}",
          f"{len(call_list) / len(raw_df):.1%} of file")
k3.metric("Avg. probability (called)", f"{call_list['probability'].mean():.1%}")
k4.metric("Avg. probability (all)", f"{full_ranked['probability'].mean():.1%}")

st.write("")
tab_list, tab_dist, tab_decile, tab_importance = st.tabs(
    ["📋 Call List", "📊 Score Distribution", "🔟 Decile Breakdown", "🔍 Feature Importance"]
)

# --------------------------------------------------------------------------
# Tab 1 -- Call list
# --------------------------------------------------------------------------
with tab_list:
    st.markdown(f"**{len(call_list):,} customers** selected under the current operating point.")
    display_cols = [c for c in call_list.columns if c not in ("decile",)]
    st.dataframe(
        call_list[display_cols].style.format({"probability": "{:.1%}"}),
        use_container_width=True,
        height=420,
    )
    st.download_button(
        "⬇ Download call list (CSV)",
        data=call_list.to_csv(index=False).encode("utf-8"),
        file_name="call_list.csv",
        mime="text/csv",
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# Tab 2 -- Score distribution
# --------------------------------------------------------------------------
with tab_dist:
    st.markdown("Predicted probability across the full uploaded file, with the current "
                 "cutoff marked.")
    cutoff_value = (
        call_list["probability"].min() if len(call_list) else full_ranked["probability"].median()
    )
    hist = alt.Chart(full_ranked).mark_bar(color=PRIMARY, opacity=0.85).encode(
        x=alt.X("probability:Q", bin=alt.Bin(maxbins=40), title="Predicted probability"),
        y=alt.Y("count()", title="Number of customers"),
    )
    rule = alt.Chart(pd.DataFrame({"cutoff": [cutoff_value]})).mark_rule(
        color=ACCENT, strokeWidth=2, strokeDash=[6, 3]
    ).encode(x="cutoff:Q")
    st.altair_chart((hist + rule).properties(height=360), use_container_width=True)
    st.caption(f"Dashed line: current cutoff ({cutoff_value:.1%}).")

# --------------------------------------------------------------------------
# Tab 3 -- Decile breakdown
# --------------------------------------------------------------------------
with tab_decile:
    st.markdown("Customers split into ten equal-sized groups by predicted probability "
                 "(decile 1 = highest-scoring).")
    decile_summary = (
        full_ranked.groupby("decile")
        .agg(customers=("probability", "size"), avg_probability=("probability", "mean"))
        .reset_index()
    )
    bar = alt.Chart(decile_summary).mark_bar(color=PRIMARY).encode(
        x=alt.X("decile:O", title="Decile (1 = highest scoring)"),
        y=alt.Y("avg_probability:Q", title="Average predicted probability", axis=alt.Axis(format="%")),
        tooltip=["decile", "customers", alt.Tooltip("avg_probability:Q", format=".1%")],
    ).properties(height=360)
    st.altair_chart(bar, use_container_width=True)
    st.dataframe(
        decile_summary.style.format({"avg_probability": "{:.1%}"}),
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# Tab 4 -- Feature importance
# --------------------------------------------------------------------------
with tab_importance:
    importance_df = get_feature_importance(top_n=15)
    if importance_df.empty:
        st.info(f"{info.get('model_type', 'This model')} does not expose native feature "
                "importances.")
    else:
        chart = alt.Chart(importance_df).mark_bar(color=ACCENT).encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")],
        ).properties(height=420)
        st.altair_chart(chart, use_container_width=True)
        st.caption("Native model importance, for a quick read. For a full explanation of "
                    "individual predictions (SHAP), see Notebook 08.")

st.write("")
st.caption("Internal decision-support tool. Not a substitute for compliance, fair-lending, "
           "or regulatory review before use in an active campaign.")
