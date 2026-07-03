import streamlit as st
import pandas as pd
import joblib

# ===================================
# Page Config
# ===================================

st.set_page_config(
    page_title="Marketing Campaign Response Predictor",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Marketing Campaign Response Predictor")

st.write(
    "Predict whether a customer is likely to respond to a term deposit."
)

# ===================================
# Load Model (Pipeline)
# ===================================

model = joblib.load("models/final_model.pkl")

# ===================================
# Inputs
# ===================================

st.header("Customer Information")

age = st.number_input("Age",18,95,35)

balance = st.number_input(
    "Balance",
    min_value=-5000,
    max_value=100000,
    value=1000
)

job = st.selectbox(
    "Job",
    [
        "admin.","blue-collar","entrepreneur",
        "housemaid","management","retired",
        "self-employed","services","student",
        "technician","unemployed","unknown"
    ]
)

marital = st.selectbox(
    "Marital",
    ["married","single","divorced"]
)

education = st.selectbox(
    "Education",
    ["primary","secondary","tertiary","unknown"]
)

default = st.selectbox(
    "Default",
    ["yes","no"]
)

housing = st.selectbox(
    "Housing Loan",
    ["yes","no"]
)

loan = st.selectbox(
    "Personal Loan",
    ["yes","no"]
)

contact = st.selectbox(
    "Contact Type",
    ["cellular","telephone","unknown"]
)

month = st.selectbox(
    "Month",
    [
        "jan","feb","mar","apr","may","jun",
        "jul","aug","sep","oct","nov","dec"
    ]
)

day_of_week = st.selectbox(
    "Day of Week",
    ["mon","tue","wed","thu","fri"]
)

campaign = st.slider(
    "Campaign Contacts",
    1,
    50,
    2
)

pdays = st.number_input(
    "Days Since Previous Contact",
    -1,
    999,
    -1
)

previous = st.slider(
    "Previous Contacts",
    0,
    50,
    0
)

poutcome = st.selectbox(
    "Previous Outcome",
    ["success","failure","other","unknown"]
)

emp_var_rate = st.number_input(
    "Employment Variation Rate",
    value=1.1
)

cons_price_idx = st.number_input(
    "Consumer Price Index",
    value=93.99
)

cons_conf_idx = st.number_input(
    "Consumer Confidence Index",
    value=-36.4
)

euribor3m = st.number_input(
    "Euribor 3M",
    value=4.85
)

nr_employed = st.number_input(
    "Number of Employees",
    value=5191.0
)

# ===================================
# Predict
# ===================================

if st.button("Predict"):

    data = pd.DataFrame({

        "age":[age],
        "job":[job],
        "marital":[marital],
        "education":[education],
        "default":[default],
        "balance":[balance],
        "housing":[housing],
        "loan":[loan],
        "contact":[contact],
        "month":[month],
        "campaign":[campaign],
        "pdays":[pdays],
        "previous":[previous],
        "poutcome":[poutcome],
        "emp.var.rate":[emp_var_rate],
        "cons.price.idx":[cons_price_idx],
        "cons.conf.idx":[cons_conf_idx],
        "euribor3m":[euribor3m],
        "nr.employed":[nr_employed],
        "day_of_week":[day_of_week]

    })

    # ===== Feature Engineering =====

    data["cluster"] = 0
    data["stable_job"] = data["job"].isin(
        ["management","technician","admin."]
    ).astype(int)

    data["multiple_contacts"] = (data["campaign"] > 3).astype(int)

    data["previous_contact"] = (data["previous"] > 0).astype(int)

    data["is_senior"] = (data["age"] >= 60).astype(int)

    data["summer_contact"] = data["month"].isin(
        ["jun","jul","aug"]
    ).astype(int)

    # ===================================

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    st.divider()

    st.subheader("Prediction")

    if prediction == 1:

        st.success(
            f"Customer is likely to respond.\n\nProbability: {probability:.2%}"
        )

    else:

        st.error(
            f"Customer is unlikely to respond.\n\nProbability: {probability:.2%}"
        )

    st.progress(float(probability))

    st.subheader("Input Data")

    st.dataframe(data)
