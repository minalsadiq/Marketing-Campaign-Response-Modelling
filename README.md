# 📊 Marketing Campaign Response Modelling

### End-to-End Machine Learning Pipeline for Predicting Customer Subscription to Bank Marketing Campaigns

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-success?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-Classification-yellow?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blueviolet?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

</p>

---

# 🚀 Live Demo

### 🌐 Streamlit App

**https://marketing-campaign-response-modelling-5yyltmwsuofq88dgr9mwwd.streamlit.app/**

---

# 📌 Project Overview

Marketing campaigns are among the most expensive customer acquisition strategies used by banks. Contacting every customer is inefficient, costly, and often results in low conversion rates.

This project develops a complete **end-to-end Machine Learning pipeline** capable of predicting whether a customer is likely to subscribe to a term deposit after a marketing campaign.

The project covers the complete Data Science lifecycle including:

- Business Understanding
- Data Profiling
- Exploratory Data Analysis
- Data Cleaning
- Feature Engineering
- Machine Learning Model Development
- Hyperparameter Optimization
- Model Explainability
- Streamlit Deployment

---

# 🎯 Business Objective

The objective is to help banks:

- Reduce marketing costs
- Improve campaign efficiency
- Increase conversion rates
- Identify high-value customers
- Support data-driven marketing decisions

Instead of contacting every customer, the bank can target only those customers who are most likely to subscribe.

---

# 📂 Dataset

**Dataset**

UCI Bank Marketing Dataset

Source

https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

Dataset Size

- 41,188 Customers
- 20 Features
- Binary Classification

Target Variable

```
y

yes = Customer subscribed

no = Customer did not subscribe
```

---

# 🧠 Project Workflow

```
Business Understanding
        │
        ▼
Data Collection
        │
        ▼
Data Profiling
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ▼
Model Building
        │
        ▼
Hyperparameter Tuning
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
Model Explainability
        │
        ▼
Final Evaluation
        │
        ▼
Business Insights
        │
        ▼
Streamlit Deployment
```

---

# 📊 Exploratory Data Analysis

Performed analysis includes:

- Missing Value Analysis
- Target Distribution
- Class Imbalance
- Numerical Feature Analysis
- Categorical Feature Analysis
- Correlation Analysis
- Outlier Detection
- Customer Demographics
- Campaign Analysis
- Economic Indicator Analysis

---

# 🧹 Data Cleaning

The preprocessing pipeline includes:

- Removing duplicate records
- Handling unknown values
- Data type correction
- Leakage prevention
- Removing duration feature
- Missing value verification

---

# ⚙ Feature Engineering

Engineered Features

- Previous Contact Indicator
- Multiple Contacts Indicator
- Senior Citizen Indicator
- Stable Job Indicator
- Summer Contact Indicator
- Customer Cluster Feature

Preprocessing Pipeline

- StandardScaler
- OneHotEncoder
- ColumnTransformer
- Pipeline

---

# 🤖 Machine Learning Models

The following models were implemented and compared:

| Model | Purpose |
|---------|----------|
| Logistic Regression | Baseline |
| Random Forest | Ensemble |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| CatBoost | Gradient Boosting |

---

# 🔧 Hyperparameter Optimization

Optimization Techniques

- GridSearchCV
- Cross Validation
- ROC-AUC Optimization

Experiment tracking performed using **MLflow**.

---

# 📈 Model Evaluation

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

---

# 🏆 Final Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **90.09%** |
| Precision | **63.27%** |
| Recall | **28.77%** |
| F1 Score | **39.56%** |
| ROC-AUC | **81.23%** |

---

# 🔍 Model Explainability

Model interpretation was performed using:

- Feature Importance
- SHAP Values
- Global Feature Importance
- Local Prediction Explanation

---

# 💼 Business Insights

Key findings include:

- Previous campaign success is the strongest predictor.
- Economic indicators significantly affect customer response.
- Customer demographics improve prediction quality.
- Removing the duration feature prevents data leakage.
- Targeted campaigns can substantially reduce marketing costs.

---

# 🌐 Streamlit Application

A fully interactive web application was built using Streamlit.

Features

- Customer information input
- Real-time prediction
- Subscription probability
- Input data preview
- Responsive web interface

Live Demo

https://marketing-campaign-response-modelling-5yyltmwsuofq88dgr9mwwd.streamlit.app/

---

# 📁 Project Structure

```
Marketing-Campaign-Response-Modelling

│
├── App
│   └── app.py
│
├── data
├── models
├── notebooks
├── reports
├── src
├── Images
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠 Technologies Used

Programming

- Python

Machine Learning

- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost

Data Analysis

- Pandas
- NumPy

Visualization

- Matplotlib
- Plotly

Experiment Tracking

- MLflow

Explainability

- SHAP

Deployment

- Streamlit
- Streamlit Community Cloud

Version Control

- Git
- GitHub

Development

- VS Code

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/minalsadiq/Marketing-Campaign-Response-Modelling.git
```

Move into the project

```bash
cd Marketing-Campaign-Response-Modelling
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Run the Application

Launch Streamlit

```bash
streamlit run App/app.py
```

---

# 📈 Future Improvements

- FastAPI Deployment
- Docker Support
- CI/CD Pipeline
- AWS Deployment
- Azure Deployment
- Real-Time Prediction API
- Automated Model Retraining
- Model Monitoring

---

# 📚 References

- UCI Machine Learning Repository
- Scikit-Learn Documentation
- XGBoost Documentation
- LightGBM Documentation
- CatBoost Documentation
- SHAP Documentation
- MLflow Documentation

---

# 👩‍💻 Author

## Minal Sadiq

Machine Learning • Data Science • Artificial Intelligence

GitHub

https://github.com/minalsadiq

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

Your support is greatly appreciated.
