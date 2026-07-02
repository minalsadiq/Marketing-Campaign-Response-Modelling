# 📊 Marketing Campaign Response Modelling
### End-to-End Machine Learning Pipeline for Predicting Customer Subscription to Bank Marketing Campaigns

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-success?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-Classification-yellow?style=for-the-badge)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

</p>

---

# 📌 Project Overview

Marketing campaigns are one of the most expensive customer acquisition strategies used by financial institutions. Contacting every customer is inefficient, costly, and often results in very low conversion rates.

This project develops a complete **end-to-end Machine Learning pipeline** capable of predicting whether a customer is likely to subscribe to a term deposit after a marketing campaign.

The solution follows the complete Data Science lifecycle including:

- Business Understanding
- Data Profiling
- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Model Development
- Hyperparameter Optimization
- Model Explainability
- Business Recommendations

The project was implemented using modern Machine Learning practices with reproducible pipelines and experiment tracking.

---

# 🎯 Business Objective

The objective is to build a predictive classification model that helps banks:

- Reduce marketing costs
- Improve campaign efficiency
- Increase conversion rates
- Identify high-potential customers
- Support data-driven marketing decisions

Instead of contacting every customer, the bank can focus only on customers with high subscription probability.

---

# 📂 Dataset Information

**Dataset:** UCI Bank Marketing Dataset

**Source:**
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

Dataset Characteristics

- 41,188 customer records
- 20 original features
- Binary Classification
- Portuguese banking institution marketing campaigns

Target Variable

```
y

Yes = Customer subscribed

No = Customer did not subscribe
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
Baseline Models
          │
          ▼
Advanced Models
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
```

---

# 📚 Exploratory Data Analysis

Extensive exploratory analysis was performed including:

✔ Missing Value Analysis

✔ Target Variable Distribution

✔ Class Imbalance Analysis

✔ Numerical Feature Analysis

✔ Categorical Feature Analysis

✔ Correlation Analysis

✔ Outlier Detection

✔ Customer Demographic Analysis

✔ Campaign Performance Analysis

✔ Economic Indicator Analysis

---

# 🧹 Data Cleaning

The preprocessing stage includes:

- Handling unknown values
- Removing duplicate records
- Feature standardization
- Data type correction
- Leakage prevention
- Removal of duration feature
- Missing value verification

---

# ⚙ Feature Engineering

Feature engineering was performed to improve predictive performance.

Created features include:

- Previous Contact Indicator
- Multiple Contacts Indicator
- Senior Citizen Indicator
- Stable Job Indicator
- Summer Contact Indicator
- Customer Cluster Feature

Additional preprocessing includes:

- Standard Scaling
- One-Hot Encoding
- Column Transformer Pipeline

---

# 🤖 Machine Learning Models

The following models were implemented and evaluated.

| Model | Purpose |
|---------|----------|
| Logistic Regression | Baseline Model |
| Random Forest | Tree Ensemble |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| CatBoost | Gradient Boosting |
| Hyperparameter Tuned Models | Performance Optimization |

---

# 🔧 Hyperparameter Tuning

Model optimization was performed using:

- Grid Search
- Cross Validation
- ROC-AUC Optimization

The tuning process was tracked using **MLflow**.

---

# 📊 Model Evaluation

Each model was evaluated using multiple classification metrics.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

---

# 🏆 Best Performing Model

After comparing all candidate models, the best-performing model was selected based on overall predictive performance and ROC-AUC score.

> Replace the values below with your final evaluation metrics.

| Metric | Score |
|---------|-------|
| Accuracy | 90.09% |
| Precision | 63.27% |
| Recall | 28.77% |
| F1 Score | 39.56% |
| ROC-AUC | 81.23% |

---

# 🔍 Model Explainability

Model interpretation was performed using:

- Feature Importance
- SHAP Values
- Global Feature Importance
- Local Prediction Interpretation

These methods improve transparency and explain model decisions.

---

# 💼 Business Insights

Key findings from the project include:

- Previous campaign success strongly influences future customer response.
- Economic indicators significantly affect customer subscription behaviour.
- Customer demographics contribute to marketing effectiveness.
- Data-driven targeting can substantially reduce marketing costs.
- Removing the "duration" feature prevents data leakage and produces a realistic predictive model.

---

# 📁 Project Structure

```
Marketing-Campaign-Response-Modelling

│
├── data
│
├── models
│
├── notebooks
│   ├── 01 Data Profiling
│   ├── 02 EDA & Cleaning
│   ├── 03 Feature Engineering
│   ├── 04 Model Building
│   ├── 05 Hyperparameter Tuning
│   ├── 06 Model Explainability
│   └── 07 Final Evaluation
│
├── reports
│
├── src
│
├── requirements.txt
│
├── README.md
│
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

Model Explainability

- SHAP

Model Serialization

- Joblib

Development

- VS Code
- Git
- GitHub

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/minalsadiq/Marketing-Campaign-Response-Modelling.git
```

Move into project

```bash
cd Marketing-Campaign-Response-Modelling
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

Launch Jupyter Notebook

```bash
jupyter notebook
```

or execute Python scripts

```bash
python src/train_model.py
```

---

# 📈 Future Improvements

Possible future enhancements include:

- Streamlit Web Application
- FastAPI Deployment
- Docker Support
- CI/CD Pipeline
- Cloud Deployment (AWS/Azure/GCP)
- Automated Model Retraining
- Model Monitoring
- Real-Time Predictions

---

# 📖 References

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

Machine Learning | Data Science | Artificial Intelligence

GitHub:
https://github.com/minalsadiq

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

Your support is greatly appreciated.
