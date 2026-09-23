# 🏦 Marketing Campaign Response Modelling
### Predicting Which Customers Will Say "Yes" to a Bank's Term Deposit Offer

**🔴 Live Application:** [marketing-campaign-response-modelling-v7fxpgbwwfmwgc8xt9rwu4.streamlit.app](https://marketing-campaign-response-modelling-v7fxpgbwwfmwgc8xt9rwu4.streamlit.app/)

> Enter any customer's profile into the app and get an instant prediction of how likely they are to subscribe before your team ever picks up the phone.

---

## 📌 The Business Problem

Banks run large outbound marketing campaigns — phone calls, offers, follow-ups — to convince customers to open a **term deposit** (a type of savings product). The problem is simple to state and expensive to ignore:

- Calling **every single customer** in the database costs time, money, and staff hours.
- The vast majority of people say **"no."**
- Call centers waste effort on low-interest customers while high-interest customers may not get prioritized.
- There is no systematic way to know, *before calling*, who is actually worth calling.

**In plain terms: the bank was spending the same effort on every customer, regardless of how likely that customer was to say yes.**

This project was built to fix exactly that.

---

## 💡 The Solution (In Business Terms)

Instead of guessing, we built a system that **learns from the bank's own historical campaign data** over 41,000 past customer interactions to recognize the patterns behind who says "yes" and who says "no."

Think of it like a very experienced call center manager who has personally reviewed every past call the bank has ever made, and can now look at a *new* customer's profile and immediately say:

> *"Based on everything I've seen before, this customer has a high/medium/low chance of subscribing."*

That "manager" is a trained **Machine Learning model**. Given a customer's age, job, past contact history, and current economic conditions, it returns a **response score** a number that tells the marketing team how promising that customer is, *before* any call is made.

### What this means for the bank:
- 🎯 **Prioritize** high-scoring customers first, instead of calling in random order.
- 💰 **Reduce cost per acquisition** by not wasting calls on customers unlikely to respond.
- 📈 **Increase overall conversion rate** by focusing effort where it counts.
- 🧭 **Make data-driven decisions** instead of relying on intuition alone.

This is decision *support*, not decision *replacement* the model doesn't replace the marketing team, it tells them where to aim.

---

## 🔬 How It Was Built (The Data Science Behind It)

As the data scientist on this project, here is the process that turned raw historical data into the live tool above:

```
1. Business Understanding      → Defined the real cost/benefit problem above
2. Data Collection             → UCI Bank Marketing Dataset (41,188 customers, 20 features)
3. Data Profiling & EDA        → Understood patterns, class imbalance, correlations
4. Data Cleaning               → Removed duplicates, handled unknowns, fixed data leakage
5. Feature Engineering         → Built new signals (e.g. "was this a repeat contact?")
6. Model Building              → Trained & compared 5 different ML algorithms
7. Hyperparameter Tuning       → Optimized the best-performing models
8. Model Explainability (SHAP) → Verified WHY the model makes each decision
9. Final Evaluation            → Locked in the best, most reliable model
10. Deployment                 → Shipped it as the live Streamlit app above
```

### Models compared
| Model | Role in the project |
|---|---|
| Logistic Regression | Simple baseline for comparison |
| **Random Forest** | ✅ **Selected as the final production model** |
| XGBoost | Gradient boosting benchmark |
| LightGBM | Gradient boosting benchmark |
| CatBoost | Gradient boosting benchmark |

**Random Forest was chosen** because it gave the most reliable, generalizable results while staying interpretable enough to explain its decisions to non-technical stakeholders an important factor for a tool business teams will actually trust and use.

One important engineering decision: the **`duration` of the phone call** was deliberately excluded from the model, even though it's a strong predictor in the raw data. Call duration is only known *after* the call has already happened using it would mean the model is technically "cheating" by seeing the future. Removing it keeps the model honest and genuinely useful for its real purpose: deciding who to call *before* the call happens.

---

## 📊 Key Business Insights

These are the findings that matter most to the business, translated out of the technical analysis:

- **A customer who responded positively to a previous campaign is by far the strongest signal** that they'll say yes again — these customers should be top priority.
- **The broader economy matters.** Indicators like interest rates and consumer confidence noticeably shift how likely people are to subscribe campaign timing isn't just a marketing decision, it's an economic one.
- **Customer demographics (age, job, education) add meaningful predictive value** on top of campaign history alone.
- **Repeatedly contacting the same customer has diminishing returns** the data does not support "just call them more."
- Overall, a **targeted approach guided by this model can meaningfully cut wasted calls** compared to contacting the full customer list.

---

## 🎯 Model Performance — What the Numbers Actually Mean

| Metric | Score | What it means in plain English |
|---|---|---|
| Accuracy | 90.09% | The model's overall calls are correct 9 times out of 10 |
| Precision | 63.27% | When the model says "this customer will say yes," it's right about 63% of the time |
| Recall | 28.77% | Of all customers who would have said yes, the model successfully flags about 29% of them |
| F1 Score | 39.56% | A balanced measure combining precision and recall |
| ROC-AUC | 81.23% | The model is significantly better than random guessing at ranking customers by likelihood |

**Honest caveat, stated the way a responsible data scientist should:** this model is a **ranking and prioritization tool**, not a crystal ball. It's meaningfully better than calling customers at random, but it should be used to *sort* customers by likelihood not treated as a guaranteed yes/no answer. The displayed percentage in the app is a **model confidence score**, not a certified probability. This is disclosed directly inside the app itself, in the interest of setting the right expectations for anyone using it operationally.

---

## 🌐 Try the Live App

**👉 [Open the live app here](https://marketing-campaign-response-modelling-v7fxpgbwwfmwgc8xt9rwu4.streamlit.app/)**

No installation needed it runs directly in your browser.

**How to use it:**
1. Enter a customer's profile (age, job, marital status, education, etc.)
2. Enter their campaign contact history and current economic indicators
3. Click **"🔮 Predict Customer Response"**
4. Instantly see:
   - A **response score** (e.g. "72% likely to respond")
   - A plain-language **business interpretation** of that score
   - The raw data the model actually used, for full transparency

This is the exact same model used throughout this analysis not a simplified demo version.

---

## 🛠 For Technical Reviewers

### Project structure
```
Marketing-Campaign-Response-Modelling
├── App/                 → Streamlit application (app.py, app_logic.py)
├── data/                → Raw and processed datasets
├── models/              → Trained models + predict.py (production scoring logic)
├── notebooks/           → Full data science workflow, notebook by notebook
├── reports/             → EDA outputs, evaluation results, model card
├── src/                 → Shared feature engineering code (train/serve consistency)
├── tests/               → Unit tests for features and prediction consistency
├── requirements.txt
└── README.md
```

A key design choice worth noting for other engineers: the feature engineering logic used during **training** and the logic used during **live prediction** both call the exact same function (`src/features.py`). This eliminates train/serve skew a common, hard-to-catch bug where a model behaves differently in production than it did during training because the two paths were implemented separately.

### Tech stack
- **Language:** Python
- **Modeling:** Scikit-Learn, XGBoost, LightGBM, CatBoost
- **Data handling:** Pandas, NumPy
- **Explainability:** SHAP
- **Experiment tracking:** MLflow
- **Deployment:** Streamlit, Streamlit Community Cloud

### Run it locally
```bash
git clone https://github.com/minalsadiq/Marketing-Campaign-Response-Modelling.git
cd Marketing-Campaign-Response-Modelling
pip install -r requirements.txt
streamlit run App/app.py
```

---

## 📈 Future Roadmap

- [ ] Wrap the model in a FastAPI endpoint for system-to-system integration
- [ ] Containerize with Docker for consistent deployment across environments
- [ ] Add a CI/CD pipeline for automated testing on every code change
- [ ] Deploy to AWS/Azure for production-scale hosting
- [ ] Add probability calibration so the score reflects a true real-world probability
- [ ] Build automated model monitoring to catch performance drift over time

---

## 📚 Dataset Source

- **UCI Machine Learning Repository — Bank Marketing Dataset**
- https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

---

## 👩‍💻 Author

**Minal Sadiq**
Machine Learning • Data Science • Artificial Intelligence
GitHub: [github.com/minalsadiq](https://github.com/minalsadiq)

---

⭐ If this project is useful to you, consider giving it a star on GitHub.
