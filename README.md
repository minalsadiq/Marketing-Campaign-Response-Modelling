# Bank Marketing Response Scoring

End-to-end ML project predicting which bank customers are likely to
subscribe to a term deposit, built as nine notebooks (data profiling
through deployment) plus a small deployable scoring package.

## Project structure

```
notebooks/   01-09, in order: profiling -> EDA -> cleaning/feature
             engineering -> split -> feature selection -> baseline
             training -> tuning -> evaluation -> deployment
src/         features.py (shared feature engineering) and plotting.py
             (shared figure-saving), imported by both the notebooks and
             the deployment package -- one implementation, not two.
models/      predict.py (scoring library/CLI) + the two artifacts it
             loads (final_model.joblib, final_preprocessor.joblib).
             See models/README.md for what's tracked in git and why.
App/         app.py (Streamlit UI) + app_logic.py (its pure/testable
             logic -- form-input handling and the predict_customer call).
data/        raw/ and processed/ -- not committed; see data/README.md.
reports/     JSON/CSV artifacts from every notebook: profiling stats,
             EDA summaries, split config, tuning config, evaluation
             results, model_card.json, monitoring_plan.json.
tests/       pytest suite covering feature engineering and the
             train/serve consistency between predict.py and the app.
```

## How scoring works

`predict.score_batch(raw_df)` is the one place raw customer data becomes
a prediction:

```
raw customer data  -->  src.features.engineer_features()  -->
saved preprocessor.transform()  -->  final_model.predict_proba()
```

`raw_df` uses the **original raw schema** -- the same column names and
values as `data/raw/bank-additional-full.csv` minus `y` (and minus
`duration`, which is dropped automatically since it's a leakage feature
only known after a call ends). Both `App/app_logic.py` (the Streamlit
form) and `models/predict.py`'s CLI call the exact same
`engineer_features` function from `src/features.py` -- there is a single
implementation of the feature engineering, used everywhere it's needed,
so training and production scoring can't silently drift apart.

## Setup

```bash
pip install -r requirements.txt
```

Make sure `final_model.joblib` and `final_preprocessor.joblib` are
present in `models/` (see `models/README.md` if they aren't).

## Run the frontend

```bash
streamlit run App/app.py
```

## Use the library/CLI directly

```bash
python models/predict.py customers.csv --capacity-pct 15
```
```python
from predict import score_batch
call_list = score_batch(new_customers_df, capacity_pct=15)
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

CI (`.github/workflows/tests.yml`) runs this suite on every push/PR.

## Known limitations -- read before using this in a real decision

Full detail lives in `reports/model_card.json`; the headlines:

- **Modest ranking signal, not a highly accurate classifier.**
  Validation PR-AUC is ~0.20 (vs. ~0.11 for a random baseline) -- a real
  but limited lift, and recall at the working 15%-capacity operating
  point is ~25% (i.e. ~75% of actual subscribers are missed at that
  operating point).
- **Not probability-calibrated.** Use the score to rank/prioritize
  customers, not as a literal real-world probability.
- **Trained mostly on a low-conversion regime (2008 crisis period).**
  A chronological train/val/test split (deliberate -- see
  `reports/split_config.json`) means the test set lands almost entirely
  in a later, much higher base-rate period, so test metrics should be
  read as an upper bound, not a steady-state estimate. This is also why
  the deployment package uses a rank/percentile-based operating point
  instead of a fixed probability threshold.
- **Uneven recall across job segments** -- see
  `reports/evaluation_results.json` / Notebook 08 for the breakdown.
- **Cost-per-call and value-per-subscription figures are placeholders.**
  The 15%-capacity default is a reasonable starting point, not a
  validated business answer -- see `reports/monitoring_plan.json` and
  Notebook 09 Section 5.

See `reports/monitoring_plan.json` for the drift-monitoring and
retraining plan this project recommends before any real production use.
