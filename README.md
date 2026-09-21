# Bank Marketing Response Scoring — Deployment Package

## Files
- `predict.py` — core scoring module (library + CLI). No UI dependencies.
- `app.py` — Streamlit frontend built on top of `predict.py`.
- `requirements.txt` — dependencies for both.

## Setup
Place these three model artifacts (produced by `09_final_model_deployment.ipynb`)
in the same folder as `predict.py` and `app.py`:
- `final_preprocessor.joblib`
- `final_model.joblib`
- `final_model_config.json` (optional — enables validation metrics in the sidebar)

```bash
pip install -r requirements.txt
```

## Run the frontend
```bash
streamlit run app.py
```

## Use the library/CLI directly
```bash
python predict.py customers.csv --capacity-pct 15 --output call_list.csv
```
```python
from predict import score_batch
call_list = score_batch(new_customers_df, capacity_pct=15)
```

## Known issue this package works around
The preprocessing pipeline (fit in Notebook 04, on the full engineered
feature set) and the final model (fit in Notebook 07, after Notebook 05
dropped a near-zero-variance one-hot column) do not agree on the exact
encoded column set. `predict.py` reconciles this automatically at import
time and logs what it did — no action needed by callers, but worth fixing
at the source in `09_final_model_deployment.ipynb` by re-fitting the
preprocessor on the post-feature-selection column set, so a future model
retrain doesn't depend on this runtime patch.
