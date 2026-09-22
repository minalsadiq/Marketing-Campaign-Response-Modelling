# Models

Only the artifacts `predict.py` (and therefore `App/app.py`) actually
load are tracked in git:

- `final_model.joblib`
- `final_preprocessor.joblib`

The per-algorithm experiment dumps from Notebooks 06-07
(`baseline_*.joblib`, `tuned_*.joblib`) are gitignored -- they're
reproducible by re-running those notebooks against the processed data,
and keeping every candidate model in version control doesn't add much
once `reports/evaluation_results.json` already records how each one
scored.

If you need those files (e.g. to inspect a specific baseline without
re-running training), regenerate them locally, or ask whoever ran the
original training for the artifact bundle.
