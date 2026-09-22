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

## Artifact versions

`final_model.joblib` and `final_preprocessor.joblib` were saved with
**scikit-learn 1.9.0**. The root `requirements.txt` pins scikit-learn to
that exact version for this reason -- sklearn does not guarantee that a
pickle saved by a newer version loads correctly (or at all) on an older
one. If you retrain and re-save these two files, check the scikit-learn
version in that environment and update the pin in `requirements.txt` to
match.
