# experiments/

Preserved, rerunnable code behind `notebooks/10_Robustness_Reliability_Backtest.ipynb` and
`reports/{balancing_comparison,bootstrap_confidence_intervals,rolling_origin_backtest,
model_improvement_experiments,breakeven_sensitivity_table}.json`.

- `smote_impl.py` -- from-scratch SMOTE (no `imbalanced-learn` in this sandbox: no network
  access to install it). Swap for `imblearn.over_sampling.SMOTE` if available; same algorithm.

Everything else that produced those reports/ files is inlined directly in notebook 10's code
cells (re-verified to reproduce byte-for-byte identical output before being committed here --
see the notebook's own text for that check). This file exists specifically so "the code that
produced these results" is never again only-in-someone's-terminal-history -- see
reports/model_improvement_experiments.json's stated purpose for why that mattered.
