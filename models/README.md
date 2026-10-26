# Saved model artifacts

24 compressed Joblib files: `{dataset}_{split_seed}_{training_variant}.joblib`.

- Datasets: `aps`, `ai4i`.
- Seeds: `11`, `29`, `47`.
- Variants: `logistic`, `hgb`, `group_hgb`, `cell_hgb`.

Each contains the fitted estimator, feature ordering, dataset/split metadata, group ranking, validation objective values, and three threshold policies (`clean`, `environment`, `default`). Main Group-HGB results use `environment`; baseline HGB and logistic use `clean`. Other policies are ablations. No artifact was selected because it had the best test score.

Use `scripts/infer.py` from the repository root so the `src` module is importable. Models were trained with scikit-learn 1.8.0; use the pinned environment. Joblib uses Python serialization, so load only trusted artifacts. The model scores are not guaranteed calibrated probabilities. There is no deployment-readiness or future-failure prediction claim.

Six additional models in `models/exploratory/` implement the post-hoc clean-repeat control. Their results are in `results/exploratory_clean_repeat.csv`; they are not part of the frozen 24-fit primary analysis.
