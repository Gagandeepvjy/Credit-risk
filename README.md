# Fintech Credit Risk Platform

End-to-end machine learning platform for credit default risk prediction on LendingClub-style loan data. Covers the full pipeline from raw data ingestion through feature engineering, model training with hyperparameter tuning, explainability, and business impact reporting.

## What It Does

1. Loads a Kaggle LendingClub CSV and maps loan outcomes to binary default labels.
2. Cleans numeric and categorical fields (outlier capping, null imputation, normalisation).
3. Encodes credit categories with a reusable training-time encoder that handles unseen labels at inference.
4. Engineers domain-specific features: loan-to-income ratio, utilisation flags, FICO risk tiers, delinquency flags, and expected-loss drivers.
5. Splits data into stratified train, validation, and test sets.
6. Trains a logistic regression baseline and an XGBoost model tuned with Optuna (up to 100 trials).
7. Evaluates with ROC-AUC, PR-AUC, Brier score, KS statistic, Gini coefficient, and full classification metrics.
8. Generates ROC, precision-recall, calibration, and SHAP summary plots.
9. Produces a business impact report: approval rate, expected portfolio loss, and bad-loan rate in approved loans.
10. Saves model and preprocessing artifacts for downstream use.

## Project Layout

```text
configs/config.yaml          Training and business settings
src/data/ingestion.py        Data loading, cleaning, encoding, splitting
src/features/engineering.py  Domain feature engineering
src/models/train.py          Model training and persistence
src/evaluation/metrics.py    Metrics, plots, SHAP explainability, expected loss
src/pipeline.py              End-to-end training pipeline entry point
tests/test_pipeline.py       Unit and integration tests
```

Generated outputs:

```text
models/
  xgboost_model.pkl
  categorical_encoder.pkl
  feature_engineer.pkl

reports/
  roc_curve.png
  pr_curve.png
  calibration.png
  shap_summary.png
  metrics.json
  business_impact.json
```

## Setup

Use Python 3.11 or 3.12.

```bash
cd ~/Desktop/fintech-credit-risk-platform
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

Recommended Kaggle dataset:

```
All Lending Club loan data
https://www.kaggle.com/datasets/wordsforthewise/lending-club
```

After downloading, place the CSV at:

```
data/raw/lending_club.csv
```

Loan status mapping:

```
Fully Paid                          -> 0 (non-default)
Charged Off / Default / Late 31-120 -> 1 (default)
Current / Issued / ambiguous        -> excluded
```

## Run The Pipeline

Quick smoke run (small sample, fewer Optuna trials):

```bash
python -m src.pipeline --quick
```

Full run using `configs/config.yaml`:

```bash
python -m src.pipeline
```

Custom data path:

```bash
python -m src.pipeline --data-path path/to/your_file.csv
```

After training, check `models/` for saved artifacts and `reports/` for evaluation plots and JSON summaries.

## Run Tests

```bash
pytest -q
```

## Preprocessing & Inference Chain

```
raw loan application
  -> clean_data
  -> CreditCategoricalEncoder
  -> CreditFeatureEngineer
  -> XGBoost model
  -> probability of default
  -> risk tier + approve/decline decision + expected loss
```

Decision threshold: `0.35` (configurable in `configs/config.yaml`)

Expected loss formula:

```
EL = probability of default × loss given default × loan amount
```

Default LGD assumption: `0.45`

## Experiment Tracking

MLflow is used to log parameters, metrics, and model artifacts during training. To view the MLflow UI:

```bash
mlflow ui --backend-store-uri mlruns
```

Then open `http://localhost:5000` in your browser.
