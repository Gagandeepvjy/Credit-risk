# Fintech Credit Risk Platform

End-to-end credit default risk platform for Kaggle LendingClub-style loan applications. It trains a baseline logistic regression model, tunes an XGBoost model, produces evaluation reports, saves model artifacts, and serves real-time predictions through FastAPI.

## What It Does

1. Loads a Kaggle LendingClub CSV.
2. Cleans numeric and categorical fields.
3. Encodes credit categories with a reusable training-time encoder.
4. Adds credit risk features such as loan-to-income, utilization flags, FICO risk tier, and expected-loss drivers.
5. Splits the data into train, validation, and test sets.
6. Trains logistic regression and XGBoost models.
7. Evaluates model quality with ROC-AUC, PR-AUC, Brier score, KS statistic, Gini, and classification metrics.
8. Creates ROC, precision-recall, calibration, SHAP, metrics, and business-impact reports.
9. Saves the model and preprocessing artifacts for API inference.
10. Serves single and batch prediction endpoints.

## Project Layout

```text
configs/config.yaml          Training, business, and API settings
src/data/ingestion.py        Kaggle data loading, cleaning, encoding, splitting
src/features/engineering.py  Domain feature engineering
src/models/train.py          Model training and persistence
src/evaluation/metrics.py    Metrics, plots, SHAP, expected loss
src/api/serve.py             FastAPI prediction service
src/pipeline.py              End-to-end training pipeline
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

Use Python 3.11 or 3.12. Some ML packages may not yet publish wheels for newer Python versions.

```bash
cd ~/Desktop/fintech-credit-risk-platform
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `python3.11` is not available, install Python 3.11 first, then rerun the setup commands.

## Run Tests

```bash
pytest -q
```

## Dataset

Recommended Kaggle dataset:

```text
All Lending Club loan data
https://www.kaggle.com/datasets/wordsforthewise/lending-club
```

After downloading it, place the main loan CSV at:

```text
~/Desktop/fintech-credit-risk-platform/data/raw/lending_club.csv
```

The Kaggle workflow maps final loan outcomes into the model target:

```text
Fully Paid -> 0
Charged Off / Default / Late (31-120 days) -> 1
Current / Issued / ambiguous statuses -> excluded
```

## Train The Model

Fast smoke run:

```bash
python -m src.pipeline --quick
```

Full run from `configs/config.yaml`:

```bash
python -m src.pipeline
```

If your CSV has a different filename or location:

```bash
python -m src.pipeline --data-path path/to/your_lending_club_file.csv --quick
```

The full run uses more data and more Optuna tuning trials, so it takes longer. After training, check `models/` for artifacts and `reports/` for evaluation outputs.

## Start The API

Train the model first so `models/xgboost_model.pkl`, `models/categorical_encoder.pkl`, and `models/feature_engineer.pkl` exist.

```bash
uvicorn src.api.serve:app --host 0.0.0.0 --port 8000 --reload
```

Open the API docs:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

Example prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 10000,
    "term": 36,
    "int_rate": 12.5,
    "installment": 350,
    "grade": "B",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "annual_inc": 60000,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "dti": 15.0,
    "delinq_2yrs": 0,
    "fico_range_low": 700,
    "open_acc": 8,
    "pub_rec": 0,
    "revol_bal": 5000,
    "revol_util": 30.0,
    "total_acc": 20,
    "mort_acc": 1,
    "pub_rec_bankruptcies": 0
  }'
```

## Docker

Train locally first so the `models/` directory contains artifacts, then run:

```bash
cd docker
docker compose up --build
```

API:

```text
http://localhost:8000/docs
```

MLflow:

```text
http://localhost:5000
```

## Workflow

Training and serving use the same preprocessing chain:

```text
raw application data
  -> clean_data
  -> CreditCategoricalEncoder
  -> CreditFeatureEngineer
  -> XGBoost model
  -> probability of default
  -> risk tier, approve/decline decision, expected loss
```

The decision threshold is currently `0.35`, and expected loss is calculated as:

```text
expected loss = probability of default * loss given default * loan amount
```

The default loss-given-default assumption is `0.45`.
