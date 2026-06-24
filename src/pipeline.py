"""
End-to-end training pipeline entry point.
Run: python -m src.pipeline
"""

import logging
import pickle
import json
import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data.ingestion import (
    load_lending_club_data,
    clean_data,
    CreditCategoricalEncoder,
    split_data,
)
from src.features.engineering import CreditFeatureEngineer
from src.models.train import train_xgboost, train_logistic_regression, save_model
from src.evaluation.metrics import (
    evaluate_model, explain_with_shap, plot_roc_curve,
    plot_pr_curve, plot_calibration, business_impact_report
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    config_path: str = "configs/config.yaml",
    quick: bool = False,
    data_path: str | None = None,
):
    config = load_config(config_path)
    data_config = config.get("data", {})
    model_config = config.get("model", {}).get("xgboost", {})

    logger.info("━━━ STEP 1: Data Loading & Preprocessing ━━━")
    n_samples = 5_000 if quick else data_config.get("n_samples", 50_000)
    n_trials = 3 if quick else model_config.get("n_trials", 30)
    random_state = data_config.get("random_state", 42)
    csv_path = data_path or data_config.get("raw_path", "data/raw/lending_club.csv")
    sample_size = n_samples if quick else data_config.get("sample_size")
    df_raw = load_lending_club_data(
        csv_path,
        sample_size=sample_size,
        random_state=random_state,
    )

    df = clean_data(df_raw)
    encoder = CreditCategoricalEncoder()
    df = encoder.fit_transform(df)

    logger.info("━━━ STEP 2: Feature Engineering ━━━")
    fe = CreditFeatureEngineer()
    df = fe.fit_transform(df)

    logger.info("━━━ STEP 3: Train/Val/Test Split ━━━")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        test_size=data_config.get("test_size", 0.2),
        val_size=data_config.get("val_size", 0.1),
        random_state=random_state,
    )

    logger.info("━━━ STEP 4: Baseline — Logistic Regression ━━━")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)

    logger.info("━━━ STEP 5: XGBoost with Optuna Tuning ━━━")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, n_trials=n_trials)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

    logger.info("━━━ STEP 6: Model Comparison ━━━")
    logger.info(f"  LR  ROC-AUC: {lr_metrics['roc_auc']:.4f} | PR-AUC: {lr_metrics['pr_auc']:.4f}")
    logger.info(f"  XGB ROC-AUC: {xgb_metrics['roc_auc']:.4f} | PR-AUC: {xgb_metrics['pr_auc']:.4f}")

    logger.info("━━━ STEP 7: Evaluation Plots ━━━")
    plot_roc_curve(xgb_model, X_test, y_test, str(REPORTS_DIR / "roc_curve.png"))
    plot_pr_curve(xgb_model, X_test, y_test,  str(REPORTS_DIR / "pr_curve.png"))
    plot_calibration(xgb_model, X_test, y_test, str(REPORTS_DIR / "calibration.png"))

    logger.info("━━━ STEP 8: SHAP Explainability ━━━")
    shap_sample_size = min(2000, len(X_test))
    explain_with_shap(
        xgb_model,
        X_test.sample(shap_sample_size, random_state=42),
        output_path=str(REPORTS_DIR / "shap_summary.png"),
    )

    logger.info("━━━ STEP 9: Business Impact ━━━")
    loan_amounts = df.loc[X_test.index, "loan_amnt"] if "loan_amnt" in df.columns \
                   else pd.Series([10000] * len(X_test))
    _, summary = business_impact_report(xgb_model, X_test, y_test, loan_amounts)

    logger.info("━━━ STEP 10: Save Artifacts ━━━")
    save_model(xgb_model, str(MODELS_DIR / "xgboost_model.pkl"))
    with open(MODELS_DIR / "categorical_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open(MODELS_DIR / "feature_engineer.pkl", "wb") as f:
        pickle.dump(fe, f)
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(xgb_metrics, f, indent=2)
    with open(REPORTS_DIR / "business_impact.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ Pipeline complete!")
    return xgb_metrics, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the credit risk platform.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--quick", action="store_true", help="Use a small dataset and fewer tuning trials.")
    parser.add_argument("--data-path", default=None, help="Path to the LendingClub Kaggle CSV file.")
    args = parser.parse_args()
    run_pipeline(
        config_path=args.config,
        quick=args.quick,
        data_path=args.data_path,
    )
