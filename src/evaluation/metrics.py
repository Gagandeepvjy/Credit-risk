"""
Model evaluation: classification metrics, SHAP explainability,
calibration curves, and business-impact reporting.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Return a comprehensive metrics dictionary."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc":        roc_auc_score(y_test, proba),
        "pr_auc":         average_precision_score(y_test, proba),
        "brier_score":    brier_score_loss(y_test, proba),
        "ks_statistic":   _ks_statistic(y_test, proba),
        "gini":           2 * roc_auc_score(y_test, proba) - 1,
    }

    report = classification_report(y_test, preds, output_dict=True)
    metrics["precision_default"] = report["1"]["precision"]
    metrics["recall_default"]    = report["1"]["recall"]
    metrics["f1_default"]        = report["1"]["f1-score"]
    metrics["accuracy"]          = report["accuracy"]

    logger.info("\n" + "=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k:<25} {v:.4f}")
    logger.info("=" * 50)
    return metrics


def _ks_statistic(y_true, y_proba) -> float:
    """Kolmogorov-Smirnov statistic — standard in credit scoring."""
    df = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values("p", ascending=False)
    total_pos = y_true.sum()
    total_neg = (1 - y_true).sum()
    df["cum_pos"] = (df["y"] == 1).cumsum() / total_pos
    df["cum_neg"] = (df["y"] == 0).cumsum() / total_neg
    return (df["cum_pos"] - df["cum_neg"]).abs().max()


# ─────────────────────────────────────────────
# SHAP explainability
# ─────────────────────────────────────────────

def explain_with_shap(model, X: pd.DataFrame, max_display: int = 20,
                      output_path: str = "reports/shap_summary.png"):
    """Generate SHAP summary plot and return SHAP values."""
    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"SHAP summary saved → {output_path}")
    return shap_values


def plot_shap_waterfall(explainer, X: pd.DataFrame, idx: int = 0,
                        output_path: str = "reports/shap_waterfall.png"):
    """Individual prediction explanation."""
    shap_values = explainer(X.iloc[[idx]])
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

def plot_roc_curve(model, X_test, y_test,
                   output_path: str = "reports/roc_curve.png"):
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Credit Risk Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved → {output_path}")


def plot_pr_curve(model, X_test, y_test,
                  output_path: str = "reports/pr_curve.png"):
    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba)
    auprc = average_precision_score(y_test, proba)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR AUC = {auprc:.3f}", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"PR curve saved → {output_path}")


def plot_calibration(model, X_test, y_test,
                     output_path: str = "reports/calibration.png"):
    proba = model.predict_proba(X_test)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)

    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, fraction_pos, "s-", label="XGBoost")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Calibration curve saved → {output_path}")


# ─────────────────────────────────────────────
# Business impact — expected loss
# ─────────────────────────────────────────────

def business_impact_report(model, X_test, y_test,
                            loan_amounts: pd.Series,
                            lgd: float = 0.45,
                            threshold: float = 0.5) -> pd.DataFrame:
    """
    Compute expected loss (EL = PD × LGD × EAD) per loan
    and report portfolio-level impact at given threshold.
    """
    proba = model.predict_proba(X_test)[:, 1]
    report_df = pd.DataFrame({
        "true_default": y_test.values,
        "pd_score":     proba,
        "loan_amnt":    loan_amounts.values,
        "predicted":    (proba >= threshold).astype(int),
    })
    report_df["expected_loss"] = report_df["pd_score"] * lgd * report_df["loan_amnt"]
    report_df["approved"]      = (report_df["predicted"] == 0).astype(int)

    approved = report_df[report_df["approved"] == 1]
    summary = {
        "total_loans":            len(report_df),
        "approved_loans":         int(report_df["approved"].sum()),
        "approval_rate":          report_df["approved"].mean(),
        "expected_portfolio_loss": approved["expected_loss"].sum(),
        "avg_expected_loss":      approved["expected_loss"].mean(),
        "bad_loans_approved":     int((approved["true_default"] == 1).sum()),
        "bad_rate_in_approved":   (approved["true_default"] == 1).mean(),
    }

    logger.info("\nBusiness Impact Summary:")
    for k, v in summary.items():
        logger.info(f"  {k:<30} {v:,.2f}" if isinstance(v, float) else f"  {k:<30} {v}")

    return report_df, summary
