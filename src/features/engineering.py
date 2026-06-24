"""
Feature engineering pipeline for credit risk modeling.
Builds domain-driven features from cleaned loan data.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that creates domain-specific
    credit risk features.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        transformed = self._transform(X)
        self.feature_names_out_ = list(transformed.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._transform(X)

        for col in getattr(self, "feature_names_out_", df.columns):
            if col not in df.columns:
                df[col] = 0
        df = df[getattr(self, "feature_names_out_", df.columns)]

        logger.info(f"Feature engineering complete: {df.shape[1]} total features")
        return df

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if "issue_d" in df.columns:
            issue_dt = pd.to_datetime(df["issue_d"], errors="coerce")
            df["issue_year"] = issue_dt.dt.year.fillna(issue_dt.dt.year.median()).fillna(0)
            df["issue_month"] = issue_dt.dt.month.fillna(0)
            df = df.drop(columns=["issue_d"])

        if "earliest_cr_line" in df.columns:
            earliest_dt = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
            reference_year = df["issue_year"] if "issue_year" in df.columns else pd.Timestamp.today().year
            df["credit_hist_years"] = (reference_year - earliest_dt.dt.year).clip(lower=0)
            df["credit_hist_years"] = df["credit_hist_years"].fillna(df["credit_hist_years"].median()).fillna(0)
            df = df.drop(columns=["earliest_cr_line"])

        # --- Debt burden features ---
        if "loan_amnt" in df.columns and "annual_inc" in df.columns:
            df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)

        if "installment" in df.columns and "annual_inc" in df.columns:
            df["installment_to_income"] = df["installment"] / (df["annual_inc"] / 12 + 1)
            df["income_to_inst"] = (df["annual_inc"] / 12 + 1) / (df["installment"] + 1)
            df["monthly_debt_strain"] = df["installment"] / (df["annual_inc"] / 12 + 1)

        if "annual_inc_joint" in df.columns and "annual_inc" in df.columns:
            df["joint_income_lift"] = (df["annual_inc_joint"].fillna(0) - df["annual_inc"]).clip(lower=0)

        # --- Credit utilisation risk ---
        if "revol_util" in df.columns:
            df["high_util_flag"] = (df["revol_util"] > 75).astype(int)
            df["util_bucket"] = pd.cut(
                df["revol_util"], bins=[0, 30, 60, 80, 100],
                labels=[0, 1, 2, 3], include_lowest=True
            ).astype(float)

        # --- FICO risk tier ---
        if "fico_range_low" in df.columns:
            df["fico_risk_tier"] = pd.cut(
                df["fico_range_low"],
                bins=[0, 620, 660, 700, 740, 780, 900],
                labels=[5, 4, 3, 2, 1, 0], include_lowest=True
            ).astype(float)

        # --- Delinquency & public record flags ---
        if "delinq_2yrs" in df.columns:
            df["has_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)

        if "inq_last_6mths" in df.columns:
            df["recent_inquiry_flag"] = (df["inq_last_6mths"] > 0).astype(int)

        if "pub_rec" in df.columns:
            df["has_pub_rec"] = (df["pub_rec"] > 0).astype(int)

        if "pub_rec_bankruptcies" in df.columns:
            df["has_bankruptcy"] = (df["pub_rec_bankruptcies"] > 0).astype(int)

        if "acc_now_delinq" in df.columns:
            df["currently_delinquent"] = (df["acc_now_delinq"] > 0).astype(int)

        if "collections_12_mths_ex_med" in df.columns:
            df["recent_collection_flag"] = (df["collections_12_mths_ex_med"] > 0).astype(int)

        # --- Account diversity ---
        if "open_acc" in df.columns and "total_acc" in df.columns:
            df["open_acc_ratio"] = df["open_acc"] / (df["total_acc"] + 1)

        # --- Interest rate risk ---
        if "int_rate" in df.columns and "dti" in df.columns:
            df["int_rate_x_dti"] = df["int_rate"] * df["dti"]

        if "dti" in df.columns and "fico_range_low" in df.columns:
            df["dti_fico_multiplier"] = df["dti"] * (850 - df["fico_range_low"]) / 100
            df["macro_fico_stress"] = df["int_rate"] * (850 - df["fico_range_low"]) / 100 \
                if "int_rate" in df.columns else (850 - df["fico_range_low"]) / 100

        # --- Revolving balance normalised ---
        if "revol_bal" in df.columns and "annual_inc" in df.columns:
            df["revol_bal_to_income"] = df["revol_bal"] / (df["annual_inc"] + 1)

        if "revol_bal" in df.columns and "total_rev_hi_lim" in df.columns:
            df["revol_bal_to_limit"] = df["revol_bal"] / (df["total_rev_hi_lim"] + 1)

        if "revol_util" in df.columns and "credit_hist_years" in df.columns:
            df["util_history_strain"] = df["revol_util"] / (df["credit_hist_years"] + 1)

        risk_parts = []
        for col in ["high_util_flag", "has_delinquency", "has_pub_rec", "has_bankruptcy", "recent_inquiry_flag"]:
            if col in df.columns:
                risk_parts.append(df[col])
        if risk_parts:
            df["high_risk_signal_count"] = sum(risk_parts)

        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df

    def get_feature_names_out(self, input_features=None):
        return list(getattr(self, "feature_names_out_", []))


def select_features(df: pd.DataFrame, target: str = "loan_status") -> list:
    """Return the list of model features (all columns except target)."""
    return [c for c in df.columns if c != target]
