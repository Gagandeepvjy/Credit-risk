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
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # --- Debt burden features ---
        if "loan_amnt" in df.columns and "annual_inc" in df.columns:
            df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)

        if "installment" in df.columns and "annual_inc" in df.columns:
            df["installment_to_income"] = df["installment"] / (df["annual_inc"] / 12 + 1)

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

        if "pub_rec" in df.columns:
            df["has_pub_rec"] = (df["pub_rec"] > 0).astype(int)

        if "pub_rec_bankruptcies" in df.columns:
            df["has_bankruptcy"] = (df["pub_rec_bankruptcies"] > 0).astype(int)

        # --- Account diversity ---
        if "open_acc" in df.columns and "total_acc" in df.columns:
            df["open_acc_ratio"] = df["open_acc"] / (df["total_acc"] + 1)

        # --- Interest rate risk ---
        if "int_rate" in df.columns and "dti" in df.columns:
            df["int_rate_x_dti"] = df["int_rate"] * df["dti"]

        # --- Revolving balance normalised ---
        if "revol_bal" in df.columns and "annual_inc" in df.columns:
            df["revol_bal_to_income"] = df["revol_bal"] / (df["annual_inc"] + 1)

        logger.info(f"Feature engineering complete: {df.shape[1]} total features")
        return df

    def get_feature_names_out(self, input_features=None):
        return list(self.transform(pd.DataFrame(columns=self.feature_names_in_)).columns)


def select_features(df: pd.DataFrame, target: str = "loan_status") -> list:
    """Return the list of model features (all columns except target)."""
    return [c for c in df.columns if c != target]
