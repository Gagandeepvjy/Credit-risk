"""
Data ingestion and preprocessing pipeline.
Handles loading raw loan/credit data, cleaning, and splitting.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MODEL_COLUMNS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "emp_title",
    "home_ownership",
    "annual_inc",
    "annual_inc_joint",
    "verification_status",
    "purpose",
    "zip_code",
    "addr_state",
    "application_type",
    "issue_d",
    "earliest_cr_line",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "fico_range_low",
    "open_acc",
    "acc_now_delinq",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_rev_hi_lim",
    "total_acc",
    "mort_acc",
    "collections_12_mths_ex_med",
    "pub_rec_bankruptcies",
    "loan_status",
]

DEFAULT_STATUSES = {
    "charged off",
    "default",
    "late (31-120 days)",
    "does not meet the credit policy. status:charged off",
}

NON_DEFAULT_STATUSES = {
    "fully paid",
    "does not meet the credit policy. status:fully paid",
}


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw credit data from CSV."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def prepare_lending_club_dataframe(
    df: pd.DataFrame,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Convert a Kaggle LendingClub-style CSV into this project's model schema.

    Ambiguous active loans such as Current, Issued, or In Grace Period are excluded
    because they do not have a final repayment/default outcome yet.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "loan_status" not in df.columns:
        raise ValueError("Kaggle/LendingClub data must include a 'loan_status' column.")

    status = df["loan_status"].astype(str).str.strip().str.lower()
    df = df[status.isin(DEFAULT_STATUSES | NON_DEFAULT_STATUSES)].copy()
    status = df["loan_status"].astype(str).str.strip().str.lower()
    df["loan_status"] = status.isin(DEFAULT_STATUSES).astype(int)

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state)

    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)")[0]

    for col in ["int_rate", "revol_util"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "", regex=False)

    for col in ["issue_d", "earliest_cr_line"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in MODEL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = [
        "loan_amnt",
        "term",
        "int_rate",
        "installment",
        "annual_inc",
        "annual_inc_joint",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "fico_range_low",
        "open_acc",
        "acc_now_delinq",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_rev_hi_lim",
        "total_acc",
        "mort_acc",
        "collections_12_mths_ex_med",
        "pub_rec_bankruptcies",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[MODEL_COLUMNS].copy()
    logger.info(
        "Prepared LendingClub data: %s rows | Default rate: %.2f%%",
        len(df),
        df["loan_status"].mean() * 100,
    )
    return df


def load_lending_club_data(
    filepath: str,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load and prepare Kaggle LendingClub loan data."""
    return prepare_lending_club_dataframe(
        load_raw_data(filepath),
        sample_size=sample_size,
        random_state=random_state,
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize raw credit data."""
    df = df.copy()

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Dropped {before - len(df)} duplicate rows")

    # Cap extreme outliers
    for col in ["annual_inc", "revol_bal", "installment"]:
        if col in df.columns:
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper)

    # Clip revol_util to 0-100
    if "revol_util" in df.columns:
        df["revol_util"] = df["revol_util"].clip(0, 100)

    # Fill missing numerics with median
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill missing categoricals with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        mode = df[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "UNKNOWN"
        df[col] = df[col].fillna(fill_value)

    logger.info("Data cleaning complete")
    return df


class CreditCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Sklearn-compatible encoder for credit application categorical fields."""

    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    emp_map = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
        "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
        "8 years": 8, "9 years": 9, "10+ years": 10
    }
    nominal_columns = [
        "home_ownership",
        "verification_status",
        "purpose",
        "sub_grade",
        "addr_state",
        "application_type",
    ]
    frequency_columns = ["zip_code", "emp_title"]

    def fit(self, X: pd.DataFrame, y=None):
        self.category_maps_ = {}
        for col in self.nominal_columns:
            if col in X.columns:
                values = sorted(X[col].astype(str).fillna("UNKNOWN").unique())
                self.category_maps_[col] = {value: idx for idx, value in enumerate(values)}

        self.frequency_maps_ = {}
        for col in self.frequency_columns:
            if col in X.columns:
                self.frequency_maps_[col] = (
                    X[col].astype(str).fillna("UNKNOWN").value_counts(normalize=True).to_dict()
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if "grade" in df.columns:
            df["grade"] = (
                df["grade"].astype(str).str.upper().map(self.grade_map).fillna(0).astype(int)
            )

        if "emp_length" in df.columns:
            df["emp_length"] = (
                df["emp_length"].astype(str).map(self.emp_map).fillna(0).astype(int)
            )

        for col in self.nominal_columns:
            if col in df.columns:
                mapping = getattr(self, "category_maps_", {}).get(col, {})
                df[col] = df[col].astype(str).fillna("UNKNOWN").map(mapping).fillna(-1).astype(int)

        for col in self.frequency_columns:
            if col in df.columns:
                mapping = getattr(self, "frequency_maps_", {}).get(col, {})
                df[f"{col}_freq"] = df[col].astype(str).fillna("UNKNOWN").map(mapping).fillna(0.0)
                df = df.drop(columns=[col])

        return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fit and apply the project categorical encoder."""
    return CreditCategoricalEncoder().fit_transform(df)


def split_data(df: pd.DataFrame, target: str = "loan_status",
               test_size: float = 0.2, val_size: float = 0.1,
               random_state: int = 42):
    """Split into train / validation / test sets."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (1 - test_size),
        random_state=random_state, stratify=y_train
    )

    logger.info(f"Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test
