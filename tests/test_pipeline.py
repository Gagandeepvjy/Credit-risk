"""
Unit & integration tests for the credit risk platform.
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd

from src.data.ingestion import (
    CreditCategoricalEncoder,
    prepare_lending_club_dataframe,
    clean_data,
    encode_categoricals,
    split_data,
)
from src.features.engineering import CreditFeatureEngineer


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    records = []
    statuses = ["Fully Paid"] * 40 + ["Charged Off"] * 20 + ["Current"] * 5
    for i, status in enumerate(statuses):
        records.append({
            "loan_amnt": 5000 + (i % 6) * 5000,
            "term": " 36 months" if i % 2 == 0 else " 60 months",
            "int_rate": f"{8 + (i % 15) * 0.8:.1f}%",
            "installment": 180 + (i % 10) * 45,
            "grade": list("ABCDEFG")[i % 7],
            "emp_length": ["< 1 year", "2 years", "5 years", "10+ years"][i % 4],
            "home_ownership": ["RENT", "OWN", "MORTGAGE"][i % 3],
            "annual_inc": 45000 + (i % 12) * 5000,
            "verification_status": ["Verified", "Source Verified", "Not Verified"][i % 3],
            "purpose": ["debt_consolidation", "credit_card", "home_improvement", "other"][i % 4],
            "dti": 8 + (i % 20) * 1.2,
            "delinq_2yrs": i % 3,
            "fico_range_low": 620 + (i % 12) * 15,
            "open_acc": 4 + i % 15,
            "pub_rec": i % 2,
            "revol_bal": 2000 + (i % 10) * 1500,
            "revol_util": f"{15 + (i % 9) * 8:.1f}%",
            "total_acc": 8 + i % 30,
            "mort_acc": i % 4,
            "pub_rec_bankruptcies": 1 if i % 17 == 0 else 0,
            "loan_status": status,
        })
    return prepare_lending_club_dataframe(pd.DataFrame(records))


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture(scope="module")
def encoded_df(clean_df):
    return encode_categoricals(clean_df)


@pytest.fixture(scope="module")
def engineered_df(encoded_df):
    fe = CreditFeatureEngineer()
    return fe.fit_transform(encoded_df)


# ─────────────────────────────────────────────
# Data ingestion tests
# ─────────────────────────────────────────────

class TestDataIngestion:
    def test_load_kaggle_shape(self, raw_df):
        assert len(raw_df) == 60
        assert "loan_status" in raw_df.columns

    def test_default_rate_reasonable(self, raw_df):
        rate = raw_df["loan_status"].mean()
        assert 0.20 < rate < 0.50, f"Default rate {rate:.2%} seems unrealistic"

    def test_clean_no_nulls(self, clean_df):
        assert clean_df.isnull().sum().sum() == 0

    def test_clean_revol_util_capped(self, clean_df):
        assert clean_df["revol_util"].max() <= 100

    def test_encode_grade_numeric(self, encoded_df):
        assert encoded_df["grade"].dtype in [np.int64, np.float64]

    def test_encoder_reuses_training_mapping(self, clean_df):
        encoder = CreditCategoricalEncoder().fit(clean_df)
        sample = clean_df.head(5)
        encoded = encoder.transform(sample)
        assert encoded["home_ownership"].dtype in [np.int64, np.int32]
        assert encoded["verification_status"].dtype in [np.int64, np.int32]
        assert encoded["purpose"].dtype in [np.int64, np.int32]

    def test_encoder_handles_unseen_categories(self, clean_df):
        encoder = CreditCategoricalEncoder().fit(clean_df)
        sample = clean_df.head(1).copy()
        sample["purpose"] = "new_unseen_purpose"
        encoded = encoder.transform(sample)
        assert encoded["purpose"].iloc[0] == -1

    def test_split_sizes(self, engineered_df):
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(engineered_df)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(engineered_df)
        assert len(X_test) > 0

    def test_prepare_lending_club_status_mapping(self):
        raw = pd.DataFrame({
            "loan_amnt": [10000, 12000, 9000],
            "term": [" 36 months", " 60 months", " 36 months"],
            "int_rate": ["12.5%", "18.2%", "8.4%"],
            "installment": [350, 420, 280],
            "grade": ["B", "D", "A"],
            "emp_length": ["5 years", "2 years", "10+ years"],
            "home_ownership": ["RENT", "OWN", "MORTGAGE"],
            "annual_inc": [60000, 52000, 90000],
            "verification_status": ["Verified", "Source Verified", "Not Verified"],
            "purpose": ["debt_consolidation", "credit_card", "home_improvement"],
            "dti": [15.0, 22.0, 9.0],
            "delinq_2yrs": [0, 1, 0],
            "fico_range_low": [700, 650, 760],
            "open_acc": [8, 10, 7],
            "pub_rec": [0, 1, 0],
            "revol_bal": [5000, 12000, 3000],
            "revol_util": ["30.0%", "88.0%", "20.0%"],
            "total_acc": [20, 24, 15],
            "mort_acc": [1, 0, 2],
            "pub_rec_bankruptcies": [0, 1, 0],
            "loan_status": ["Fully Paid", "Charged Off", "Current"],
        })

        prepared = prepare_lending_club_dataframe(raw)

        assert len(prepared) == 2
        assert prepared["loan_status"].tolist() == [0, 1]
        assert prepared["term"].tolist() == [36, 60]
        assert prepared["int_rate"].tolist() == [12.5, 18.2]
        assert prepared["revol_util"].tolist() == [30.0, 88.0]


# ─────────────────────────────────────────────
# Feature engineering tests
# ─────────────────────────────────────────────

class TestFeatureEngineering:
    def test_new_features_created(self, engineered_df):
        expected = ["loan_to_income", "installment_to_income", "high_util_flag",
                    "fico_risk_tier", "has_delinquency"]
        for feat in expected:
            assert feat in engineered_df.columns, f"Missing feature: {feat}"

    def test_no_nulls_after_engineering(self, engineered_df):
        assert engineered_df.isnull().sum().sum() == 0

    def test_feature_count_increased(self, encoded_df, engineered_df):
        assert engineered_df.shape[1] > encoded_df.shape[1]

    def test_high_util_flag_binary(self, engineered_df):
        assert set(engineered_df["high_util_flag"].unique()).issubset({0, 1})

    def test_loan_to_income_positive(self, engineered_df):
        assert (engineered_df["loan_to_income"] >= 0).all()


# ─────────────────────────────────────────────
# API schema tests
# ─────────────────────────────────────────────

class TestAPISchema:
    def test_valid_grade(self):
        from src.api.serve import LoanApplication
        with pytest.raises(Exception):
            LoanApplication(
                loan_amnt=10000, term=36, int_rate=12.5, installment=350,
                grade="Z",  # invalid
                emp_length="5 years", home_ownership="RENT", annual_inc=60000,
                verification_status="Verified", purpose="debt_consolidation",
                dti=15.0, fico_range_low=700, open_acc=8, revol_bal=5000,
                revol_util=30.0, total_acc=20,
            )

    def test_negative_loan_rejected(self):
        from src.api.serve import LoanApplication
        with pytest.raises(Exception):
            LoanApplication(
                loan_amnt=-5000, term=36, int_rate=12.5, installment=350,
                grade="B", emp_length="5 years", home_ownership="RENT",
                annual_inc=60000, verification_status="Verified",
                purpose="debt_consolidation", dti=15.0, fico_range_low=700,
                open_acc=8, revol_bal=5000, revol_util=30.0, total_acc=20,
            )
