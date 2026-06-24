"""
FastAPI serving layer for the Credit Risk model.
Endpoints: /predict, /predict/batch, /health, /model/info
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

from src.data.ingestion import clean_data

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Enterprise-grade credit default prediction service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ──────────────────────────────────────────────
_model = None
_categorical_encoder = None
_feature_engineer = None
MODEL_PATH = Path("models/xgboost_model.pkl")
ENCODER_PATH = Path("models/categorical_encoder.pkl")
FE_PATH = Path("models/feature_engineer.pkl")


def load_artifacts():
    global _model, _categorical_encoder, _feature_engineer
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        logger.info("Model loaded from disk")
    if ENCODER_PATH.exists():
        with open(ENCODER_PATH, "rb") as f:
            _categorical_encoder = pickle.load(f)
        logger.info("Categorical encoder loaded from disk")
    if FE_PATH.exists():
        with open(FE_PATH, "rb") as f:
            _feature_engineer = pickle.load(f)
        logger.info("Feature engineer loaded from disk")


@app.on_event("startup")
async def startup_event():
    load_artifacts()


# ── Schemas ─────────────────────────────────────────────────────────

class LoanApplication(BaseModel):
    loan_amnt:            float = Field(..., gt=0, description="Requested loan amount in USD")
    term:                 int   = Field(..., description="Loan term in months (36 or 60)")
    int_rate:             float = Field(..., gt=0, lt=40)
    installment:          float = Field(..., gt=0)
    grade:                str   = Field(..., description="LC grade: A-G")
    emp_length:           str   = Field(..., description="Employment length string")
    home_ownership:       str
    annual_inc:           float = Field(..., gt=0)
    verification_status:  str
    purpose:              str
    dti:                  float = Field(..., ge=0)
    delinq_2yrs:          int   = Field(0, ge=0)
    fico_range_low:       int   = Field(..., ge=300, le=850)
    open_acc:             int   = Field(..., ge=0)
    pub_rec:              int   = Field(0, ge=0)
    revol_bal:            float = Field(..., ge=0)
    revol_util:           float = Field(..., ge=0, le=100)
    total_acc:            int   = Field(..., ge=0)
    mort_acc:             int   = Field(0, ge=0)
    pub_rec_bankruptcies: int   = Field(0, ge=0)

    @field_validator("grade")
    @classmethod
    def grade_must_be_valid(cls, v):
        if v.upper() not in list("ABCDEFG"):
            raise ValueError("grade must be A–G")
        return v.upper()


class PredictionResponse(BaseModel):
    default_probability: float
    risk_tier:           str
    decision:            str
    confidence:          str
    expected_loss_usd:   Optional[float]


class BatchRequest(BaseModel):
    applications: List[LoanApplication]


# ── Helpers ─────────────────────────────────────────────────────────

def _risk_tier(prob: float) -> str:
    if prob < 0.10:  return "LOW"
    if prob < 0.25:  return "MEDIUM"
    if prob < 0.50:  return "HIGH"
    return "VERY HIGH"


def _decision(prob: float, threshold: float = 0.35) -> str:
    return "APPROVE" if prob < threshold else "DECLINE"


def _to_dataframe(app: LoanApplication) -> pd.DataFrame:
    return pd.DataFrame([app.model_dump()])


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [
        name for name, artifact in {
            "categorical encoder": _categorical_encoder,
            "feature engineer": _feature_engineer,
        }.items()
        if artifact is None
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Model preprocessing artifact(s) missing: {', '.join(missing)}",
        )

    df = clean_data(df)
    df = _categorical_encoder.transform(df)
    df = _feature_engineer.transform(df)
    return df


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "categorical_encoder_loaded": _categorical_encoder is not None,
        "feature_engineer_loaded": _feature_engineer is not None,
    }


@app.get("/model/info")
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type":    type(_model).__name__,
        "n_features":    getattr(_model, "n_features_in_", "unknown"),
        "threshold":     0.35,
        "lgd_assumption": 0.45,
        "preprocessing": ["clean_data", "CreditCategoricalEncoder", "CreditFeatureEngineer"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = _to_dataframe(application)
    df = _prepare_features(df)

    prob = float(_model.predict_proba(df)[:, 1][0])
    el   = round(prob * 0.45 * application.loan_amnt, 2)

    return PredictionResponse(
        default_probability=round(prob, 4),
        risk_tier=_risk_tier(prob),
        decision=_decision(prob),
        confidence="HIGH" if abs(prob - 0.35) > 0.15 else "MEDIUM",
        expected_loss_usd=el,
    )


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    records = [a.model_dump() for a in request.applications]
    df = pd.DataFrame(records)
    df = _prepare_features(df)

    probas = _model.predict_proba(df)[:, 1]
    results = []
    for i, (app, prob) in enumerate(zip(request.applications, probas)):
        results.append({
            "index": i,
            "default_probability": round(float(prob), 4),
            "risk_tier": _risk_tier(prob),
            "decision": _decision(prob),
            "expected_loss_usd": round(prob * 0.45 * app.loan_amnt, 2),
        })
    return {"results": results, "count": len(results)}


if __name__ == "__main__":
    uvicorn.run("src.api.serve:app", host="0.0.0.0", port=8000, reload=True)
