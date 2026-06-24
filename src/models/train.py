"""
Model training: XGBoost, LightGBM, and a PyTorch Neural Network
with Optuna hyperparameter tuning and MLflow experiment tracking.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────
# XGBoost with Optuna tuning
# ─────────────────────────────────────────────

def _xgb_objective(trial, X_train, y_train, X_val, y_val):
    pos = max((y_train == 1).sum(), 1)
    neg = max((y_train == 0).sum(), 1)
    class_ratio = neg / pos
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", max(1.0, class_ratio * 0.5), class_ratio * 1.5),
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    from sklearn.metrics import average_precision_score
    preds = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, preds)


def train_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 30,
                  experiment_name: str = "credit-risk-xgb") -> XGBClassifier:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="xgboost_optuna"):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: _xgb_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
        )
        best_params = study.best_params
        best_params.update({"eval_metric": "aucpr", "tree_method": "hist", "n_jobs": -1, "random_state": 42})
        logger.info(f"XGBoost best PR-AUC: {study.best_value:.4f}")

        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        mlflow.log_params(best_params)
        mlflow.log_metric("val_pr_auc", study.best_value)
        mlflow.xgboost.log_model(model, "xgboost_model")

    return model


# ─────────────────────────────────────────────
# Logistic Regression baseline
# ─────────────────────────────────────────────

def train_logistic_regression(X_train, y_train,
                               experiment_name: str = "credit-risk-lr") -> Pipeline:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       solver="lbfgs", random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        mlflow.sklearn.log_model(pipeline, "lr_model")
    return pipeline


# ─────────────────────────────────────────────
# PyTorch Neural Network
# ─────────────────────────────────────────────

class CreditRiskNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_neural_net(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     epochs: int = 30, batch_size: int = 512,
                     lr: float = 1e-3,
                     experiment_name: str = "credit-risk-nn") -> CreditRiskNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Neural Net on {device}")

    # Scale
    mean, std = X_train.mean(0), X_train.std(0) + 1e-8
    X_tr = torch.tensor((X_train - mean) / std, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    X_v = torch.tensor((X_val - mean) / std, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val.values, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    model = CreditRiskNet(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="pytorch_nn"):
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_v), y_v).item()
            scheduler.step()

            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}/{epochs} | train_loss={total_loss/len(loader):.4f} | val_loss={val_loss:.4f}")
                mlflow.log_metrics({"train_loss": total_loss / len(loader), "val_loss": val_loss}, step=epoch)

    return model


# ─────────────────────────────────────────────
# Persist models
# ─────────────────────────────────────────────

def save_model(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), path)
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    logger.info(f"Model saved → {path}")


def load_model(path: str, model_class=None, **kwargs):
    if model_class is not None:
        m = model_class(**kwargs)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        return m
    with open(path, "rb") as f:
        return pickle.load(f)
