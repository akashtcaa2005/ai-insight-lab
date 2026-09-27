"""
Machine learning engine.

What?  Builds a preprocessing pipeline, trains a chosen model, evaluates
       it, and stores it in memory for later predictions.
Why?   Isolates every scikit-learn detail from the API layer so main.py
       stays a thin set of routes.
How?   ColumnTransformer (impute+scale numeric, impute+one-hot
       categorical) inside a Pipeline, fit only on the training split
       to avoid leakage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import plots
from .analysis import split_columns

CLASSIFICATION_MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=42),
    "random_forest": lambda: RandomForestClassifier(random_state=42, n_estimators=200),
    "knn": lambda: KNeighborsClassifier(),
    "svm": lambda: SVC(probability=True, random_state=42),
}

REGRESSION_MODELS = {
    "linear_regression": lambda: LinearRegression(),
    "decision_tree": lambda: DecisionTreeRegressor(random_state=42),
    "random_forest": lambda: RandomForestRegressor(random_state=42, n_estimators=200),
}


@dataclass
class TrainedModel:
    pipeline: Pipeline
    problem_type: str
    model_name: str
    target: str
    feature_names: list[str]
    numerical_features: list[str]
    categorical_features: list[str]
    class_labels: list[str] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    charts: dict[str, str] = field(default_factory=dict)
    feature_importance: list[dict] | None = None


def infer_problem_type(df: pd.DataFrame, target: str) -> str:
    series = df[target].dropna()
    if series.dtype.kind in "OUS" or series.nunique() <= 15:
        return "classification"
    return "regression"


def build_preprocessor(numerical: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numerical),
            ("cat", categorical_pipe, categorical),
        ]
    )


def train_model(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    problem_type: str,
    model_name: str,
    test_size: float = 0.2,
    cv_folds: int | None = None,
) -> TrainedModel:
    data = df[features + [target]].dropna(subset=[target]).copy()
    if data.empty:
        raise ValueError("No usable rows remain after dropping missing target values.")

    X = data[features]
    y = data[target]

    if y.nunique() <= 1:
        raise ValueError("The target column must contain at least two distinct values to train a model.")

    numerical, categorical = split_columns(X)
    preprocessor = build_preprocessor(numerical, categorical)

    registry = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}' for problem type '{problem_type}'")

    class_labels = None
    if problem_type == "classification":
        y = y.astype(str)
        class_labels = sorted(y.unique().tolist())
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            raise ValueError("The target column must contain at least two classes for classification.")
        if class_counts.min() < 2:
            raise ValueError(
                "Each class must have at least 2 rows to train and validate a model. "
                "Choose a different target or merge rare categories."
            )
        if len(y) < 4:
            raise ValueError("Classification training requires at least 4 non-missing rows.")

    if problem_type == "regression" and y.nunique() <= 1:
        raise ValueError("Regression target is constant; choose a different target column.")

    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", registry[model_name]())])

    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = round(time.time() - start, 4)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    trained = TrainedModel(
        pipeline=pipeline,
        problem_type=problem_type,
        model_name=model_name,
        target=target,
        feature_names=features,
        numerical_features=numerical,
        categorical_features=categorical,
        class_labels=class_labels,
    )

    if problem_type == "classification":
        trained.metrics = {
            "train_accuracy": round(accuracy_score(y_train, y_pred_train), 4),
            "test_accuracy": round(accuracy_score(y_test, y_pred_test), 4),
            "precision": round(precision_score(y_test, y_pred_test, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_test, average="weighted", zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred_test, average="weighted", zero_division=0), 4),
            "training_time_sec": train_time,
        }
        cm = confusion_matrix(y_test, y_pred_test, labels=class_labels)
        trained.charts["confusion_matrix"] = plots.confusion_matrix_plot(cm, class_labels)
    else:
        trained.metrics = {
            "train_r2": round(r2_score(y_train, y_pred_train), 4),
            "test_r2": round(r2_score(y_test, y_pred_test), 4),
            "mae": round(mean_absolute_error(y_test, y_pred_test), 4),
            "mse": round(mean_squared_error(y_test, y_pred_test), 4),
            "rmse": round(mean_squared_error(y_test, y_pred_test) ** 0.5, 4),
            "training_time_sec": train_time,
        }
        trained.charts["actual_vs_predicted"] = plots.actual_vs_predicted_plot(
            y_test.tolist(), y_pred_test.tolist()
        )
        trained.charts["residuals"] = plots.residual_plot(y_test.tolist(), y_pred_test.tolist())

    if cv_folds:
        scores = cross_val_score(pipeline, X, y, cv=cv_folds)
        trained.metrics["cv_mean"] = round(float(scores.mean()), 4)
        trained.metrics["cv_std"] = round(float(scores.std()), 4)
        trained.metrics["cv_folds"] = cv_folds

    model_step = pipeline.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        feature_names_out = pipeline.named_steps["preprocess"].get_feature_names_out()
        clean_names = [
            n.replace("num__", "").replace("cat__", "") for n in feature_names_out
        ]
        importances = model_step.feature_importances_
        trained.feature_importance = sorted(
            [
                {"feature": name, "importance": round(float(imp), 4)}
                for name, imp in zip(clean_names, importances)
            ],
            key=lambda r: -r["importance"],
        )[:15]
        trained.charts["feature_importance"] = plots.feature_importance_plot(
            [f["feature"] for f in trained.feature_importance],
            [f["importance"] for f in trained.feature_importance],
        )
    else:
        trained.feature_importance = None

    return trained


def predict(trained: TrainedModel, values: dict[str, Any]) -> dict:
    row = {f: values.get(f) for f in trained.feature_names}
    X = pd.DataFrame([row])
    prediction = trained.pipeline.predict(X)[0]
    result: dict[str, Any] = {"prediction": _to_native(prediction)}

    if trained.problem_type == "classification" and hasattr(trained.pipeline, "predict_proba"):
        proba = trained.pipeline.predict_proba(X)[0]
        classes = trained.pipeline.named_steps["model"].classes_
        result["probabilities"] = {
            str(c): round(float(p), 4) for c, p in zip(classes, proba)
        }
    return result


def _to_native(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
