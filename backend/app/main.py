"""
AI Insight Lab -- FastAPI backend.

What?  Serves the dashboard (a single static HTML file) and a small
       JSON API that turns an uploaded CSV into dataset statistics,
       charts, and trained ML models.
Why?   One process, one port, no build step -- `uvicorn app.main:app`
       is the whole deployment.
How?   Dataset + trained model live in memory (module-level `STATE`).
       This is a single-user local tool, not a multi-tenant service --
       see README for what you'd change to make it one.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analysis, ml, plots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-insight-lab")

MAX_UPLOAD_MB = 25
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_data"
FRONTEND_DIR = BASE_DIR.parent.parent / "frontend"

class ApiModel(BaseModel):
    model_config = {"protected_namespaces": ()}


app = FastAPI(title="AI Insight Lab", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class State:
    """Everything the API needs to remember between requests."""

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.dataset_name: Optional[str] = None
        self.trained: Optional[ml.TrainedModel] = None


STATE = State()

SAMPLE_DATASETS = {
    "house_prices": "House Price Prediction",
    "customer_churn": "Customer Churn",
    "student_performance": "Student Performance",
    "employee_salary": "Employee Salary",
    "iris_species": "Iris Species Classification",
}


# ---------------------------------------------------------------- helpers
def require_dataset() -> pd.DataFrame:
    if STATE.df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded yet. Upload a CSV or pick a sample first.")
    return STATE.df


def require_model() -> ml.TrainedModel:
    if STATE.trained is None:
        raise HTTPException(status_code=400, detail="No model has been trained yet.")
    return STATE.trained


# ------------------------------------------------------------------ data
@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds the {MAX_UPLOAD_MB}MB limit.")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as-is
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded dataset is empty.")
    if df.columns.duplicated().any():
        raise HTTPException(status_code=400, detail="The dataset has duplicate column names.")

    df.columns = [str(c).strip() for c in df.columns]
    STATE.df = df
    STATE.dataset_name = file.filename
    STATE.trained = None
    logger.info("Loaded uploaded dataset %s: %s rows x %s cols", file.filename, *df.shape)
    return {"dataset_name": file.filename, "overview": analysis.dataset_overview(df)}


@app.get("/api/samples")
def list_samples():
    return [{"key": k, "label": v} for k, v in SAMPLE_DATASETS.items()]


@app.post("/api/samples/{key}")
def load_sample(key: str):
    if key not in SAMPLE_DATASETS:
        raise HTTPException(status_code=404, detail="Unknown sample dataset.")
    path = SAMPLE_DIR / f"{key}.csv"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Sample file missing on disk.")
    df = pd.read_csv(path)
    STATE.df = df
    STATE.dataset_name = SAMPLE_DATASETS[key]
    STATE.trained = None
    return {"dataset_name": STATE.dataset_name, "overview": analysis.dataset_overview(df)}


@app.get("/api/summary")
def summary():
    df = require_dataset()
    return {
        "dataset_name": STATE.dataset_name,
        "overview": analysis.dataset_overview(df),
        "columns": analysis.column_profile(df),
    }


@app.get("/api/eda")
def eda():
    df = require_dataset()
    return {
        "numerical_summary": analysis.numerical_summary(df),
        "categorical_summary": analysis.categorical_summary(df),
        "missing_values": analysis.missing_value_report(df),
        "outliers": analysis.outlier_report(df),
        "top_correlations": plots.top_correlated_pairs(df),
    }


@app.get("/api/visualizations")
def visualizations():
    df = require_dataset()
    charts = {
        "histograms": plots.histogram_grid(df),
        "boxplots": plots.boxplot_grid(df),
        "correlation_heatmap": plots.correlation_heatmap(df),
        "categorical_bars": plots.categorical_bar_charts(df),
    }
    return {k: v for k, v in charts.items() if v is not None} | {
        "unavailable": [k for k, v in charts.items() if v is None]
    }


class CleanRequest(BaseModel):
    strategy: dict[str, str]


@app.post("/api/clean")
def clean(req: CleanRequest):
    df = require_dataset()
    cleaned = analysis.clean_dataset(df, req.strategy)
    STATE.df = cleaned
    STATE.trained = None
    return {"overview": analysis.dataset_overview(cleaned)}


# -------------------------------------------------------------------- ml
class TrainRequest(ApiModel):
    target: str
    features: list[str]
    problem_type: Optional[str] = None
    model_name: str
    test_size: float = 0.2
    cv_folds: Optional[int] = None


@app.get("/api/ml/options")
def ml_options():
    df = require_dataset()
    numerical, categorical = analysis.split_columns(df)
    return {
        "columns": list(df.columns),
        "numerical": numerical,
        "categorical": categorical,
        "classification_models": list(ml.CLASSIFICATION_MODELS.keys()),
        "regression_models": list(ml.REGRESSION_MODELS.keys()),
    }


@app.get("/api/ml/suggest-problem-type")
def suggest_problem_type(target: str):
    df = require_dataset()
    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Unknown target column.")
    return {"problem_type": ml.infer_problem_type(df, target)}


@app.post("/api/ml/train")
def train(req: TrainRequest):
    df = require_dataset()
    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found in dataset.")
    if req.target in req.features:
        raise HTTPException(status_code=400, detail="Target column cannot also be a feature.")
    if not req.features:
        raise HTTPException(status_code=400, detail="Select at least one feature column.")

    problem_type = req.problem_type or ml.infer_problem_type(df, req.target)
    try:
        trained = ml.train_model(
            df=df,
            target=req.target,
            features=req.features,
            problem_type=problem_type,
            model_name=req.model_name,
            test_size=req.test_size,
            cv_folds=req.cv_folds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    STATE.trained = trained
    return {
        "problem_type": trained.problem_type,
        "model_name": trained.model_name,
        "target": trained.target,
        "features": trained.feature_names,
        "metrics": trained.metrics,
        "charts": trained.charts,
        "feature_importance": trained.feature_importance,
        "class_labels": trained.class_labels,
    }


@app.get("/api/ml/model-info")
def model_info():
    trained = require_model()
    return {
        "problem_type": trained.problem_type,
        "model_name": trained.model_name,
        "target": trained.target,
        "features": trained.feature_names,
        "numerical_features": trained.numerical_features,
        "categorical_features": trained.categorical_features,
        "class_labels": trained.class_labels,
    }


class PredictRequest(BaseModel):
    values: dict[str, Any]


@app.post("/api/ml/predict")
def predict(req: PredictRequest):
    trained = require_model()
    missing = [f for f in trained.feature_names if f not in req.values]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing values for: {', '.join(missing)}")
    try:
        return ml.predict(trained, req.values)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.get("/api/ml/download-model")
def download_model():
    trained = require_model()
    out_path = Path("/tmp/ai_insight_lab_model.joblib")
    joblib.dump(
        {
            "pipeline": trained.pipeline,
            "problem_type": trained.problem_type,
            "model_name": trained.model_name,
            "target": trained.target,
            "feature_names": trained.feature_names,
            "class_labels": trained.class_labels,
        },
        out_path,
    )
    return FileResponse(out_path, filename="model.joblib", media_type="application/octet-stream")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------------------ frontend
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
