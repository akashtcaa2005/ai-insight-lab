# AI Insight Lab

Upload a CSV, get an end-to-end data analysis: dataset overview, data-quality
checks, statistics, visualizations, and trained ML models with predictions —
all from a single dashboard, no notebook required.

## What it does

1. **Upload** a CSV (or pick one of five bundled sample datasets)
2. **Overview** — rows, columns, dtypes, missing values, duplicates, per-column profile
3. **Data Quality** — missing-value report with a per-column cleaning strategy you choose and apply; IQR-based outlier detection
4. **Statistics** — descriptive stats, skewness/kurtosis, categorical breakdowns, top correlated feature pairs
5. **Visualizations** — histograms, boxplots, a correlation heatmap, and categorical bar charts, generated automatically from the dataset's column types
6. **Train Model** — pick a target column, features, and a model (Logistic Regression / Decision Tree / Random Forest / KNN / SVM for classification; Linear Regression / Decision Tree / Random Forest for regression). Preprocessing (imputation, scaling, one-hot encoding) is built with a scikit-learn `Pipeline` + `ColumnTransformer`, fit only on the training split. Shows accuracy/precision/recall/F1 or MAE/RMSE/R², a confusion matrix or actual-vs-predicted plot, feature importance, and optional cross-validation.
7. **Predict** — enter feature values and get a live prediction (with class probabilities for classification), and download the trained model as a `.joblib` file.

## Tech stack

- **Backend**: FastAPI, pandas, NumPy, scikit-learn, Matplotlib/Seaborn (charts rendered server-side as PNGs), Joblib
- **Frontend**: a single dependency-free `index.html` (vanilla JS, no build step) served by FastAPI itself

One process, one port — no Node, no separate frontend server.

## Project structure

```
ai-insight-lab/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI routes
│   │   ├── analysis.py      # pandas/numpy dataset analysis
│   │   ├── ml.py            # preprocessing, training, prediction
│   │   ├── plots.py         # matplotlib/seaborn chart generation
│   │   └── sample_data/     # bundled sample CSVs
│   ├── generate_samples.py  # regenerates the sample CSVs
│   └── requirements.txt
├── frontend/
│   └── index.html           # the entire UI
└── README.md
```

## Setup

Requires Python 3.10+.

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Open **http://127.0.0.1:8000** — the dashboard and the API are served from the
same port.

## API

All endpoints are under `/api`:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/upload` | Upload a CSV |
| GET  | `/samples` | List bundled sample datasets |
| POST | `/samples/{key}` | Load a sample dataset |
| GET  | `/summary` | Dataset overview + per-column profile |
| GET  | `/eda` | Numerical/categorical summaries, missing values, outliers, correlations |
| GET  | `/visualizations` | Chart set as base64 PNGs |
| POST | `/clean` | Apply a per-column missing-value strategy |
| GET  | `/ml/options` | Columns and available models for the current dataset |
| GET  | `/ml/suggest-problem-type?target=` | Heuristic classification/regression suggestion |
| POST | `/ml/train` | Train a model |
| GET  | `/ml/model-info` | Info about the currently trained model |
| POST | `/ml/predict` | Predict from feature values |
| GET  | `/ml/download-model` | Download the trained model as `.joblib` |

Interactive API docs: **http://127.0.0.1:8000/docs**

## Notes on scope

This is intentionally a **single-user, in-memory** tool: the current dataset
and trained model live in a module-level `State` object in `main.py`, not a
database. That keeps the project simple to run and read end-to-end. To make
it multi-user you'd move `State` into a session (e.g. keyed by a cookie or
API key) and persist datasets/models to disk or a database instead of memory.

## Regenerating sample data

```bash
cd backend
python generate_samples.py
```

This creates `house_prices.csv`, `student_performance.csv`,
`customer_churn.csv`, and `employee_salary.csv` as synthetic-but-realistic
datasets, and `iris_species.csv` from scikit-learn's bundled Iris dataset.
