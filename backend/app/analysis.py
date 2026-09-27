"""
Dataset analysis engine.

What?  Turns a raw pandas DataFrame into the structured summaries the
       dashboard displays (overview, per-column stats, missing values,
       outliers).
Why?   Keeps main.py thin -- routes just call these functions and
       return the result as JSON.
How?   Pure pandas / numpy. No side effects on the DataFrame unless the
       function name says so (e.g. clean_dataset).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numerical_columns, categorical_columns)."""
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numerical]
    return numerical, categorical


def dataset_overview(df: pd.DataFrame) -> dict:
    numerical, categorical = split_columns(df)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numerical_features": len(numerical),
        "categorical_features": len(categorical),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_kb": round(float(df.memory_usage(deep=True).sum()) / 1024, 2),
    }


def column_profile(df: pd.DataFrame) -> list[dict]:
    """Per-column profile: dtype, missing %, unique count, a small preview."""
    numerical, _ = split_columns(df)
    profiles = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        profile = {
            "name": col,
            "dtype": str(series.dtype),
            "is_numeric": col in numerical,
            "missing_count": missing,
            "missing_pct": round(missing / len(df) * 100, 2) if len(df) else 0,
            "unique_count": int(series.nunique(dropna=True)),
        }
        if col in numerical:
            desc = series.describe()
            profile.update(
                {
                    "mean": _safe_round(desc.get("mean")),
                    "std": _safe_round(desc.get("std")),
                    "min": _safe_round(desc.get("min")),
                    "max": _safe_round(desc.get("max")),
                }
            )
        else:
            mode = series.mode(dropna=True)
            profile["top_value"] = str(mode.iloc[0]) if not mode.empty else None
        profiles.append(profile)
    return profiles


def numerical_summary(df: pd.DataFrame) -> list[dict]:
    numerical, _ = split_columns(df)
    if not numerical:
        return []
    desc = df[numerical].describe().T
    rows = []
    for col, row in desc.iterrows():
        rows.append(
            {
                "column": col,
                "count": int(row["count"]),
                "mean": _safe_round(row["mean"]),
                "std": _safe_round(row["std"]),
                "min": _safe_round(row["min"]),
                "q25": _safe_round(row["25%"]),
                "median": _safe_round(row["50%"]),
                "q75": _safe_round(row["75%"]),
                "max": _safe_round(row["max"]),
                "skewness": _safe_round(df[col].skew()),
                "kurtosis": _safe_round(df[col].kurt()),
            }
        )
    return rows


def categorical_summary(df: pd.DataFrame) -> list[dict]:
    _, categorical = split_columns(df)
    rows = []
    for col in categorical:
        counts = df[col].value_counts(dropna=True)
        top = counts.index[0] if len(counts) else None
        rows.append(
            {
                "column": col,
                "unique_count": int(df[col].nunique(dropna=True)),
                "top_value": str(top) if top is not None else None,
                "top_frequency": int(counts.iloc[0]) if len(counts) else 0,
                "top_categories": [
                    {"value": str(idx), "count": int(cnt)}
                    for idx, cnt in counts.head(5).items()
                ],
            }
        )
    return rows


def missing_value_report(df: pd.DataFrame) -> list[dict]:
    report = []
    total = len(df)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing == 0:
            continue
        pct = round(missing / total * 100, 2) if total else 0
        if df[col].dtype.kind in "biufc":
            strategy = "median" if pct < 30 else "consider dropping column"
        else:
            strategy = "mode" if pct < 30 else "consider dropping column"
        report.append(
            {
                "column": col,
                "missing_count": missing,
                "missing_pct": pct,
                "recommended_strategy": strategy,
            }
        )
    return sorted(report, key=lambda r: -r["missing_pct"])


def outlier_report(df: pd.DataFrame) -> list[dict]:
    """IQR-method outlier detection for numerical columns."""
    numerical, _ = split_columns(df)
    report = []
    for col in numerical:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) == 0:
            continue
        report.append(
            {
                "column": col,
                "lower_bound": _safe_round(lower),
                "upper_bound": _safe_round(upper),
                "outlier_count": int(len(outliers)),
                "outlier_pct": round(len(outliers) / len(series) * 100, 2),
            }
        )
    return report


def clean_dataset(df: pd.DataFrame, strategy: dict[str, str]) -> pd.DataFrame:
    """
    Apply a per-column missing-value strategy chosen by the user.
    strategy: {column_name: "mean" | "median" | "mode" | "constant:<value>" | "drop_rows" | "drop_column"}
    """
    df = df.copy()
    drop_cols = [c for c, s in strategy.items() if s == "drop_column"]
    df = df.drop(columns=drop_cols, errors="ignore")

    drop_row_cols = [c for c, s in strategy.items() if s == "drop_rows" and c in df.columns]
    if drop_row_cols:
        df = df.dropna(subset=drop_row_cols)

    for col, s in strategy.items():
        if col not in df.columns or s in ("drop_column", "drop_rows"):
            continue
        if s == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif s == "median":
            df[col] = df[col].fillna(df[col].median())
        elif s == "mode":
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
        elif s.startswith("constant:"):
            value = s.split(":", 1)[1]
            df[col] = df[col].fillna(value)
    return df


def _safe_round(value, ndigits: int = 4):
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(value), ndigits)
