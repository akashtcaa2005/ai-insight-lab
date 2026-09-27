"""
Visualization engine.

What?  Generates the chart set for a dataset (histograms, boxplots,
       correlation heatmap, categorical bar charts, scatter/pairwise
       relationships, confusion matrix, actual-vs-predicted).
Why?   The frontend has no charting library -- every chart is rendered
       server-side with Matplotlib/Seaborn and shipped as a base64 PNG,
       which keeps the frontend a single dependency-free HTML file.
How?   Each function builds a Matplotlib figure, saves it to an
       in-memory buffer, and returns a "data:image/png;base64,..." URL.
"""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .analysis import split_columns

sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#12171C00"})
PALETTE = ["#4FD1C5", "#F2C94C", "#F87171", "#818CF8", "#34D399", "#F472B6"]
plt.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "text.color": "#C9D1D9",
        "axes.labelcolor": "#C9D1D9",
        "axes.edgecolor": "#3A4552",
        "xtick.color": "#8B98A5",
        "ytick.color": "#8B98A5",
        "font.family": "monospace",
        "font.size": 10,
    }
)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def histogram_grid(df: pd.DataFrame, max_cols: int = 6) -> str | None:
    numerical, _ = split_columns(df)
    numerical = numerical[:max_cols]
    if not numerical:
        return None
    n = len(numerical)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numerical):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color=PALETTE[i % len(PALETTE)])
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel("Count")
    for j in range(len(numerical), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return _fig_to_base64(fig)


def boxplot_grid(df: pd.DataFrame, max_cols: int = 6) -> str | None:
    numerical, _ = split_columns(df)
    numerical = numerical[:max_cols]
    if not numerical:
        return None
    n = len(numerical)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numerical):
        sns.boxplot(y=df[col].dropna(), ax=axes[i], color=PALETTE[i % len(PALETTE)])
        axes[i].set_title(col, fontsize=10)
    for j in range(len(numerical), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return _fig_to_base64(fig)


def correlation_heatmap(df: pd.DataFrame) -> str | None:
    numerical, _ = split_columns(df)
    if len(numerical) < 2:
        return None
    corr = df[numerical].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(1.1 * len(numerical) + 2, 1.0 * len(numerical) + 2))
    cmap = sns.diverging_palette(200, 20, s=80, l=45, as_cmap=True)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax,
        linewidths=0.5, linecolor="#232B32", cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    return _fig_to_base64(fig)


def top_correlated_pairs(df: pd.DataFrame, limit: int = 5) -> list[dict]:
    numerical, _ = split_columns(df)
    if len(numerical) < 2:
        return []
    corr = df[numerical].corr(numeric_only=True).abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    pairs.sort(key=lambda p: -p[2] if p[2] == p[2] else 0)  # NaN-safe sort
    out = []
    for a, b, v in pairs[:limit]:
        if v != v:
            continue
        strength = (
            "strong" if v >= 0.7 else "moderate" if v >= 0.4 else "weak"
        )
        direction = "positive" if df[[a, b]].corr().iloc[0, 1] >= 0 else "negative"
        out.append(
            {
                "feature_a": a,
                "feature_b": b,
                "correlation": round(float(v), 3),
                "description": f"{a} and {b} show a {strength} {direction} linear relationship.",
            }
        )
    return out


def categorical_bar_charts(df: pd.DataFrame, max_cols: int = 4) -> str | None:
    _, categorical = split_columns(df)
    categorical = [c for c in categorical if df[c].nunique(dropna=True) <= 20][:max_cols]
    if not categorical:
        return None
    ncols = min(2, len(categorical))
    nrows = int(np.ceil(len(categorical) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(categorical):
        counts = df[col].value_counts(dropna=True).head(10)
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=axes[i], color=PALETTE[i % len(PALETTE)])
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Count")
    for j in range(len(categorical), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return _fig_to_base64(fig)


def confusion_matrix_plot(cm: np.ndarray, labels: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cmap = sns.light_palette("#4FD1C5", as_cmap=True)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, linecolor="#232B32", cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return _fig_to_base64(fig)


def actual_vs_predicted_plot(y_true, y_pred) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.scatter(y_true, y_pred, alpha=0.6, color="#4FD1C5", edgecolor="none", s=28)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "--", color="#F87171", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _fig_to_base64(fig)


def residual_plot(y_true, y_pred) -> str:
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.scatter(y_pred, residuals, alpha=0.6, color="#818CF8", edgecolor="none", s=28)
    ax.axhline(0, linestyle="--", color="#F87171", linewidth=1.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot")
    fig.tight_layout()
    return _fig_to_base64(fig)


def feature_importance_plot(names: list[str], importances: list[float]) -> str:
    order = np.argsort(importances)[::-1][:15]
    names = [names[i] for i in order]
    importances = [importances[i] for i in order]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(names) + 1.5))
    sns.barplot(x=importances, y=names, ax=ax, color="#4FD1C5")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return _fig_to_base64(fig)
