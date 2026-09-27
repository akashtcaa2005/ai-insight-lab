from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import analysis, ml, plots


SAMPLE_DIR = ROOT / "backend" / "app" / "sample_data"
SAMPLE_FILES = {
    "House Price Prediction": "house_prices.csv",
    "Customer Churn": "customer_churn.csv",
    "Student Performance": "student_performance.csv",
    "Employee Salary": "employee_salary.csv",
    "Iris Species Classification": "iris_species.csv",
}

st.set_page_config(page_title="AI Insight Lab", page_icon="📊", layout="wide")


@st.cache_data
def load_sample_dataframe(sample_name: str) -> pd.DataFrame:
    sample_file = SAMPLE_DIR / SAMPLE_FILES[sample_name]
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample dataset missing: {sample_file}")
    return pd.read_csv(sample_file)


def validate_dataframe(df: pd.DataFrame) -> None:
    if df is None:
        raise ValueError("No dataset loaded.")
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    if df.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names. Please fix them before continuing.")
    if len(df.columns) == 0:
        raise ValueError("The dataset does not contain any columns.")


def load_dataset_from_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("Please upload a CSV file.")
    if not uploaded_file.name.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # pragma: no cover - surfaced to the user
        raise ValueError(f"Could not read CSV: {exc}") from exc
    validate_dataframe(df)
    return df


def safe_display_image(chart_data: str | None, caption: str) -> None:
    if chart_data:
        st.image(chart_data, caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} is not available for the current dataset.")


def sidebar_dataset_controls() -> tuple[pd.DataFrame | None, str | None]:
    with st.sidebar:
        st.title("AI Insight Lab")
        st.caption("Upload a CSV or use a built-in sample dataset.")

        sample_choice = st.selectbox(
            "Sample datasets",
            options=["-- Select sample --", *list(SAMPLE_FILES.keys())],
            index=0,
        )
        if sample_choice != "-- Select sample --":
            try:
                df = load_sample_dataframe(sample_choice)
                validate_dataframe(df)
                st.session_state["df"] = df
                st.session_state["dataset_name"] = sample_choice
                st.session_state["trained_model"] = None
                st.success(f"Loaded sample dataset: {sample_choice}")
                return df, sample_choice
            except Exception as exc:
                st.error(f"Failed to load sample: {exc}")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = load_dataset_from_upload(uploaded)
                st.session_state["df"] = df
                st.session_state["dataset_name"] = uploaded.name
                st.session_state["trained_model"] = None
                st.success(f"Loaded dataset: {uploaded.name}")
                return df, uploaded.name
            except Exception as exc:
                st.error(str(exc))

        if "df" in st.session_state and st.session_state["df"] is not None:
            return st.session_state["df"], st.session_state.get("dataset_name")

        return None, None


def render_overview_tab(df: pd.DataFrame) -> None:
    st.subheader("Dataset overview")
    overview = analysis.dataset_overview(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", overview["rows"])
    col2.metric("Columns", overview["columns"])
    col3.metric("Missing values", overview["missing_values"])
    col4.metric("Duplicate rows", overview["duplicate_rows"])

    st.markdown("### Data preview")
    st.dataframe(df.head(12), use_container_width=True)

    st.markdown("### Column profile")
    profile_df = pd.DataFrame(analysis.column_profile(df))
    if profile_df.empty:
        st.warning("No column profile is available for the current dataset.")
    else:
        st.dataframe(profile_df, use_container_width=True)


def render_eda_tab(df: pd.DataFrame) -> None:
    st.subheader("Exploratory data analysis")

    numerical_summary = analysis.numerical_summary(df)
    categorical_summary = analysis.categorical_summary(df)
    missing_values = analysis.missing_value_report(df)
    outliers = analysis.outlier_report(df)
    top_correlations = plots.top_correlated_pairs(df)

    if numerical_summary:
        st.markdown("### Numerical summary")
        st.dataframe(pd.DataFrame(numerical_summary), use_container_width=True)
    else:
        st.info("No numeric columns were found in this dataset.")

    if categorical_summary:
        st.markdown("### Categorical summary")
        st.dataframe(pd.DataFrame(categorical_summary), use_container_width=True)
    else:
        st.info("No categorical columns were found in this dataset.")

    if missing_values:
        st.markdown("### Missing value report")
        st.dataframe(pd.DataFrame(missing_values), use_container_width=True)
    else:
        st.success("No missing values detected.")

    if outliers:
        st.markdown("### Outlier report")
        st.dataframe(pd.DataFrame(outliers), use_container_width=True)
    else:
        st.success("No major outliers detected using the IQR method.")

    if top_correlations:
        st.markdown("### Top correlated pairs")
        st.dataframe(pd.DataFrame(top_correlations), use_container_width=True)
    else:
        st.info("Not enough numeric columns to compute pairwise correlations.")


def render_visualizations_tab(df: pd.DataFrame) -> None:
    st.subheader("Visualizations")
    charts = {
        "Histograms": plots.histogram_grid(df),
        "Boxplots": plots.boxplot_grid(df),
        "Correlation heatmap": plots.correlation_heatmap(df),
        "Categorical bar charts": plots.categorical_bar_charts(df),
    }

    chart_cols = st.columns(2)
    for index, (label, chart_data) in enumerate(charts.items()):
        with chart_cols[index % 2]:
            safe_display_image(chart_data, label)


def render_model_tab(df: pd.DataFrame) -> None:
    st.subheader("Machine learning")

    target = st.selectbox("Target column", options=df.columns.tolist(), index=0 if df.columns.size else None)
    feature_candidates = [col for col in df.columns if col != target]
    selected_features = st.multiselect("Feature columns", options=feature_candidates, default=feature_candidates[: min(5, len(feature_candidates))] if feature_candidates else [])

    if not selected_features:
        st.warning("Choose at least one feature column to train a model.")
        return

    problem_type_choice = st.selectbox(
        "Problem type",
        options=["Auto detect", "classification", "regression"],
        index=0,
    )
    inferred_problem = ml.infer_problem_type(df, target)

    if problem_type_choice == "Auto detect":
        problem_type = inferred_problem
    else:
        problem_type = problem_type_choice

    model_options = list(ml.CLASSIFICATION_MODELS.keys()) if problem_type == "classification" else list(ml.REGRESSION_MODELS.keys())
    model_name = st.selectbox("Model", options=model_options, index=0)
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    cv_folds = st.number_input("Cross-validation folds (optional)", min_value=0, max_value=10, value=0)

    if st.button("Train model"):
        try:
            trained = ml.train_model(
                df=df,
                target=target,
                features=selected_features,
                problem_type=problem_type,
                model_name=model_name,
                test_size=float(test_size),
                cv_folds=int(cv_folds) if cv_folds else None,
            )
            st.session_state["trained_model"] = trained
            st.success("Model trained successfully.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return

    trained_model = st.session_state.get("trained_model")
    if trained_model is None:
        st.info("Train a model to see metrics and predictions.")
        return

    st.markdown("### Metrics")
    st.json(trained_model.metrics)

    st.markdown("### Model charts")
    chart_cols = st.columns(2)
    for idx, (chart_name, chart_data) in enumerate(trained_model.charts.items()):
        with chart_cols[idx % 2]:
            safe_display_image(chart_data, chart_name.replace("_", " ").title())

    if trained_model.feature_importance:
        st.markdown("### Feature importance")
        st.dataframe(pd.DataFrame(trained_model.feature_importance), use_container_width=True)

    st.markdown("### Prediction")
    input_values: dict[str, object] = {}
    for feature in selected_features:
        value = df[feature].dropna().iloc[0] if df[feature].dropna().size else 0
        sample_value = value
        if pd.api.types.is_numeric_dtype(df[feature]):
            input_values[feature] = st.number_input(f"{feature}", value=float(sample_value))
        else:
            unique_values = list(df[feature].dropna().unique())
            options = unique_values[:10] if len(unique_values) > 10 else unique_values
            default = unique_values[0] if unique_values else ""
            input_values[feature] = st.selectbox(f"{feature}", options=options, index=0 if options else None)

    if st.button("Predict"):
        try:
            result = ml.predict(trained_model, input_values)
            st.success(f"Prediction: {result['prediction']}")
            if "probabilities" in result:
                st.json(result["probabilities"])
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def main() -> None:
    df, _ = sidebar_dataset_controls()
    if df is None:
        st.info("Load a sample dataset or upload a CSV file to start analyzing your data.")
        return

    st.subheader(f"Dataset: {st.session_state.get('dataset_name') or 'Loaded dataset'}")

    tabs = st.tabs(["Overview", "EDA", "Visualizations", "ML"])
    with tabs[0]:
        render_overview_tab(df)
    with tabs[1]:
        render_eda_tab(df)
    with tabs[2]:
        render_visualizations_tab(df)
    with tabs[3]:
        render_model_tab(df)


if __name__ == "__main__":
    main()
