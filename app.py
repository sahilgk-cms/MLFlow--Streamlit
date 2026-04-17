import streamlit as st
from utils.mlfow_helpers import initiate_client, get_all_models, get_model_versions, get_multiple_versions_context
from utils.graphs import rmse_comparison_between_model_version,  plot_rmse_vs_column, overfit_gap_bar_chart, cv_vs_test_scatter, parallel_cordinates, precision_recall_comparison, precision_recall_scatter, plot_precision_recall_vs_column
from config import MLFLOW_URI, FEATURES_DATA_COLS, ML_PARAMS_COLS

st.set_page_config(layout="wide")

st.title("📊 MLflow Model Dashboard (Minimal)")




client = initiate_client(MLFLOW_URI)

model_names = get_all_models(client)
selected_model = st.sidebar.selectbox(
    "Select Model",
    model_names
)

st.subheader(f"📌 Model: {selected_model}")

versions = get_model_versions(client, registered_model_name=selected_model)


st.header("📊 Model Comparison")

df = get_multiple_versions_context(client, selected_model, versions)
df["overfit_gap"] = df["cv_rmse"] - df["test_rmse"]

col1, col2 = st.columns(2)

with col1:
    rmse_model_fig = rmse_comparison_between_model_version(df)
    st.plotly_chart(rmse_model_fig, use_container_width=True)

with col2:
    overfit_gap_fig = overfit_gap_bar_chart(df)
    st.plotly_chart(overfit_gap_fig, use_container_width=True)  # overfit gap

cv_vs_test_fig = cv_vs_test_scatter(df)
st.plotly_chart(cv_vs_test_fig, use_container_width=True)

st.subheader("🎯 Prediction Performance (Precision / Recall)")

col_pr1, col_pr2 = st.columns(2)

with col_pr1:
    pr_bar_fig = precision_recall_comparison(df)
    st.plotly_chart(pr_bar_fig, use_container_width=True)

with col_pr2:
    pr_scatter_fig = precision_recall_scatter(df)
    st.plotly_chart(pr_scatter_fig, use_container_width=True)


st.subheader("🎯 RMSE vs features and ML config")
col3, col4 = st.columns(2)
with col3:
    selected_feature_col = st.selectbox(
        "Select Feature",
        FEATURES_DATA_COLS,
        key="features_config_for_rmse"
    )
    rmse_vs_col_fig1 = plot_rmse_vs_column(df, selected_feature_col)
    st.plotly_chart(rmse_vs_col_fig1, use_container_width=True)

with col4:
    selected_ml_param_col = st.selectbox(
        "Select ML Param",
        ML_PARAMS_COLS,
        key="ml_config_for_rmse"
    )
    rmse_vs_col_fig2 = plot_rmse_vs_column(df, selected_ml_param_col)
    st.plotly_chart(rmse_vs_col_fig2, use_container_width=True)


st.subheader("🎯 Precision-Recall vs features and ML config")
col3, col4 = st.columns(2)
with col3:
    selected_feature_col = st.selectbox(
        "Select Feature",
        FEATURES_DATA_COLS,
        key="features_config_for_prec_rec"
    )
    pr_vs_col_fig1 = plot_precision_recall_vs_column(df, selected_feature_col)
    st.plotly_chart(pr_vs_col_fig1, use_container_width=True)

with col4:
    selected_ml_param_col = st.selectbox(
        "Select ML Param",
        ML_PARAMS_COLS,
        key="ml_config_for_prec_rec"
    )
    pr_vs_col_fig2 = plot_precision_recall_vs_column(df, selected_ml_param_col)
    st.plotly_chart(pr_vs_col_fig2, use_container_width=True)


st.markdown("### 🏆 Best Model")
best = df.loc[df["test_rmse"].idxmin()]
col_center = st.columns([1,2,1])[1]

with col_center:
    st.metric(
        label="Best Model (Test RMSE)",
        value=f"v{best['version']}",
        delta=f"{best['test_rmse']:.3f}"
    )

st.subheader("Parallel co-ordinates")
col5, col6 = st.columns(2)

with col5:
    parlel_cord_features_fig = parallel_cordinates(df, FEATURES_DATA_COLS)
    st.plotly_chart(parlel_cord_features_fig, use_container_width=True)

with col6:
    ml_features_fig = parallel_cordinates(df, ML_PARAMS_COLS)
    st.plotly_chart(ml_features_fig, use_container_width=True)

with st.expander("🔍 View Full DataFrame"):
    st.dataframe(df, use_container_width=True)

st.download_button(
    "📥 Download Data",
    df.to_csv(index=False),
    f"{selected_model}_data.csv"
)