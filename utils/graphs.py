import plotly.graph_objects as go
import pandas as pd
import plotly
import plotly.express as px
from typing import List
import ast


def rmse_comparison_between_model_version(df: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x = df["version"],
            y = df["test_rmse"],
            name = "Test RMSE"
        )
    )

    fig.add_trace(
        go.Bar(
            x = df["version"],
            y = df["cv_rmse"],
            name = "CV RMSE"
        )
    )

    fig.update_layout(
        barmode="group",
        xaxis_title="Model Version",
        yaxis_title="RMSE",
        title="Model Performance Comparison",
        template="plotly_white"
    )

    fig.update_traces(
        texttemplate='%{y:.2f}',
        textposition='outside'
    )

    return fig

def precision_recall_comparison(df: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["version"],
            y=df["precision"],
            name="Precision"
        )
    )

    fig.add_trace(
        go.Bar(
            x=df["version"],
            y=df["recall"],
            name="Recall"
        )
    )

    fig.update_layout(
        barmode="group",
        xaxis_title="Model Version",
        yaxis_title="Score",
        title="Precision vs Recall Comparison",
        template="plotly_white"
    )

    fig.update_traces(
        texttemplate='%{y:.3f}',
        textposition='outside'
    )

    return fig


def plot_rmse_vs_column(df: pd.DataFrame, column: str):
    df = df.copy()
    df = df.sort_values(by=column)
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df[column],
            y=df["test_rmse"],
            mode="lines+markers",
            name="Test RMSE",
            text = df['version'],
            hovertemplate=
                f"{column}: %{{x}}<br>"
                "Test RMSE: %{y:.3f}<br>"
                "Version: %{text}<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[column],
            y=df["cv_rmse"],
            mode="lines+markers",
            name="CV RMSE",
            text = df['version'],
            hovertemplate=
                f"{column}: %{{x}}<br>"
                "CV RMSE: %{y:.3f}<br>"
                "Version: %{text}<extra></extra>"
        )
    )

    fig.update_layout(
        title=f"RMSE vs {column}",
        xaxis_title=column,
        yaxis_title="RMSE",
        template="plotly_white",
         legend=dict(
            orientation="h",
            y=1.1,
            x=1,
            xanchor="right"
        )
       
    )

    return fig



def plot_precision_recall_vs_column(df: pd.DataFrame, column: str):
    df = df.copy()
    df = df.sort_values(by=column)
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df[column],
            y=df["precision"],
            mode="lines+markers",
            name="Precision",
            text = df['version'],
            hovertemplate=
                f"{column}: %{{x}}<br>"
                "Precision: %{y:.3f}<br>"
                "Version: %{text}<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[column],
            y=df["recall"],
            mode="lines+markers",
            name="Recall",
            text = df['version'],
            hovertemplate=
                f"{column}: %{{x}}<br>"
                "Recall: %{y:.3f}<br>"
                "Version: %{text}<extra></extra>"
        )
    )

    fig.update_layout(
        title=f"RMSE vs {column}",
        xaxis_title=column,
        yaxis_title="Precision-Recall",
        template="plotly_white",
         legend=dict(
            orientation="h",
            y=1.1,
            x=1,
            xanchor="right"
        )
       
    )

    return fig


def overfit_gap_bar_chart(df: pd.DataFrame):
    fig = px.bar(
        df,
        x="version",
        y="overfit_gap",
        title="Overfitting Gap (CV - Test RMSE)",
    )
    return fig

def cv_vs_test_scatter(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="cv_rmse",
        y="test_rmse",
        text="version",
        title="CV vs Test RMSE"
    )
    fig.update_traces(textposition='top center')
    return fig


def parallel_cordinates(df: pd.DataFrame, columns: List[str]):
    df = df.copy()

    clean_cols = []

    for col in columns:

        # try numeric conversions
        df[col] = pd.to_numeric(df[col], errors="ignore")

        # handle list like strings
        if df[col].dtype == "object":
            try:
                df[col] = df[col].apply(
                    lambda x: sum(ast.literal_eval(x)) if isinstance(x, str) and x.startswith("[") else x
                )
            except:
                continue

        # keep only numeric cols
        if pd.api.types.is_numeric_dtype(df[col]):
            clean_cols.append(col)

    #normalize
    # df[clean_cols] =  (df[clean_cols] - df[clean_cols].min()) / (
    #     df[clean_cols].max() - df[clean_cols].min() + 1e-9
    # )
    dimensions = []
    for col in clean_cols:
        dimensions.append(
            dict(
                label=col,
                values=df[col],
                tickvals=[df[col].min(), df[col].max()],
                ticktext=[
                    f"{df[col].min():.3f}",
                    f"{df[col].max():.3f}"
                ]
            )
        )

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(color=df["test_rmse"]),
            dimensions=dimensions
        )
    )

    fig = px.parallel_coordinates(
        df,
        dimensions=columns,
        color="test_rmse",
        color_continuous_scale=px.colors.diverging.Tealrose
        )
    
    fig.update_layout(
        title="Parallel Coordinates Plot",
        template="plotly_white",
        width=1200
    )

    fig.update_traces(
        labelfont=dict(size=12),
        tickfont=dict(size=10)
    )

    return fig

def precision_recall_scatter(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="precision",
        y="recall",
        text="version",
        title="Precision vs Recall Tradeoff"
    )

    fig.update_traces(textposition='top center')

    return fig