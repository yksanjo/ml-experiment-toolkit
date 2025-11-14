"""Visualization utilities for model comparison."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from mlexp.models.comparator import ModelComparison


def plot_model_comparison(
    comparison: ModelComparison,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    style: str = "whitegrid",
) -> None:
    """
    Create visualizations for model comparison results.

    Parameters
    ----------
    comparison : ModelComparison
        ModelComparison object with results
    metrics : list of str, optional
        Specific metrics to plot. Uses all available if None.
    figsize : tuple, default=(12, 6)
        Figure size
    style : str, default='whitegrid'
        Seaborn style

    Examples
    --------
    >>> import mlexp
    >>> comparison = mlexp.compare_models(X_train, y_train, X_test, y_test, models)
    >>> mlexp.plot_model_comparison(comparison)
    """
    if comparison.metrics_df is None or len(comparison.metrics_df) == 0:
        raise ValueError("No comparison results to visualize")

    sns.set_style(style)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    df = comparison.metrics_df.copy()

    # Select metrics to plot
    if metrics is None:
        metrics = df.columns.tolist()
    else:
        metrics = [m for m in metrics if m in df.columns]

    if len(metrics) == 0:
        raise ValueError("No valid metrics found to plot")

    # Plot 1: Bar chart comparing metrics
    ax1 = axes[0]
    df_plot = df[metrics].T
    df_plot.plot(kind="bar", ax=ax1, width=0.8)
    ax1.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Metrics", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Plot 2: Heatmap of metrics
    ax2 = axes[1]
    sns.heatmap(
        df[metrics].T,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        center=0.5,
        ax=ax2,
        cbar_kws={"label": "Score"},
    )
    ax2.set_title("Model Performance Heatmap", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Models", fontsize=12)
    ax2.set_ylabel("Metrics", fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_metric_comparison(
    comparison: ModelComparison,
    metric: str,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Create a focused comparison plot for a single metric.

    Parameters
    ----------
    comparison : ModelComparison
        ModelComparison object with results
    metric : str
        Metric name to plot
    figsize : tuple, default=(8, 6)
        Figure size

    Examples
    --------
    >>> mlexp.plot_metric_comparison(comparison, metric='accuracy')
    """
    if comparison.metrics_df is None or len(comparison.metrics_df) == 0:
        raise ValueError("No comparison results to visualize")

    if metric not in comparison.metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    df = comparison.metrics_df
    values = df[metric].sort_values(ascending=False)

    colors = sns.color_palette("husl", len(values))
    bars = ax.barh(values.index, values.values, color=colors)

    # Add value labels on bars
    for i, (idx, val) in enumerate(zip(values.index, values.values)):
        ax.text(val, i, f" {val:.4f}", va="center", fontweight="bold")

    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Model Comparison: {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

