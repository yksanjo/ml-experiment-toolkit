"""Visualization utilities for experiment tracking."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, Tuple, List
from mlexp.models.tracker import ExperimentTracker, get_experiments


def plot_experiment_history(
    metric: str,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    tracker: Optional[ExperimentTracker] = None,
) -> None:
    """
    Plot experiment history for a specific metric over time.

    Parameters
    ----------
    metric : str
        Metric name to plot
    name : str, optional
        Filter by experiment name
    tags : list of str, optional
        Filter by tags
    figsize : tuple, default=(12, 6)
        Figure size
    tracker : ExperimentTracker, optional
        Custom tracker instance. Uses default if None.

    Examples
    --------
    >>> import mlexp
    >>> mlexp.plot_experiment_history(metric='accuracy')
    """
    from mlexp.models.tracker import _get_default_tracker

    if tracker is None:
        tracker = _get_default_tracker()

    df = tracker.get_experiments(name=name, tags=tags)

    if len(df) == 0:
        raise ValueError("No experiments found")

    # Extract metric values
    metric_values = []
    timestamps = []
    exp_names = []

    for _, row in df.iterrows():
        if row["metrics"] and metric in row["metrics"]:
            metric_values.append(row["metrics"][metric])
            timestamps.append(pd.to_datetime(row["timestamp"]))
            exp_names.append(row["name"])

    if len(metric_values) == 0:
        raise ValueError(f"Metric '{metric}' not found in any experiments")

    # Create dataframe for plotting
    plot_df = pd.DataFrame({
        "timestamp": timestamps,
        "metric_value": metric_values,
        "name": exp_names,
    }).sort_values("timestamp")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    # Plot line chart
    ax.plot(plot_df["timestamp"], plot_df["metric_value"], marker="o", linewidth=2, markersize=8)

    # Add labels for each point
    for i, (ts, val, name) in enumerate(zip(plot_df["timestamp"], plot_df["metric_value"], plot_df["name"])):
        ax.annotate(
            name,
            (ts, val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel("Timestamp", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Experiment History: {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def plot_experiment_comparison(
    metric: str,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    tracker: Optional[ExperimentTracker] = None,
) -> None:
    """
    Create a bar chart comparing experiments by a metric.

    Parameters
    ----------
    metric : str
        Metric name to compare
    name : str, optional
        Filter by experiment name
    tags : list of str, optional
        Filter by tags
    figsize : tuple, default=(10, 6)
        Figure size
    tracker : ExperimentTracker, optional
        Custom tracker instance. Uses default if None.

    Examples
    --------
    >>> mlexp.plot_experiment_comparison(metric='accuracy')
    """
    from mlexp.models.tracker import _get_default_tracker

    if tracker is None:
        tracker = _get_default_tracker()

    df = tracker.get_experiments(name=name, tags=tags)

    if len(df) == 0:
        raise ValueError("No experiments found")

    # Extract metric values
    exp_data = []
    for _, row in df.iterrows():
        if row["metrics"] and metric in row["metrics"]:
            exp_data.append({
                "name": row["name"],
                "value": row["metrics"][metric],
            })

    if len(exp_data) == 0:
        raise ValueError(f"Metric '{metric}' not found in any experiments")

    plot_df = pd.DataFrame(exp_data).sort_values("value", ascending=False)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("husl", len(plot_df))
    bars = ax.barh(plot_df["name"], plot_df["value"], color=colors)

    # Add value labels
    for i, (name, val) in enumerate(zip(plot_df["name"], plot_df["value"])):
        ax.text(val, i, f" {val:.4f}", va="center", fontweight="bold")

    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Experiment Comparison: {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

