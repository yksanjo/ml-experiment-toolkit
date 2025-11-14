"""Data profiling and quality checks."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class DataProfile:
    """Container for data profiling results."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.shape = df.shape
        self.missing_values = None
        self.duplicate_rows = None
        self.data_types = None
        self.numeric_summary = None
        self.categorical_summary = None
        self.quality_issues = []

    def __repr__(self):
        return f"DataProfile(shape={self.shape}, issues={len(self.quality_issues)})"


def profile(
    df: pd.DataFrame,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 8),
) -> DataProfile:
    """
    Generate a comprehensive data profile with quality checks and visualizations.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to profile
    show_plot : bool, default=True
        Whether to display visualization plots
    figsize : tuple, default=(12, 8)
        Figure size for plots

    Returns
    -------
    DataProfile
        Profile object containing all analysis results

    Examples
    --------
    >>> import pandas as pd
    >>> import mlexp
    >>> df = pd.read_csv('data.csv')
    >>> profile = mlexp.profile(df)
    """
    profile_obj = DataProfile(df)

    # Basic statistics
    profile_obj.shape = df.shape
    profile_obj.data_types = df.dtypes.value_counts().to_dict()

    # Missing values analysis
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    profile_obj.missing_values = pd.DataFrame({
        "count": missing,
        "percentage": missing_pct
    }).sort_values("count", ascending=False)
    profile_obj.missing_values = profile_obj.missing_values[
        profile_obj.missing_values["count"] > 0
    ]

    # Duplicate rows
    profile_obj.duplicate_rows = df.duplicated().sum()

    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        profile_obj.numeric_summary = df[numeric_cols].describe()

    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        cat_summary = {}
        for col in categorical_cols:
            cat_summary[col] = {
                "unique_count": df[col].nunique(),
                "top_value": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                "top_frequency": df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0,
            }
        profile_obj.categorical_summary = pd.DataFrame(cat_summary).T

    # Quality issues detection
    profile_obj.quality_issues = _detect_quality_issues(profile_obj)

    # Print summary
    _print_profile_summary(profile_obj)

    # Generate visualizations
    if show_plot:
        _plot_profile(profile_obj, figsize)

    return profile_obj


def _detect_quality_issues(profile: DataProfile) -> List[Dict]:
    """Detect data quality issues."""
    issues = []

    # Missing values
    if profile.missing_values is not None and len(profile.missing_values) > 0:
        high_missing = profile.missing_values[
            profile.missing_values["percentage"] > 50
        ]
        if len(high_missing) > 0:
            issues.append({
                "type": "high_missing_values",
                "severity": "warning",
                "message": f"{len(high_missing)} columns have >50% missing values",
                "columns": high_missing.index.tolist(),
            })

    # Duplicate rows
    if profile.duplicate_rows > 0:
        dup_pct = (profile.duplicate_rows / profile.shape[0]) * 100
        severity = "error" if dup_pct > 10 else "warning"
        issues.append({
            "type": "duplicate_rows",
            "severity": severity,
            "message": f"{profile.duplicate_rows} duplicate rows ({dup_pct:.1f}%)",
        })

    # Low variance numeric columns
    if profile.numeric_summary is not None:
        numeric_cols = profile.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            std = profile.df[col].std()
            mean = abs(profile.df[col].mean())
            if mean > 0 and std / mean < 0.01:  # Coefficient of variation < 1%
                issues.append({
                    "type": "low_variance",
                    "severity": "info",
                    "message": f"Column '{col}' has very low variance",
                    "columns": [col],
                })

    # High cardinality categorical columns
    if profile.categorical_summary is not None:
        high_card = profile.categorical_summary[
            profile.categorical_summary["unique_count"] > profile.shape[0] * 0.9
        ]
        if len(high_card) > 0:
            issues.append({
                "type": "high_cardinality",
                "severity": "warning",
                "message": f"{len(high_card)} categorical columns have very high cardinality",
                "columns": high_card.index.tolist(),
            })

    return issues


def _print_profile_summary(profile: DataProfile):
    """Print a formatted summary of the profile."""
    print("=" * 60)
    print("DATA PROFILE SUMMARY")
    print("=" * 60)
    print(f"\nShape: {profile.shape[0]} rows × {profile.shape[1]} columns")
    print(f"Memory usage: {profile.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n" + "-" * 60)
    print("DATA TYPES")
    print("-" * 60)
    for dtype, count in profile.data_types.items():
        print(f"  {dtype}: {count} columns")

    if profile.missing_values is not None and len(profile.missing_values) > 0:
        print("\n" + "-" * 60)
        print("MISSING VALUES")
        print("-" * 60)
        print(profile.missing_values.to_string())

    if profile.duplicate_rows > 0:
        print("\n" + "-" * 60)
        print(f"DUPLICATE ROWS: {profile.duplicate_rows}")

    if len(profile.quality_issues) > 0:
        print("\n" + "-" * 60)
        print("QUALITY ISSUES")
        print("-" * 60)
        for issue in profile.quality_issues:
            severity_symbol = {
                "error": "❌",
                "warning": "⚠️",
                "info": "ℹ️",
            }.get(issue["severity"], "•")
            print(f"{severity_symbol} [{issue['severity'].upper()}] {issue['message']}")

    print("\n" + "=" * 60)


def _plot_profile(profile: DataProfile, figsize: Tuple[int, int]):
    """Generate visualization plots for the profile."""
    n_plots = 0
    plots_to_show = []

    # Missing values plot
    if profile.missing_values is not None and len(profile.missing_values) > 0:
        n_plots += 1
        plots_to_show.append(("missing", profile.missing_values))

    # Numeric columns distribution
    numeric_cols = profile.df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        n_plots += 1
        plots_to_show.append(("numeric", numeric_cols))

    # Categorical columns
    categorical_cols = profile.df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0 and len(categorical_cols) <= 5:
        n_plots += 1
        plots_to_show.append(("categorical", categorical_cols))

    if n_plots == 0:
        return

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = figsize

    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot missing values
    if any(p[0] == "missing" for p in plots_to_show):
        missing_data = next(p[1] for p in plots_to_show if p[0] == "missing")
        ax = axes[plot_idx]
        missing_data["percentage"].plot(kind="barh", ax=ax, color="coral")
        ax.set_xlabel("Missing Percentage (%)")
        ax.set_title("Missing Values by Column")
        ax.grid(axis="x", alpha=0.3)
        plot_idx += 1

    # Plot numeric distributions
    if any(p[0] == "numeric" for p in plots_to_show):
        numeric_cols = next(p[1] for p in plots_to_show if p[0] == "numeric")
        ax = axes[plot_idx]
        # Show distribution of first few numeric columns
        cols_to_plot = numeric_cols[:4]  # Limit to 4 columns
        profile.df[cols_to_plot].hist(bins=30, ax=ax, alpha=0.7)
        ax.set_title("Numeric Columns Distribution")
        ax.legend(cols_to_plot)
        plot_idx += 1

    # Plot categorical distributions
    if any(p[0] == "categorical" for p in plots_to_show):
        cat_cols = next(p[1] for p in plots_to_show if p[0] == "categorical")
        ax = axes[plot_idx]
        # Show value counts for first categorical column
        if len(cat_cols) > 0:
            top_values = profile.df[cat_cols[0]].value_counts().head(10)
            top_values.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title(f"Top Values in '{cat_cols[0]}'")
            ax.set_xlabel(cat_cols[0])
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
        plot_idx += 1

    plt.tight_layout()
    plt.show()

