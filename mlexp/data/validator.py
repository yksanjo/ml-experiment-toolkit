"""Data validation helpers."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable


def validate_data(
    df: pd.DataFrame,
    checks: Optional[List[str]] = None,
    custom_checks: Optional[List[Callable]] = None,
) -> Dict[str, bool]:
    """
    Run validation checks on a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate
    checks : list of str, optional
        List of built-in checks to run. Options:
        - 'no_missing': Check for missing values
        - 'no_duplicates': Check for duplicate rows
        - 'numeric_range': Check numeric columns are in valid ranges
        - 'categorical_values': Check categorical columns have expected values
    custom_checks : list of callables, optional
        Custom validation functions. Each should take a DataFrame and return bool.

    Returns
    -------
    dict
        Dictionary mapping check names to pass/fail status

    Examples
    --------
    >>> import mlexp.data.validator as validator
    >>> results = validator.validate_data(df, checks=['no_missing', 'no_duplicates'])
    """
    if checks is None:
        checks = ["no_missing", "no_duplicates"]

    results = {}

    # Built-in checks
    if "no_missing" in checks:
        results["no_missing"] = df.isnull().sum().sum() == 0

    if "no_duplicates" in checks:
        results["no_duplicates"] = df.duplicated().sum() == 0

    if "numeric_range" in checks:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results["numeric_range"] = all(
            df[col].notna().all() or df[col].between(-1e10, 1e10).all()
            for col in numeric_cols
        )

    # Custom checks
    if custom_checks:
        for i, check_func in enumerate(custom_checks):
            check_name = f"custom_check_{i}"
            try:
                results[check_name] = check_func(df)
            except Exception as e:
                results[check_name] = False
                results[f"{check_name}_error"] = str(e)

    return results


def check_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detect outliers in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    method : str, default='iqr'
        Method to use: 'iqr' (interquartile range) or 'zscore'
    threshold : float, default=3.0
        Threshold for outlier detection

    Returns
    -------
    pd.DataFrame
        Boolean dataframe indicating outlier positions
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)

    for col in numeric_cols:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask[col] = z_scores > threshold

    return outlier_mask

