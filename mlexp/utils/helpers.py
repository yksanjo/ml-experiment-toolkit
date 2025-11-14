"""Common utility functions."""

import numpy as np
import pandas as pd
from typing import List, Union


def detect_task_type(y: Union[pd.Series, np.ndarray]) -> str:
    """
    Automatically detect if the task is classification or regression.

    Parameters
    ----------
    y : array-like
        Target variable

    Returns
    -------
    str
        'classification' or 'regression'
    """
    y_array = np.asarray(y)

    # Check if it's numeric
    if not np.issubdtype(y_array.dtype, np.number):
        return "classification"

    # Check if it's integer and has few unique values (likely classification)
    if np.issubdtype(y_array.dtype, np.integer):
        unique_count = len(np.unique(y_array))
        total_count = len(y_array)
        # If less than 20 unique values and they represent a small fraction, likely classification
        if unique_count < 20 and unique_count / total_count < 0.1:
            return "classification"

    # Check if values are in a small discrete set (even if float)
    unique_count = len(np.unique(y_array))
    if unique_count < 20:
        # Check if values look like class labels (0, 1, 2, ... or similar)
        unique_vals = np.sort(np.unique(y_array))
        if len(unique_vals) <= 10:
            # Check if they're roughly sequential integers
            if np.allclose(unique_vals, np.arange(len(unique_vals))):
                return "classification"

    return "regression"


def get_default_metrics(task_type: str) -> List[str]:
    """
    Get default metrics for a task type.

    Parameters
    ----------
    task_type : str
        'classification' or 'regression'

    Returns
    -------
    list of str
        List of metric names
    """
    if task_type.lower() == "classification":
        return ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    else:  # regression
        return ["mse", "mae", "rmse", "r2_score"]

