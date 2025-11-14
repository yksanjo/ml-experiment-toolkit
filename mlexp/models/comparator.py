"""Model comparison utilities."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import warnings


class ModelComparison:
    """Container for model comparison results."""

    def __init__(self):
        self.results = []
        self.metrics_df = None
        self.task_type = None

    def __repr__(self):
        if self.metrics_df is not None:
            return f"ModelComparison({len(self.results)} models)"
        return "ModelComparison(empty)"

    def get_best_model(self, metric: Optional[str] = None) -> Dict:
        """Get the best performing model based on a metric."""
        if self.metrics_df is None or len(self.metrics_df) == 0:
            raise ValueError("No comparison results available")

        if metric is None:
            # Use default metric based on task type
            if self.task_type == "classification":
                metric = "accuracy"
            else:
                metric = "r2_score"

        if metric not in self.metrics_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        # For classification, higher is better; for regression, depends on metric
        ascending = metric in ["mse", "mae", "rmse"]

        best_idx = self.metrics_df[metric].idxmin() if ascending else self.metrics_df[metric].idxmax()
        best_result = self.results[best_idx]

        return {
            "model_name": best_result["model_name"],
            "metric": metric,
            "score": self.metrics_df.loc[best_idx, metric],
            "full_results": best_result,
        }


def compare_models(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    models: List[BaseEstimator],
    model_names: Optional[List[str]] = None,
    task_type: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    verbose: bool = True,
) -> ModelComparison:
    """
    Compare multiple models side-by-side with automated metrics.

    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    X_test : array-like
        Test features
    y_test : array-like
        Test targets
    models : list of sklearn estimators
        List of models to compare
    model_names : list of str, optional
        Names for the models. If None, uses model class names
    task_type : str, optional
        Task type: 'classification' or 'regression'. Auto-detected if None
    metrics : list of str, optional
        Custom metrics to compute. Uses defaults if None
    verbose : bool, default=True
        Whether to print comparison results

    Returns
    -------
    ModelComparison
        Comparison object with results and metrics dataframe

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> import mlexp
    >>>
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> models = [RandomForestClassifier(), GradientBoostingClassifier()]
    >>> results = mlexp.compare_models(X_train, y_train, X_test, y_test, models=models)
    """
    from mlexp.utils.helpers import detect_task_type, get_default_metrics

    # Auto-detect task type if not provided
    if task_type is None:
        task_type = detect_task_type(y_train)
    else:
        task_type = task_type.lower()

    if task_type not in ["classification", "regression"]:
        raise ValueError("task_type must be 'classification' or 'regression'")

    # Get default metrics if not provided
    if metrics is None:
        metrics = get_default_metrics(task_type)

    # Generate model names if not provided
    if model_names is None:
        model_names = [type(model).__name__ for model in models]
    elif len(model_names) != len(models):
        raise ValueError("model_names length must match models length")

    comparison = ModelComparison()
    comparison.task_type = task_type

    # Train and evaluate each model
    for model, name in zip(models, model_names):
        try:
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None

            # Get probability predictions for classification if available
            if task_type == "classification" and hasattr(model, "predict_proba"):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except Exception:
                    pass

            # Calculate metrics
            model_metrics = _calculate_metrics(
                y_test, y_pred, y_pred_proba, task_type, metrics
            )

            result = {
                "model_name": name,
                "model": model,
                "metrics": model_metrics,
                "predictions": y_pred,
                "predictions_proba": y_pred_proba,
            }
            comparison.results.append(result)

        except Exception as e:
            if verbose:
                warnings.warn(f"Error evaluating model '{name}': {str(e)}")
            continue

    # Create metrics dataframe
    if len(comparison.results) > 0:
        metrics_list = []
        for result in comparison.results:
            metrics_dict = {"model_name": result["model_name"]}
            metrics_dict.update(result["metrics"])
            metrics_list.append(metrics_dict)

        comparison.metrics_df = pd.DataFrame(metrics_list)
        comparison.metrics_df = comparison.metrics_df.set_index("model_name")

        # Print results if verbose
        if verbose:
            _print_comparison(comparison)

    return comparison


def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    task_type: str,
    metrics: List[str],
) -> Dict[str, float]:
    """Calculate specified metrics for predictions."""
    results = {}

    if task_type == "classification":
        for metric in metrics:
            metric_lower = metric.lower()
            try:
                if metric_lower == "accuracy":
                    results["accuracy"] = accuracy_score(y_true, y_pred)
                elif metric_lower in ["precision", "precision_score"]:
                    results["precision"] = precision_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )
                elif metric_lower in ["recall", "recall_score"]:
                    results["recall"] = recall_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )
                elif metric_lower in ["f1", "f1_score"]:
                    results["f1_score"] = f1_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    )
                elif metric_lower in ["roc_auc", "auc"]:
                    if y_pred_proba is not None:
                        # Handle binary and multiclass
                        if y_pred_proba.shape[1] == 2:
                            results["roc_auc"] = roc_auc_score(
                                y_true, y_pred_proba[:, 1]
                            )
                        else:
                            results["roc_auc"] = roc_auc_score(
                                y_true, y_pred_proba, multi_class="ovr", average="weighted"
                            )
                    else:
                        results["roc_auc"] = np.nan
            except Exception as e:
                results[metric] = np.nan

    else:  # regression
        for metric in metrics:
            metric_lower = metric.lower()
            try:
                if metric_lower in ["mse", "mean_squared_error"]:
                    results["mse"] = mean_squared_error(y_true, y_pred)
                elif metric_lower in ["mae", "mean_absolute_error"]:
                    results["mae"] = mean_absolute_error(y_true, y_pred)
                elif metric_lower in ["rmse"]:
                    results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
                elif metric_lower in ["r2", "r2_score"]:
                    results["r2_score"] = r2_score(y_true, y_pred)
            except Exception as e:
                results[metric] = np.nan

    return results


def _print_comparison(comparison: ModelComparison):
    """Print formatted comparison results."""
    print("=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(f"\nTask Type: {comparison.task_type.title()}")
    print(f"Number of Models: {len(comparison.results)}\n")

    if comparison.metrics_df is not None and len(comparison.metrics_df) > 0:
        # Format dataframe for display
        display_df = comparison.metrics_df.copy()
        for col in display_df.columns:
            if display_df[col].dtype in [np.float64, np.float32]:
                display_df[col] = display_df[col].round(4)

        print(display_df.to_string())
        print()

        # Show best model
        try:
            best = comparison.get_best_model()
            print(f"Best Model: {best['model_name']} ({best['metric']}={best['score']:.4f})")
        except Exception:
            pass

    print("=" * 60)

