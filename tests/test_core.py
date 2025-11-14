"""Basic tests for ML Experiment Toolkit."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

import mlexp
from mlexp.data.profiler import profile
from mlexp.models.comparator import compare_models
from mlexp.models.tracker import track_experiment, get_experiments, ExperimentTracker
from mlexp.utils.helpers import detect_task_type, get_default_metrics


def test_profile():
    """Test data profiling functionality."""
    # Create sample data
    df = pd.DataFrame({
        "numeric1": np.random.randn(100),
        "numeric2": np.random.randn(100),
        "categorical": np.random.choice(["A", "B", "C"], 100),
    })
    df.loc[0:5, "numeric1"] = np.nan  # Add some missing values

    profile_obj = profile(df, show_plot=False)
    assert profile_obj.shape == (100, 3)
    assert len(profile_obj.missing_values) > 0
    assert profile_obj.duplicate_rows >= 0


def test_compare_models_classification():
    """Test model comparison for classification."""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        RandomForestClassifier(n_estimators=10, random_state=42),
        RandomForestClassifier(n_estimators=20, random_state=42),
    ]

    comparison = compare_models(X_train, y_train, X_test, y_test, models=models, verbose=False)
    assert len(comparison.results) == 2
    assert comparison.metrics_df is not None
    assert "accuracy" in comparison.metrics_df.columns


def test_compare_models_regression():
    """Test model comparison for regression."""
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        RandomForestRegressor(n_estimators=10, random_state=42),
        RandomForestRegressor(n_estimators=20, random_state=42),
    ]

    comparison = compare_models(
        X_train, y_train, X_test, y_test, models=models, task_type="regression", verbose=False
    )
    assert len(comparison.results) == 2
    assert "r2_score" in comparison.metrics_df.columns


def test_track_experiment():
    """Test experiment tracking."""
    # Use a temporary database
    tracker = ExperimentTracker(db_path=":memory:")
    exp_id = tracker.track(
        name="test_exp",
        metrics={"accuracy": 0.95},
        params={"n_estimators": 100},
    )
    assert exp_id > 0

    experiments = tracker.get_experiments()
    assert len(experiments) == 1
    assert experiments.iloc[0]["name"] == "test_exp"


def test_get_experiments():
    """Test querying experiments."""
    tracker = ExperimentTracker(db_path=":memory:")
    tracker.track(name="exp1", metrics={"accuracy": 0.9}, tags=["test"])
    tracker.track(name="exp2", metrics={"accuracy": 0.95}, tags=["test", "baseline"])

    # Query by tags
    results = tracker.get_experiments(tags=["baseline"])
    assert len(results) == 1
    assert results.iloc[0]["name"] == "exp2"

    # Query by name
    results = tracker.get_experiments(name="exp1")
    assert len(results) == 1


def test_detect_task_type():
    """Test task type detection."""
    # Classification (integer labels)
    y_class = np.array([0, 1, 0, 1, 2, 0, 1])
    assert detect_task_type(y_class) == "classification"

    # Regression (continuous values)
    y_reg = np.random.randn(100)
    assert detect_task_type(y_reg) == "regression"


def test_get_default_metrics():
    """Test default metrics retrieval."""
    cls_metrics = get_default_metrics("classification")
    assert "accuracy" in cls_metrics

    reg_metrics = get_default_metrics("regression")
    assert "r2_score" in reg_metrics


def test_best_model():
    """Test getting best model from comparison."""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        RandomForestClassifier(n_estimators=10, random_state=42),
        RandomForestClassifier(n_estimators=50, random_state=42),
    ]

    comparison = compare_models(X_train, y_train, X_test, y_test, models=models, verbose=False)
    best = comparison.get_best_model(metric="accuracy")
    assert "model_name" in best
    assert "score" in best


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

