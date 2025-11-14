"""
Regression example using ML Experiment Toolkit.

This example demonstrates a complete regression workflow with
data profiling, model comparison, and experiment tracking.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

import mlexp

# Load Diabetes dataset
print("Loading Diabetes dataset...")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Convert to DataFrame
df = pd.DataFrame(X, columns=diabetes.feature_names)
df["target"] = y

# Step 1: Data Profiling
print("\n" + "=" * 60)
print("DATA PROFILING")
print("=" * 60)
profile = mlexp.profile(df, show_plot=True)

# Step 2: Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Compare Multiple Regression Models
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

models = [
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(n_estimators=100, random_state=42),
    AdaBoostRegressor(n_estimators=50, random_state=42),
    LinearRegression(),
    Ridge(alpha=1.0),
    SVR(kernel="rbf"),
]

model_names = [
    "Random Forest",
    "Gradient Boosting",
    "AdaBoost",
    "Linear Regression",
    "Ridge Regression",
    "SVR",
]

comparison = mlexp.compare_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models=models,
    model_names=model_names,
    task_type="regression",
    verbose=True,
)

# Step 4: Visualize Results
print("\n" + "=" * 60)
print("VISUALIZATIONS")
print("=" * 60)

# Full comparison plot
mlexp.plot_model_comparison(comparison)

# Focused metric comparison (R² score)
print("\nPlotting R² score comparison...")
mlexp.plot_metric_comparison(comparison, metric="r2_score")

# Step 5: Track Experiments with Hyperparameter Variations
print("\n" + "=" * 60)
print("EXPERIMENT TRACKING")
print("=" * 60)

# Track each model
for result in comparison.results:
    mlexp.track_experiment(
        name=f"diabetes_{result['model_name'].lower().replace(' ', '_')}",
        metrics=result["metrics"],
        params={
            "model": result["model_name"],
            "dataset": "diabetes",
            "n_features": X.shape[1],
        },
        tags=["regression", "diabetes"],
        notes=f"Regression experiment on Diabetes dataset using {result['model_name']}",
    )

# Run additional experiments with different hyperparameters
print("\nRunning additional experiments with different hyperparameters...")

# Random Forest with different n_estimators
for n_est in [50, 100, 200]:
    model = RandomForestRegressor(n_estimators=n_est, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2_score": r2_score(y_test, y_pred),
    }

    mlexp.track_experiment(
        name=f"diabetes_rf_n{n_est}",
        metrics=metrics,
        params={"model": "RandomForestRegressor", "n_estimators": n_est},
        tags=["regression", "diabetes", "hyperparameter_tuning"],
    )

# Query and visualize experiment history
print("\nExperiment history for Random Forest models:")
experiments = mlexp.get_experiments(name="rf")
print(experiments[["name", "timestamp", "metrics"]].head())

# Plot experiment history
print("\nPlotting experiment history for R² score...")
mlexp.plot_experiment_history(metric="r2_score", name="rf")

# Get best experiment
print("\nBest experiment by R² score:")
tracker = mlexp.ExperimentTracker()
best = tracker.get_best_experiment("r2_score", higher_is_better=True)
print(f"  Model: {best['name']}")
print(f"  R² Score: {best['metric_value']:.4f}")
print(f"  Parameters: {best['params']}")

print("\n" + "=" * 60)
print("Regression example complete!")
print("=" * 60)

