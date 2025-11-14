"""
Quickstart example for ML Experiment Toolkit.

This example demonstrates the basic usage of the toolkit for a simple
machine learning workflow.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import mlexp

# Generate sample data
print("Generating sample data...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42,
)

# Convert to DataFrame for better profiling
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target"] = y

# Step 1: Data Profiling
print("\n" + "=" * 60)
print("STEP 1: Data Profiling")
print("=" * 60)
profile = mlexp.profile(df, show_plot=True)

# Step 2: Prepare data for modeling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Compare Models
print("\n" + "=" * 60)
print("STEP 2: Model Comparison")
print("=" * 60)
models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    LogisticRegression(max_iter=1000, random_state=42),
]

comparison = mlexp.compare_models(
    X_train, y_train, X_test, y_test, models=models, verbose=True
)

# Step 4: Visualize Comparison
print("\n" + "=" * 60)
print("STEP 3: Visualization")
print("=" * 60)
mlexp.plot_model_comparison(comparison)

# Step 5: Track Experiments
print("\n" + "=" * 60)
print("STEP 4: Experiment Tracking")
print("=" * 60)
for result in comparison.results:
    mlexp.track_experiment(
        name=result["model_name"],
        metrics=result["metrics"],
        params={"model_type": result["model_name"]},
        tags=["quickstart", "classification"],
    )

# View tracked experiments
experiments = mlexp.get_experiments()
print("\nTracked Experiments:")
print(experiments[["name", "timestamp", "metrics"]].head())

print("\n" + "=" * 60)
print("Quickstart complete!")
print("=" * 60)

