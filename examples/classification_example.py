"""
Classification example using ML Experiment Toolkit.

This example demonstrates a complete classification workflow with
data profiling, model comparison, and experiment tracking.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import mlexp

# Load Iris dataset
print("Loading Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Convert to DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y
df["target_name"] = [iris.target_names[t] for t in y]

# Step 1: Comprehensive Data Profiling
print("\n" + "=" * 60)
print("DATA PROFILING")
print("=" * 60)
profile = mlexp.profile(df, show_plot=True)

# Check for quality issues
if len(profile.quality_issues) > 0:
    print("\nQuality Issues Detected:")
    for issue in profile.quality_issues:
        print(f"  - {issue['message']}")

# Step 2: Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 3: Compare Multiple Models
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    AdaBoostClassifier(n_estimators=50, random_state=42),
    SVC(probability=True, random_state=42),
    KNeighborsClassifier(n_neighbors=5),
]

model_names = [
    "Random Forest",
    "Gradient Boosting",
    "AdaBoost",
    "SVM",
    "K-Nearest Neighbors",
]

comparison = mlexp.compare_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models=models,
    model_names=model_names,
    verbose=True,
)

# Step 4: Visualize Results
print("\n" + "=" * 60)
print("VISUALIZATIONS")
print("=" * 60)

# Full comparison plot
mlexp.plot_model_comparison(comparison)

# Focused metric comparison
print("\nPlotting accuracy comparison...")
mlexp.plot_metric_comparison(comparison, metric="accuracy")

# Step 5: Track Experiments
print("\n" + "=" * 60)
print("EXPERIMENT TRACKING")
print("=" * 60)

for i, result in enumerate(comparison.results):
    mlexp.track_experiment(
        name=f"iris_{result['model_name'].lower().replace(' ', '_')}",
        metrics=result["metrics"],
        params={
            "model": result["model_name"],
            "dataset": "iris",
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
        },
        tags=["classification", "iris", "multiclass"],
        notes=f"Classification experiment on Iris dataset using {result['model_name']}",
    )

# Query experiments
print("\nAll tracked experiments:")
experiments = mlexp.get_experiments(tags=["iris"])
print(experiments[["name", "timestamp", "metrics"]].head())

# Get best experiment
print("\nBest experiment by accuracy:")
tracker = mlexp.ExperimentTracker()
best = tracker.get_best_experiment("accuracy", higher_is_better=True)
print(f"  Model: {best['name']}")
print(f"  Accuracy: {best['metric_value']:.4f}")
print(f"  Parameters: {best['params']}")

print("\n" + "=" * 60)
print("Classification example complete!")
print("=" * 60)

