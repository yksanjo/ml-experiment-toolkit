"""
ML Experiment Toolkit - A lightweight library for rapid ML experimentation.
"""

__version__ = "0.1.0"

from mlexp.data.profiler import profile
from mlexp.models.comparator import compare_models, ModelComparison
from mlexp.models.tracker import track_experiment, get_experiments, ExperimentTracker
from mlexp.viz.comparison import plot_model_comparison, plot_metric_comparison
from mlexp.viz.experiments import plot_experiment_history, plot_experiment_comparison

__all__ = [
    "profile",
    "compare_models",
    "ModelComparison",
    "track_experiment",
    "get_experiments",
    "ExperimentTracker",
    "plot_model_comparison",
    "plot_metric_comparison",
    "plot_experiment_history",
    "plot_experiment_comparison",
]

