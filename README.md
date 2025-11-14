# ML Experiment Toolkit

A lightweight Python library for rapid ML experimentation that combines data profiling, automated model comparison, experiment tracking, and built-in visualizations in a simple, opinionated API.

## Features

- **Data Profiling**: Quick data quality checks, statistical summaries, and visualizations
- **Model Comparison**: Compare multiple models side-by-side with automated metrics
- **Experiment Tracking**: Lightweight experiment logging with SQLite backend
- **Beautiful Visualizations**: Model performance charts and experiment dashboards

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ml-experiment-toolkit

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### Data Profiling

```python
import pandas as pd
import mlexp

df = pd.read_csv('your_data.csv')
profile = mlexp.profile(df)  # One-line data profiling with visualizations
```

### Model Comparison

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import mlexp

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = [
    RandomForestClassifier(),
    GradientBoostingClassifier()
]

comparison = mlexp.compare_models(X_train, y_train, X_test, y_test, models=models)
mlexp.plot_model_comparison(comparison)  # Visualize results
```

### Experiment Tracking

```python
import mlexp

mlexp.track_experiment(
    name="random_forest_v1",
    metrics={"accuracy": 0.95, "f1": 0.93},
    params={"n_estimators": 100, "max_depth": 10},
    tags=["classification", "baseline"]
)

# Query experiments
experiments = mlexp.get_experiments(tags=["classification"])
print(experiments)
```

## Documentation

### Data Profiling

The `profile()` function provides comprehensive data analysis:

```python
profile = mlexp.profile(
    df,
    show_plot=True,      # Display visualizations
    figsize=(12, 8)      # Figure size
)
```

**Features:**
- Missing values analysis
- Duplicate row detection
- Statistical summaries for numeric columns
- Categorical column analysis
- Data quality issue detection
- Automatic visualizations

**Quality Issues Detected:**
- High missing value percentages (>50%)
- Duplicate rows
- Low variance numeric columns
- High cardinality categorical columns

### Model Comparison

Compare multiple models with automated metric calculation:

```python
comparison = mlexp.compare_models(
    X_train, y_train,
    X_test, y_test,
    models=[model1, model2, model3],
    model_names=["Model A", "Model B", "Model C"],  # Optional
    task_type="classification",  # Auto-detected if None
    metrics=["accuracy", "f1_score", "roc_auc"],  # Optional
    verbose=True
)

# Get best model
best = comparison.get_best_model(metric="accuracy")
print(f"Best model: {best['model_name']} with accuracy {best['score']:.4f}")

# Visualize
mlexp.plot_model_comparison(comparison)
mlexp.plot_metric_comparison(comparison, metric="accuracy")
```

**Supported Metrics:**
- **Classification**: accuracy, precision, recall, f1_score, roc_auc
- **Regression**: mse, mae, rmse, r2_score

### Experiment Tracking

Lightweight experiment tracking with SQLite backend:

```python
# Track an experiment
exp_id = mlexp.track_experiment(
    name="experiment_1",
    metrics={"accuracy": 0.95, "f1": 0.93},
    params={"n_estimators": 100, "max_depth": 10},
    tags=["classification", "baseline"],
    notes="Initial baseline model",
    artifact_path="models/model.pkl"  # Optional
)

# Query experiments
experiments = mlexp.get_experiments(
    name="experiment",  # Filter by name (partial match)
    tags=["classification"],  # Filter by tags
    limit=10  # Limit results
)

# Get best experiment
tracker = mlexp.ExperimentTracker()
best = tracker.get_best_experiment(
    metric="accuracy",
    higher_is_better=True
)

# Visualize experiment history
mlexp.plot_experiment_history(metric="accuracy")
mlexp.plot_experiment_comparison(metric="accuracy")
```

### Advanced Usage

#### Custom Tracker

```python
from mlexp.models.tracker import ExperimentTracker

# Use custom database path
tracker = ExperimentTracker(db_path="my_experiments.db")
tracker.track(name="exp1", metrics={"accuracy": 0.95})
```

#### Data Validation

```python
from mlexp.data.validator import validate_data, check_outliers

# Run validation checks
results = validate_data(
    df,
    checks=["no_missing", "no_duplicates"],
    custom_checks=[lambda df: len(df) > 100]  # Custom check
)

# Detect outliers
outlier_mask = check_outliers(df, method="iqr", threshold=3.0)
```

## API Reference

### Main Functions

#### `mlexp.profile(df, show_plot=True, figsize=(12, 8))`
Generate comprehensive data profile with quality checks and visualizations.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `show_plot` (bool): Whether to display visualizations
- `figsize` (tuple): Figure size for plots

**Returns:** `DataProfile` object

#### `mlexp.compare_models(X_train, y_train, X_test, y_test, models, ...)`
Compare multiple models side-by-side.

**Parameters:**
- `X_train`, `y_train`: Training data
- `X_test`, `y_test`: Test data
- `models` (list): List of sklearn estimators
- `model_names` (list, optional): Names for models
- `task_type` (str, optional): "classification" or "regression"
- `metrics` (list, optional): Custom metrics to compute
- `verbose` (bool): Whether to print results

**Returns:** `ModelComparison` object

#### `mlexp.track_experiment(name, metrics=None, params=None, ...)`
Track an experiment with metrics and parameters.

**Parameters:**
- `name` (str): Experiment name
- `metrics` (dict, optional): Dictionary of metrics
- `params` (dict, optional): Dictionary of parameters
- `tags` (list, optional): List of tags
- `notes` (str, optional): Additional notes
- `artifact_path` (str, optional): Path to saved artifacts

**Returns:** Experiment ID (int)

#### `mlexp.get_experiments(name=None, tags=None, limit=None)`
Query tracked experiments.

**Returns:** pd.DataFrame with experiment results

### Visualization Functions

#### `mlexp.plot_model_comparison(comparison, metrics=None, figsize=(12, 6))`
Create visualizations for model comparison results.

#### `mlexp.plot_metric_comparison(comparison, metric, figsize=(8, 6))`
Create focused comparison plot for a single metric.

#### `mlexp.plot_experiment_history(metric, name=None, tags=None, figsize=(12, 6))`
Plot experiment history for a metric over time.

#### `mlexp.plot_experiment_comparison(metric, name=None, tags=None, figsize=(10, 6))`
Create bar chart comparing experiments by a metric.

## Examples

Comprehensive examples are available in the `examples/` directory:

- **quickstart.py**: Basic usage demonstration
- **classification_example.py**: Complete classification workflow
- **regression_example.py**: Complete regression workflow

Run examples:

```bash
python examples/quickstart.py
python examples/classification_example.py
python examples/regression_example.py
```

## Key Differentiators

1. **One-line APIs**: Common workflows in single function calls
2. **Opinionated defaults**: Sensible choices so users don't have to configure everything
3. **Lightweight**: No external dependencies beyond standard ML stack
4. **Beautiful defaults**: Visualizations that look good out of the box
5. **Local-first**: Everything runs locally, no cloud dependencies

## Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow PEP 8** style guidelines
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black mlexp/

# Lint code
flake8 mlexp/
```

### Code Style

- Use Black for code formatting (line length: 100)
- Follow PEP 8 conventions
- Add type hints where appropriate
- Write docstrings for all public functions

## License

MIT License - see LICENSE file for details

## Roadmap

- [ ] Support for time series tasks
- [ ] Integration with popular ML frameworks (XGBoost, LightGBM)
- [ ] Export experiment results to various formats (CSV, JSON, HTML)
- [ ] Interactive dashboards with Plotly
- [ ] Model explainability integration (SHAP, LIME)
- [ ] Hyperparameter optimization helpers

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## Acknowledgments

Built with love for the data science community. Inspired by the need for lightweight, opinionated ML experimentation tools.
