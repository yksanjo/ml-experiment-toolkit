"""Lightweight experiment tracking with SQLite backend."""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from pathlib import Path


class ExperimentTracker:
    """Lightweight experiment tracker using SQLite."""

    def __init__(self, db_path: str = "experiments.db"):
        """
        Initialize experiment tracker.

        Parameters
        ----------
        db_path : str, default='experiments.db'
            Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metrics TEXT,
                params TEXT,
                tags TEXT,
                notes TEXT,
                artifact_path TEXT
            )
        """)

        conn.commit()
        conn.close()

    def track(
        self,
        name: str,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        artifact_path: Optional[str] = None,
    ) -> int:
        """
        Track an experiment.

        Parameters
        ----------
        name : str
            Experiment name
        metrics : dict, optional
            Dictionary of metrics (e.g., {'accuracy': 0.95, 'f1': 0.93})
        params : dict, optional
            Dictionary of parameters (e.g., {'n_estimators': 100})
        tags : list of str, optional
            List of tags for filtering
        notes : str, optional
            Additional notes about the experiment
        artifact_path : str, optional
            Path to saved model or other artifacts

        Returns
        -------
        int
            Experiment ID

        Examples
        --------
        >>> tracker = ExperimentTracker()
        >>> exp_id = tracker.track(
        ...     name="random_forest_v1",
        ...     metrics={"accuracy": 0.95, "f1": 0.93},
        ...     params={"n_estimators": 100, "max_depth": 10}
        ... )
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        metrics_json = json.dumps(metrics) if metrics else None
        params_json = json.dumps(params) if params else None
        tags_json = json.dumps(tags) if tags else None

        cursor.execute("""
            INSERT INTO experiments (name, timestamp, metrics, params, tags, notes, artifact_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, timestamp, metrics_json, params_json, tags_json, notes, artifact_path))

        exp_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return exp_id

    def get_experiments(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query experiments.

        Parameters
        ----------
        name : str, optional
            Filter by experiment name (partial match)
        tags : list of str, optional
            Filter by tags (experiment must have all tags)
        limit : int, optional
            Maximum number of results to return

        Returns
        -------
        pd.DataFrame
            DataFrame with experiment results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")

        if tags:
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        conn.close()

        if len(rows) == 0:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(rows, columns=columns)

        # Parse JSON columns
        for col in ["metrics", "params", "tags"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if x else None)

        return df

    def get_best_experiment(
        self,
        metric: str,
        higher_is_better: bool = True,
        name: Optional[str] = None,
    ) -> Dict:
        """
        Get the best experiment based on a metric.

        Parameters
        ----------
        metric : str
            Metric name to optimize
        higher_is_better : bool, default=True
            Whether higher values are better
        name : str, optional
            Filter by experiment name

        Returns
        -------
        dict
            Best experiment details
        """
        df = self.get_experiments(name=name)

        if len(df) == 0:
            raise ValueError("No experiments found")

        # Extract metric values
        metric_values = []
        for idx, row in df.iterrows():
            if row["metrics"] and metric in row["metrics"]:
                metric_values.append((idx, row["metrics"][metric]))
            else:
                metric_values.append((idx, None))

        # Filter out None values
        valid_values = [(idx, val) for idx, val in metric_values if val is not None]

        if len(valid_values) == 0:
            raise ValueError(f"Metric '{metric}' not found in any experiments")

        # Find best
        if higher_is_better:
            best_idx, best_val = max(valid_values, key=lambda x: x[1])
        else:
            best_idx, best_val = min(valid_values, key=lambda x: x[1])

        best_row = df.loc[best_idx]
        return {
            "id": int(best_row["id"]),
            "name": best_row["name"],
            "timestamp": best_row["timestamp"],
            "metrics": best_row["metrics"],
            "params": best_row["params"],
            "metric": metric,
            "metric_value": best_val,
        }

    def delete_experiment(self, exp_id: int):
        """Delete an experiment by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
        conn.commit()
        conn.close()

    def clear_all(self):
        """Clear all experiments (use with caution!)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiments")
        conn.commit()
        conn.close()


# Global tracker instance
_default_tracker = None


def _get_default_tracker() -> ExperimentTracker:
    """Get or create the default tracker instance."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = ExperimentTracker()
    return _default_tracker


def track_experiment(
    name: str,
    metrics: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    artifact_path: Optional[str] = None,
    tracker: Optional[ExperimentTracker] = None,
) -> int:
    """
    Track an experiment (convenience function).

    Parameters
    ----------
    name : str
        Experiment name
    metrics : dict, optional
        Dictionary of metrics
    params : dict, optional
        Dictionary of parameters
    tags : list of str, optional
        List of tags
    notes : str, optional
        Additional notes
    artifact_path : str, optional
        Path to saved artifacts
    tracker : ExperimentTracker, optional
        Custom tracker instance. Uses default if None.

    Returns
    -------
    int
        Experiment ID

    Examples
    --------
    >>> import mlexp
    >>> mlexp.track_experiment(
    ...     name="random_forest_v1",
    ...     metrics={"accuracy": 0.95},
    ...     params={"n_estimators": 100}
    ... )
    """
    if tracker is None:
        tracker = _get_default_tracker()
    return tracker.track(name, metrics, params, tags, notes, artifact_path)


def get_experiments(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: Optional[int] = None,
    tracker: Optional[ExperimentTracker] = None,
) -> pd.DataFrame:
    """
    Get experiments (convenience function).

    Parameters
    ----------
    name : str, optional
        Filter by experiment name
    tags : list of str, optional
        Filter by tags
    limit : int, optional
        Maximum number of results
    tracker : ExperimentTracker, optional
        Custom tracker instance. Uses default if None.

    Returns
    -------
    pd.DataFrame
        DataFrame with experiment results
    """
    if tracker is None:
        tracker = _get_default_tracker()
    return tracker.get_experiments(name, tags, limit)

