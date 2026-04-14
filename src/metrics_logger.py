"""
Persistent metrics logger for tracking model performance across notebook runs.

Appends one row per run to ``runs/metrics_log.csv``. Each row captures:
  - run metadata  : timestamp, git commit, notebook name
  - dataset info  : n_samples, n_features, backbone used
  - cycle 1 model : best_model, test_r2_avg, per-target R²/MAE/RMSE, CV R²
  - cycle 1+2 ext : best_model, test_r2_avg (when available)
  - free-form tags: arbitrary key=value annotations (e.g. "smogn=True")

Usage (from a notebook cell)::

    from src.metrics_logger import RunMetrics, log_run

    m = RunMetrics(notebook="microstructure_demo")
    m.set_dataset(n_samples=82, n_features=45, backbone="resnet50")
    m.set_c1(best_model="ABR", test_r2_avg=0.885,
             per_target={"holdingtemp": (0.91, 12.4, 18.1),
                         "holdingtime": (0.86, 5.2, 8.0)},
             cv_r2_mean=0.801, cv_r2_std=0.063)
    m.set_c1c2(best_model="RF", test_r2_avg=0.742)
    m.add_tag("smogn=True", "pruned=True")
    log_run(m)           # writes / appends to runs/metrics_log.csv
    m.print_summary()
"""

from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# (r2, mae, rmse)
TargetMetrics = Tuple[float, float, float]

_LOG_PATH = Path(__file__).parent.parent / "runs" / "metrics_log.csv"

_COLUMNS = [
    # ── metadata ──────────────────────────────────────────────────────────────
    "timestamp",
    "git_commit",
    "notebook",
    # ── dataset ───────────────────────────────────────────────────────────────
    "n_samples",
    "n_features",
    "backbone",
    # ── feature stream dimensions ─────────────────────────────────────────────
    "n_image_features",
    "n_morph_features",
    "n_tabular_features",
    "morph_rows_matched",
    # ── cycle 1 ───────────────────────────────────────────────────────────────
    "c1_best_model",
    "c1_test_r2_avg",
    "c1_test_mae_avg",
    "c1_test_rmse_avg",
    "c1_holdingtemp_r2",
    "c1_holdingtemp_mae",
    "c1_holdingtemp_rmse",
    "c1_holdingtime_r2",
    "c1_holdingtime_mae",
    "c1_holdingtime_rmse",
    "c1_cv_r2_mean",
    "c1_cv_r2_std",
    # ── cycle 1+2 extension ───────────────────────────────────────────────────
    "c1c2_best_model",
    "c1c2_test_r2_avg",
    "c1c2_test_mae_avg",
    "c1c2_test_rmse_avg",
    # ── tags ──────────────────────────────────────────────────────────────────
    "tags",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).parent.parent),
        ).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class RunMetrics:
    notebook: str = "microstructure_demo"

    # dataset
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    backbone: Optional[str] = None

    # feature stream dimensions
    n_image_features: Optional[int] = None
    n_morph_features: Optional[int] = None
    n_tabular_features: Optional[int] = None
    morph_rows_matched: Optional[int] = None

    # cycle 1
    c1_best_model: Optional[str] = None
    c1_test_r2_avg: Optional[float] = None
    c1_test_mae_avg: Optional[float] = None
    c1_test_rmse_avg: Optional[float] = None
    c1_per_target: Dict[str, TargetMetrics] = field(default_factory=dict)
    c1_cv_r2_mean: Optional[float] = None
    c1_cv_r2_std: Optional[float] = None

    # cycle 1+2
    c1c2_best_model: Optional[str] = None
    c1c2_test_r2_avg: Optional[float] = None
    c1c2_test_mae_avg: Optional[float] = None
    c1c2_test_rmse_avg: Optional[float] = None

    # tags
    _tags: list = field(default_factory=list)

    # ── setters (fluent) ──────────────────────────────────────────────────────

    def set_dataset(
        self,
        n_samples: int,
        n_features: int,
        backbone: str = "none",
        n_image_features: Optional[int] = None,
        n_morph_features: Optional[int] = None,
        n_tabular_features: Optional[int] = None,
        morph_rows_matched: Optional[int] = None,
    ) -> "RunMetrics":
        self.n_samples = n_samples
        self.n_features = n_features
        self.backbone = backbone
        self.n_image_features = n_image_features
        self.n_morph_features = n_morph_features
        self.n_tabular_features = n_tabular_features
        self.morph_rows_matched = morph_rows_matched
        return self

    def set_c1(
        self,
        best_model: str,
        test_r2_avg: float,
        test_mae_avg: Optional[float] = None,
        test_rmse_avg: Optional[float] = None,
        per_target: Optional[Dict[str, TargetMetrics]] = None,
        cv_r2_mean: Optional[float] = None,
        cv_r2_std: Optional[float] = None,
    ) -> "RunMetrics":
        self.c1_best_model = best_model
        self.c1_test_r2_avg = test_r2_avg
        self.c1_test_mae_avg = test_mae_avg
        self.c1_test_rmse_avg = test_rmse_avg
        self.c1_per_target = per_target or {}
        self.c1_cv_r2_mean = cv_r2_mean
        self.c1_cv_r2_std = cv_r2_std
        return self

    def set_c1c2(
        self,
        best_model: str,
        test_r2_avg: float,
        test_mae_avg: Optional[float] = None,
        test_rmse_avg: Optional[float] = None,
    ) -> "RunMetrics":
        self.c1c2_best_model = best_model
        self.c1c2_test_r2_avg = test_r2_avg
        self.c1c2_test_mae_avg = test_mae_avg
        self.c1c2_test_rmse_avg = test_rmse_avg
        return self

    def add_tag(self, *tags: str) -> "RunMetrics":
        self._tags.extend(tags)
        return self

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_row(self) -> Dict[str, object]:
        # Unpack per-target dict (keys: holdingtemp, holdingtime)
        def _pt(key: str, idx: int):
            t = self.c1_per_target.get(key)
            return t[idx] if t is not None else None

        return {
            "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit":         _git_commit(),
            "notebook":           self.notebook,
            "n_samples":          self.n_samples,
            "n_features":         self.n_features,
            "backbone":           self.backbone,
            "n_image_features":   self.n_image_features,
            "n_morph_features":   self.n_morph_features,
            "n_tabular_features": self.n_tabular_features,
            "morph_rows_matched": self.morph_rows_matched,
            "c1_best_model":      self.c1_best_model,
            "c1_test_r2_avg":     _fmt(self.c1_test_r2_avg),
            "c1_test_mae_avg":    _fmt(self.c1_test_mae_avg),
            "c1_test_rmse_avg":   _fmt(self.c1_test_rmse_avg),
            "c1_holdingtemp_r2":  _fmt(_pt("holdingtemp", 0)),
            "c1_holdingtemp_mae": _fmt(_pt("holdingtemp", 1)),
            "c1_holdingtemp_rmse":_fmt(_pt("holdingtemp", 2)),
            "c1_holdingtime_r2":  _fmt(_pt("holdingtime", 0)),
            "c1_holdingtime_mae": _fmt(_pt("holdingtime", 1)),
            "c1_holdingtime_rmse":_fmt(_pt("holdingtime", 2)),
            "c1_cv_r2_mean":      _fmt(self.c1_cv_r2_mean),
            "c1_cv_r2_std":       _fmt(self.c1_cv_r2_std),
            "c1c2_best_model":    self.c1c2_best_model,
            "c1c2_test_r2_avg":   _fmt(self.c1c2_test_r2_avg),
            "c1c2_test_mae_avg":  _fmt(self.c1c2_test_mae_avg),
            "c1c2_test_rmse_avg": _fmt(self.c1c2_test_rmse_avg),
            "tags":               "; ".join(self._tags),
        }

    def print_summary(self) -> None:
        row = self.to_row()
        print("=" * 60)
        print(f"RUN LOGGED  [{row['timestamp']}]  commit={row['git_commit']}")
        print("=" * 60)
        img_dim  = row.get('n_image_features') or ''
        morph_dim = row.get('n_morph_features') or ''
        tab_dim  = row.get('n_tabular_features') or ''
        morph_match = row.get('morph_rows_matched') or ''
        print(f"  dataset   : {row['n_samples']} samples, {row['n_features']} features, backbone={row['backbone']}")
        if any([img_dim, morph_dim, tab_dim]):
            print(f"  streams   : image={img_dim}  morph={morph_dim} (matched={morph_match})  tabular={tab_dim}")
        print(f"  C1 model  : {row['c1_best_model']}  test R²={row['c1_test_r2_avg']}  "
              f"MAE={row['c1_test_mae_avg']}  RMSE={row['c1_test_rmse_avg']}")
        print(f"             holdingtemp  R²={row['c1_holdingtemp_r2']}  "
              f"MAE={row['c1_holdingtemp_mae']}  RMSE={row['c1_holdingtemp_rmse']}")
        print(f"             holdingtime  R²={row['c1_holdingtime_r2']}  "
              f"MAE={row['c1_holdingtime_mae']}  RMSE={row['c1_holdingtime_rmse']}")
        print(f"             CV R²={row['c1_cv_r2_mean']} ± {row['c1_cv_r2_std']}")
        if row['c1c2_best_model']:
            print(f"  C1+C2     : {row['c1c2_best_model']}  test R²={row['c1c2_test_r2_avg']}  "
                  f"MAE={row['c1c2_test_mae_avg']}  RMSE={row['c1c2_test_rmse_avg']}")
        if row['tags']:
            print(f"  tags      : {row['tags']}")
        print("=" * 60)


def _fmt(v) -> str:
    """Format a float to 4 d.p., or empty string if None."""
    if v is None:
        return ""
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def log_run(metrics: RunMetrics, log_path: Optional[Path] = None) -> Path:
    """
    Append one row to the metrics log CSV and return the path.

    Creates the file with a header row on first call.
    Thread-safe for single-writer use (notebook runs are sequential).
    """
    path = Path(log_path) if log_path else _LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    row = metrics.to_row()
    write_header = not path.exists() or path.stat().st_size == 0

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return path


def load_log(log_path: Optional[Path] = None):
    """Return the metrics log as a pandas DataFrame (requires pandas)."""
    import pandas as pd
    path = Path(log_path) if log_path else _LOG_PATH
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(path, on_bad_lines="warn")
    # Add any columns present in the current schema but missing from the file
    # (happens when loading a log written by an older version of the logger)
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[_COLUMNS]
