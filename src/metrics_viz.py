"""
Metrics history visualiser.

Reads ``runs/<name>/history.csv`` and writes ``runs/<name>/metrics_history.png``
— a multi-panel chart showing each tracked metric across all historical runs.

Each run-name has its own metric schema, so the chart layout is derived
automatically from the columns present in the CSV.

Public API
----------
plot_history(name, base_dir, save, show)  → Path | None
  Renders the history chart for a named run family.

plot_all(base_dir, save, show)
  Renders charts for every run family that has a history.csv.

Schema per run-name
-------------------
"pipeline"
  best_preproc_r2, best_reg_r2, best_backbone_r2, best_tuned_r2,
  n_samples, n_features, n_trials

"bayes"
  best_cv_r2 (per model columns), n_trials, feature_matrix_rows,
  feature_matrix_cols, scope

"demo"
  c1_test_r2_avg, c1_holdingtemp_r2, c1_holdingtime_r2,
  c1_cv_r2_mean, c1c2_test_r2_avg, n_samples, n_features,
  backbone
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RUNS_DIR = Path(__file__).parent.parent / "runs"

# ---------------------------------------------------------------------------
# Per-name chart layout specs
# Each entry: (column, display_label, y_min, y_max or None)
# Groups of columns become subplot panels.
# ---------------------------------------------------------------------------

_PIPELINE_PANELS = [
    {
        "title": "Best R² by Stage",
        "series": [
            ("best_preproc_r2",  "Preprocessing"),
            ("best_reg_r2",      "Regressor"),
            ("best_backbone_r2", "Backbone"),
            ("best_tuned_r2",    "Bayesian-tuned"),
        ],
        "ylabel": "Mean CV R²",
        "ylim": (0, 1),
    },
    {
        "title": "Dataset & Budget",
        "series": [
            ("n_samples",  "Samples"),
            ("n_features", "Features"),
        ],
        "ylabel": "Count",
        "ylim": None,
        "secondary": {
            "series": [("n_trials", "Optuna trials")],
            "ylabel": "Trials",
        },
    },
]

_BAYES_PANELS = [
    {
        "title": "Tuned CV R² by Model",
        "series": None,          # filled dynamically from columns matching *_tuned_cv_r2
        "ylabel": "CV R²",
        "ylim": (0, 1),
        "col_pattern": "_tuned_cv_r2",
    },
    {
        "title": "Improvement (Δ R²)",
        "series": None,
        "ylabel": "Δ R²",
        "ylim": None,
        "col_pattern": "_delta",
    },
    {
        "title": "Feature Matrix Size",
        "series": [
            ("feature_matrix_rows", "Rows"),
            ("feature_matrix_cols", "Cols"),
        ],
        "ylabel": "Dimension",
        "ylim": None,
    },
]

_DEMO_PANELS = [
    {
        "title": "Test R² — Cycle 1",
        "series": [
            ("c1_test_r2_avg",    "Avg"),
            ("c1_holdingtemp_r2", "HoldingTemp"),
            ("c1_holdingtime_r2", "HoldingTime"),
        ],
        "ylabel": "Test R²",
        "ylim": (0, 1),
    },
    {
        "title": "CV R² — Cycle 1",
        "series": [
            ("c1_cv_r2_mean", "CV mean"),
        ],
        "errbar": "c1_cv_r2_std",
        "ylabel": "CV R²",
        "ylim": (0, 1),
    },
    {
        "title": "Cycle 1+2 Extension",
        "series": [
            ("c1c2_test_r2_avg", "Test R²"),
        ],
        "ylabel": "Test R²",
        "ylim": (0, 1),
    },
    {
        "title": "Dataset Size",
        "series": [
            ("n_samples",  "Samples"),
            ("n_features", "Features"),
        ],
        "ylabel": "Count",
        "ylim": None,
    },
]

_SCHEMA: dict[str, list[dict]] = {
    "pipeline": _PIPELINE_PANELS,
    "bayes":    _BAYES_PANELS,
    "demo":     _DEMO_PANELS,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_history(
    name: str,
    base_dir: Union[str, Path] | None = None,
    save: bool = True,
    show: bool = False,
) -> Path | None:
    """
    Render the run-history chart for *name* and save to
    ``runs/<name>/metrics_history.png``.

    Parameters
    ----------
    name :
        Run-family name: ``"pipeline"``, ``"bayes"``, or ``"demo"``.
    base_dir :
        Override the default ``runs/`` directory.
    save :
        Write the figure to disk.
    show :
        Call ``plt.show()`` (useful in notebooks).

    Returns
    -------
    Path to the saved PNG, or ``None`` if the history CSV is empty/missing.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    base = Path(base_dir).resolve() if base_dir else _RUNS_DIR
    history_csv = base / name / "history.csv"

    if not history_csv.exists() or history_csv.stat().st_size == 0:
        logger.warning("plot_history: no history at %s", history_csv)
        return None

    df = pd.read_csv(history_csv)
    if df.empty:
        logger.warning("plot_history: history CSV is empty: %s", history_csv)
        return None

    # Convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").where(
            pd.to_numeric(df[col], errors="coerce").notna(), df[col]
        )

    # x-axis: run index (integer) for even spacing; label with run_id
    x = np.arange(len(df))
    x_labels = df["run_id"].tolist() if "run_id" in df.columns else [str(i) for i in x]
    # Shorten labels to timestamp part only
    x_labels_short = [str(lbl)[:15] for lbl in x_labels]

    panels = _resolve_panels(name, df)
    if not panels:
        logger.warning("plot_history: no panels resolved for name=%r", name)
        return None

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5 * n_panels, 4.5),
        squeeze=False,
    )
    axes = axes[0]

    colours = plt.cm.tab10.colors

    for ax, panel in zip(axes, panels, strict=False):
        series = panel.get("series") or []
        col_pattern = panel.get("col_pattern")

        # Dynamic series from column pattern (bayes panels)
        if col_pattern and not series:
            series = [
                (col, col.replace(col_pattern, "").upper())
                for col in df.columns
                if col.endswith(col_pattern)
            ]

        for ci, (col, label) in enumerate(series):
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            color = colours[ci % len(colours)]
            ax.plot(x, vals, marker="o", linewidth=1.8, markersize=5,
                    label=label, color=color)

            # Error bars (demo CV panel)
            errbar_col = panel.get("errbar")
            if errbar_col and errbar_col in df.columns:
                errs = pd.to_numeric(df[errbar_col], errors="coerce").fillna(0)
                ax.fill_between(x, vals - errs, vals + errs,
                                alpha=0.15, color=color)

        # Secondary y-axis
        secondary = panel.get("secondary")
        if secondary:
            ax2 = ax.twinx()
            for ci, (col, label) in enumerate(secondary["series"]):
                if col not in df.columns:
                    continue
                vals = pd.to_numeric(df[col], errors="coerce")
                ax2.plot(x, vals, marker="s", linewidth=1.4, markersize=4,
                         linestyle="--", label=label,
                         color=colours[(len(series) + ci) % len(colours)])
            ax2.set_ylabel(secondary["ylabel"], fontsize=9)
            ax2.legend(loc="upper right", fontsize=8)

        ax.set_title(panel["title"], fontweight="bold", fontsize=10)
        ax.set_ylabel(panel["ylabel"], fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels_short, rotation=30, ha="right", fontsize=7)

        ylim = panel.get("ylim")
        if ylim:
            ax.set_ylim(*ylim)

        if series:
            ax.legend(fontsize=8, loc="best")

        ax.grid(axis="y", alpha=0.4)
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    # Best-model annotation strip at the bottom of relevant panels
    _annotate_best_model(df, axes, panels)

    title_map = {
        "pipeline": "Pipeline Benchmark — Run History",
        "bayes":    "Bayesian Tuning — Run History",
        "demo":     "Microstructure Demo — Run History",
    }
    fig.suptitle(
        title_map.get(name, f"{name} — Run History"),
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_path: Path | None = None
    if save:
        out_path = base / name / "metrics_history.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        logger.info("plot_history: saved %s", out_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def plot_all(
    base_dir: Union[str, Path] | None = None,
    save: bool = True,
    show: bool = False,
) -> dict[str, Path | None]:
    """
    Render history charts for all run families that have a ``history.csv``.

    Returns a dict mapping name → saved path (or None if skipped).
    """
    base = Path(base_dir).resolve() if base_dir else _RUNS_DIR
    results: dict[str, Path | None] = {}
    for candidate in sorted(base.iterdir()):
        if candidate.is_dir() and (candidate / "history.csv").exists():
            results[candidate.name] = plot_history(
                candidate.name, base_dir=base, save=save, show=show
            )
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_panels(name: str, df: pd.DataFrame) -> list[dict]:
    """Return the panel spec list for *name*, falling back to generic."""
    if name in _SCHEMA:
        return _SCHEMA[name]

    # Generic fallback: one panel with all numeric columns
    numeric_cols = [
        c for c in df.columns
        if c not in ("run_id", "saved_at", "name", "scope", "backbone", "best_model")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        return []
    return [{
        "title": f"{name} metrics",
        "series": [(c, c) for c in numeric_cols[:8]],
        "ylabel": "value",
        "ylim": None,
    }]


def _annotate_best_model(
    df: pd.DataFrame,
    axes,
    panels: list[dict],
) -> None:
    """Add a text strip along the x-axis showing the best model per run."""
    model_col = next(
        (c for c in ("best_model", "c1_best_model", "best_tuned_model") if c in df.columns),
        None,
    )
    if model_col is None:
        return

    x = np.arange(len(df))
    ax = axes[0]
    for xi, val in zip(x, df[model_col], strict=False):
        if pd.notna(val) and str(val).strip():
            ax.text(
                xi, ax.get_ylim()[0],
                str(val),
                ha="center", va="top",
                fontsize=6.5, color="#555555",
                rotation=0,
            )
