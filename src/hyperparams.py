"""
Hyperparameter store — canonical persistence layer for Optuna tuning results.

Canonical path: ``runs/hyperparams.json``

Schema (one top-level key per scope)::

    {
      "dp_steel": {               # tuning scope / dataset filter
        "saved_at":    "2026-04-14 11:23:01",
        "git_commit":  "8273fd7",
        "n_trials":    100,
        "cv_protocol": "RepeatedKFold(n_splits=5, n_repeats=5)",
        "feature_matrix_shape": [54, 38],
        "best_model":  "RF",
        "models": {
          "RF": {
            "best_cv_r2": 0.8312,
            "tuned_cv_r2": 0.8401,   # re-evaluated with 5×10
            "delta":       0.0089,
            "params": {
              "n_estimators": 312,
              "max_depth": 11,
              ...
            }
          },
          ...
        }
      },
      "all_alloys": { ... }
    }

Public API
----------
save(scope, models_dict, ...)   — write / merge results for a scope
load(scope)                     — return the models dict for a scope, or {}
best_model(scope)               — return (model_name, params) for top model
has_params(scope, model_name)   — True if tuned params exist
list_scopes()                   — list all saved scopes
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_STORE_PATH = Path(__file__).parent.parent / "runs" / "hyperparams.json"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).parent.parent),
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_store(path: Path) -> Dict[str, Any]:
    if path.exists() and path.stat().st_size > 0:
        with open(path) as f:
            return json.load(f)
    return {}


def _write_store(store: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(store, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def save(
    scope: str,
    models_dict: Dict[str, Dict],
    n_trials: int = 0,
    cv_protocol: str = "",
    feature_matrix_shape: Optional[Tuple[int, int]] = None,
    store_path: Optional[Path] = None,
) -> Path:
    """
    Persist tuning results for *scope* to the hyperparameter store.

    Parameters
    ----------
    scope:
        Identifier for the dataset / constraint, e.g. ``"dp_steel"`` or
        ``"all_alloys"``. Used as the top-level key in the JSON store.
    models_dict:
        ``{model_name: {"best_cv_r2": float, "tuned_cv_r2": float,
                        "delta": float, "params": dict}}``
        At minimum ``params`` must be present; all other keys are optional.
    n_trials:
        Number of Optuna trials that were run (stored for provenance).
    cv_protocol:
        Human-readable description of the CV used, e.g.
        ``"RepeatedKFold(n_splits=5, n_repeats=5)"``.
    feature_matrix_shape:
        ``(n_rows, n_cols)`` of the feature matrix used during tuning.
    store_path:
        Override the canonical path (useful in tests).

    Returns
    -------
    Path
        The path that was written.
    """
    path  = Path(store_path) if store_path else _STORE_PATH
    store = _load_store(path)

    best = max(models_dict, key=lambda k: models_dict[k].get("tuned_cv_r2",
                                          models_dict[k].get("best_cv_r2", 0)))
    store[scope] = {
        "saved_at":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit":           _git_commit(),
        "n_trials":             n_trials,
        "cv_protocol":          cv_protocol,
        "feature_matrix_shape": list(feature_matrix_shape) if feature_matrix_shape else None,
        "best_model":           best,
        "models":               models_dict,
    }

    _write_store(store, path)
    return path


def load(scope: str, store_path: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Return the ``models`` dict for *scope*, or an empty dict if not found.

    Example return value::

        {
          "RF":  {"best_cv_r2": 0.80, "tuned_cv_r2": 0.83, "delta": 0.03,
                  "params": {"n_estimators": 312, "max_depth": 11, ...}},
          "GBR": { ... },
        }
    """
    path  = Path(store_path) if store_path else _STORE_PATH
    store = _load_store(path)
    return store.get(scope, {}).get("models", {})


def best_model(
    scope: str,
    store_path: Optional[Path] = None,
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Return ``(model_name, params)`` for the best tuned model in *scope*.

    Returns ``(None, None)`` if the scope does not exist or has no models.
    """
    path  = Path(store_path) if store_path else _STORE_PATH
    store = _load_store(path)
    entry = store.get(scope)
    if not entry:
        return None, None
    name   = entry.get("best_model")
    models = entry.get("models", {})
    if not name or name not in models:
        return None, None
    return name, models[name].get("params")


def has_params(
    scope: str,
    model_name: str,
    store_path: Optional[Path] = None,
) -> bool:
    """Return True if tuned params exist for *model_name* in *scope*."""
    models = load(scope, store_path)
    return model_name in models and "params" in models[model_name]


def list_scopes(store_path: Optional[Path] = None) -> list:
    """Return all scope names currently in the store."""
    path  = Path(store_path) if store_path else _STORE_PATH
    store = _load_store(path)
    return list(store.keys())


def summary(store_path: Optional[Path] = None) -> str:
    """Return a human-readable summary of the store contents."""
    path  = Path(store_path) if store_path else _STORE_PATH
    store = _load_store(path)
    if not store:
        return "Hyperparameter store is empty."
    lines = []
    for scope, entry in store.items():
        lines.append(f"Scope: {scope}")
        lines.append(f"  saved_at:   {entry.get('saved_at', '?')}")
        lines.append(f"  git_commit: {entry.get('git_commit', '?')}")
        lines.append(f"  n_trials:   {entry.get('n_trials', '?')}")
        lines.append(f"  cv:         {entry.get('cv_protocol', '?')}")
        lines.append(f"  best_model: {entry.get('best_model', '?')}")
        for name, m in entry.get("models", {}).items():
            r2  = m.get("tuned_cv_r2", m.get("best_cv_r2", "?"))
            d   = m.get("delta", "?")
            marker = " ◀" if name == entry.get("best_model") else ""
            lines.append(f"    {name:<14} tuned_cv_r2={r2}  delta={d}{marker}")
    return "\n".join(lines)
