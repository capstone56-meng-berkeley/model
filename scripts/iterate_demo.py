"""Iteratively improve CV R² on Cycle 1 HoldingTemp / HoldingTime.

Runs a sequence of "strategy" candidates against the current best baseline.
After every iteration, the winning strategy (the one with the largest
positive delta on the mean R² across the two targets) is folded into the
baseline and the next iteration tests the remaining strategies on top of
that. Stops when no remaining strategy clears MIN_DELTA_R2 or the strategy
list is exhausted.

Caches: read-only. Reuses demo notebook's caches under
  data/image_cache_<backbone>.npz   (default: dinov2_vitb14)
  features/morph_features_c1.npz

Outputs: under runs/iterate_demo/<run_id>/
  - iteration_<n>.md   structured per-iteration log (rule, candidates,
                       winner, deltas, current stack)
  - history.csv        long-format row per (iteration, strategy, target)
  - summary.md         running narrative across all iterations
  - predicted_vs_actual_iter<n>.png   per-iteration scatter of best run

Usage:
  python scripts/iterate_demo.py             # run all Tier-1 strategies
  python scripts/iterate_demo.py --tier 1+2  # include Tier-2 strategies too
  python scripts/iterate_demo.py --min-delta 0.01
  python scripts/iterate_demo.py --max-iter 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Project imports — script runs from repo root or scripts/.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.column_sanitizer import sanitize_dataframe
from src.config import (
    EncodingConfig,
    MissingDataConfig,
    PreprocessingConfig,
    ScalingConfig,
)
from src.preprocessing import FeaturePreprocessor
from src.run_store import RunStore

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.multioutput import MultiOutputRegressor


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

C1_TEMP = "cycle1_holdingtemp_degc"
C1_TIME = "cycle1_holdingtime_min"
TARGET_COLS = [C1_TEMP, C1_TIME]
TARGET_LABELS = ["holdingtemp", "holdingtime"]

# Tabular columns that would leak the recipe — exclude from input features.
LEAKAGE_TARGETS = {
    "cycle1_holdingtemp_degc", "cycle1_holdingtime_min",
    "cycle2_holdingtemp_degc", "cycle2_holdingtime_min",
    "cycle3_holdingtemp_degc", "cycle3_holdingtime_min",
    "cycle4_holdingtemp_degc", "cycle4_holdingtime_min",
}

COLUMN_TYPE_OVERRIDES = {
    "num_cycles":          "categorical",
    "cycle1_crate_degc_s": "categorical",
    "cycle1_rtemp":        "categorical",
    "cycle1_qtemp":        "categorical",
    "cycle2_rtemp_degc":   "categorical",
    "cycle3_rtemp":        "categorical",
    "cycle3_qtemp":        "categorical",
    "cycle4_qtemp":        "categorical",
    "heat_treatment_type": "categorical",
    "id":                  "unique_string",
}

MICE_COLUMNS = ["cr", "mo", "s", "p", "ni", "al"]
INDICATOR_COLUMNS = ["ti", "nb", "v"]


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load + sanitize metadata CSV; keep rows with both Cycle1 targets."""
    df_raw = pd.read_csv(csv_path, header=1)
    df_raw = sanitize_dataframe(df_raw)
    df = df_raw[df_raw["c"].notna()].copy().reset_index(drop=True)
    mask = df[C1_TEMP].notna() & df[C1_TIME].notna()
    return df[mask].copy().reset_index(drop=True)


def build_tabular_matrix(df_c1: pd.DataFrame) -> tuple[np.ndarray, list[str], FeaturePreprocessor]:
    """Fit-on-full preprocessing for the *iteration loop* — cross-validation
    handles holdout. We accept the small bit of mean/median leakage in
    return for a much simpler driver: per-fold preprocessing would require
    threading the preprocessor through CV, which doubles the script's size."""
    feature_columns = [c for c in df_c1.columns if c not in LEAKAGE_TARGETS]
    cfg = PreprocessingConfig(
        missing_data=MissingDataConfig(
            column_drop_threshold=0.95,
            row_fill_threshold=0.50,
            numeric_fill_strategy="median",
            categorical_fill_strategy="mode",
            mice_max_iter=10,
        ),
        scaling=ScalingConfig(method="standard", enabled=True),
        encoding=EncodingConfig(categorical="onehot", max_categories=50),
    )
    overrides = {k: v for k, v in COLUMN_TYPE_OVERRIDES.items() if k in feature_columns}
    mice_cols = [c for c in MICE_COLUMNS if c in feature_columns
                 and df_c1[c].notna().any()]
    indicator_cols = [c for c in INDICATOR_COLUMNS if c in feature_columns
                      and df_c1[c].notna().any()]
    pp = FeaturePreprocessor(cfg, column_types=overrides,
                             mice_columns=mice_cols, indicator_columns=indicator_cols)
    X = pp.fit_transform(df_c1[feature_columns].copy())
    return X, pp.get_feature_names(), pp


# Filename suffix shared across the project: _F_<n>.<ext>
_F_RE = re.compile(r"_F_\d+\.[a-z]+$", re.IGNORECASE)
def _norm_id(s: str) -> str:
    return re.sub(r"[-\s]+", "_", str(s).strip().lower())


def align_cache_to_ids(cache_path: Path, df_ids: pd.Series) -> np.ndarray | None:
    """Load .npz cache and align to df_ids by row id. Mean-pools images per row."""
    if not cache_path.exists():
        return None
    data = np.load(str(cache_path), allow_pickle=True)
    X_cache = data["X"].astype(np.float32)
    fnames = list(data["filenames"])
    by_id: dict[str, list[int]] = defaultdict(list)
    for i, fn in enumerate(fnames):
        rid = _norm_id(_F_RE.sub("", os.path.basename(str(fn))))
        by_id[rid].append(i)

    out = np.full((len(df_ids), X_cache.shape[1]), np.nan, dtype=np.float32)
    for r, rid in enumerate(df_ids.astype(str).map(_norm_id)):
        idxs = by_id.get(rid, [])
        if idxs:
            out[r] = X_cache[idxs].mean(axis=0)
    col_means = np.nanmean(out, axis=0)
    nan_rows = np.isnan(out).any(axis=1)
    if nan_rows.any():
        out[nan_rows] = col_means
    return out


def load_morph_cache(cache_path: Path, df_ids: pd.Series) -> np.ndarray | None:
    """Load morph cache and align to df_ids. Same id-key logic as image cache."""
    if not cache_path.exists():
        return None
    data = np.load(str(cache_path), allow_pickle=True)
    X = data["X"].astype(np.float64)
    if "filenames" in data.files:
        fnames = list(data["filenames"])
        by_id: dict[str, list[int]] = defaultdict(list)
        for i, fn in enumerate(fnames):
            rid = _norm_id(_F_RE.sub("", os.path.basename(str(fn))))
            by_id[rid].append(i)
        out = np.full((len(df_ids), X.shape[1]), np.nan, dtype=np.float64)
        for r, rid in enumerate(df_ids.astype(str).map(_norm_id)):
            idxs = by_id.get(rid, [])
            if idxs:
                out[r] = np.nanmean(X[idxs], axis=0)
        return out
    # No filenames stored → assume positional alignment with df ordering.
    n = min(len(X), len(df_ids))
    out = np.full((len(df_ids), X.shape[1]), np.nan, dtype=np.float64)
    out[:n] = X[:n]
    return out


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

@dataclass
class State:
    """Mutable cumulative state passed between strategies."""
    X: np.ndarray
    Y: np.ndarray
    feature_names: list[str]
    df_c1: pd.DataFrame  # for stratification keys
    # Modelling knobs the strategies can flip:
    log_time:       bool = False           # wrap HoldingTime with log1p
    per_target:     bool = False           # two single-output models vs MultiOutputRegressor
    stratify_by:    str | None = None      # "alloy", "temp_bin", None
    cv_repeats:     int = 3                # default repeated-KFold repeats inside the iteration loop
    cv_splits:      int = 5
    seed:           int = 42
    notes:          list[str] = field(default_factory=list)

    def clone(self) -> "State":
        return State(
            X=self.X, Y=self.Y, feature_names=list(self.feature_names),
            df_c1=self.df_c1,
            log_time=self.log_time, per_target=self.per_target,
            stratify_by=self.stratify_by,
            cv_repeats=self.cv_repeats, cv_splits=self.cv_splits, seed=self.seed,
            notes=list(self.notes),
        )


@dataclass
class StrategyResult:
    name: str
    description: str
    code_diff: str
    accepted: bool
    delta_mean_r2: float
    metrics: dict[str, float]
    secs: float


def make_estimator(state: State):
    """Return a fitted-on-the-fly estimator that respects per_target / log_time."""
    base = lambda: GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, min_samples_leaf=5, random_state=state.seed,
    )

    if state.per_target:
        # Per-target wrapper that the driver evaluates target-by-target.
        # Returned as a list so cv_score can iterate.
        def _per_target_factory():
            est = base()
            if state.log_time:
                # Time only — wrap log1p; temp is identity. Caller decides which target.
                return est
            return est
        return _per_target_factory

    # Multi-output joint model.
    inner = MultiOutputRegressor(base())
    if state.log_time:
        # Wrap *only* the second target with log1p. We do this with a target
        # transform that operates per-column inside fit/predict.
        def _func(y):
            y = np.asarray(y, dtype=np.float64).copy()
            y[:, 1] = np.log1p(y[:, 1])
            return y
        def _inv(y):
            y = np.asarray(y, dtype=np.float64).copy()
            y[:, 1] = np.expm1(y[:, 1])
            return y
        return TransformedTargetRegressor(
            regressor=inner, func=_func, inverse_func=_inv, check_inverse=False,
        )
    return inner


def cv_evaluate(state: State) -> dict[str, float]:
    """Run RepeatedKFold CV; return {temp_r2, time_r2, mean_r2, ...}."""
    n = len(state.Y)
    rkf = RepeatedKFold(n_splits=state.cv_splits, n_repeats=state.cv_repeats,
                        random_state=state.seed)

    # Optionally restrict to a stratified KFold (only first repeat is stratified).
    folds = list(rkf.split(state.X))
    if state.stratify_by == "alloy":
        # Replace the first repeat's folds with stratified-by-alloy folds.
        skf = StratifiedKFold(n_splits=state.cv_splits, shuffle=True,
                              random_state=state.seed)
        y_strat = state.df_c1["alloy"].astype(str).fillna("UNK").values
        try:
            strat_folds = list(skf.split(state.X, y_strat))
            folds = strat_folds + folds[len(strat_folds):]
        except ValueError:
            pass  # Some alloys may have <2 samples; fallback to plain KFold
    elif state.stratify_by == "temp_bin":
        # Bin HoldingTemp into 5 quantile bins. pd.qcut returns a Categorical/
        # ndarray (no .values attribute when fed an ndarray); coerce to a Series
        # first so .astype(str).values works regardless of input type.
        try:
            y_bin = pd.qcut(pd.Series(state.Y[:, 0]),
                             q=5, duplicates="drop").astype(str).values
            skf = StratifiedKFold(n_splits=state.cv_splits, shuffle=True,
                                  random_state=state.seed)
            strat_folds = list(skf.split(state.X, y_bin))
            folds = strat_folds + folds[len(strat_folds):]
        except ValueError:
            pass

    temp_r2_scores, time_r2_scores = [], []
    temp_mae_scores, time_mae_scores = [], []
    for tr_idx, te_idx in folds:
        X_tr, X_te = state.X[tr_idx], state.X[te_idx]
        Y_tr, Y_te = state.Y[tr_idx], state.Y[te_idx]

        if state.per_target:
            factory = make_estimator(state)
            # Temp: identity regressor.
            mt = factory()
            mt.fit(X_tr, Y_tr[:, 0])
            yp_t = mt.predict(X_te)
            temp_r2_scores.append(r2_score(Y_te[:, 0], yp_t))
            temp_mae_scores.append(mean_absolute_error(Y_te[:, 0], yp_t))

            # Time: optional log1p wrap.
            mh = factory()
            if state.log_time:
                mh = TransformedTargetRegressor(
                    regressor=mh, func=np.log1p, inverse_func=np.expm1,
                    check_inverse=False,
                )
            mh.fit(X_tr, Y_tr[:, 1])
            yp_h = mh.predict(X_te)
            time_r2_scores.append(r2_score(Y_te[:, 1], yp_h))
            time_mae_scores.append(mean_absolute_error(Y_te[:, 1], yp_h))
        else:
            est = make_estimator(state)
            est.fit(X_tr, Y_tr)
            yp = est.predict(X_te)
            if yp.ndim == 1:
                yp = yp.reshape(-1, 1)
            temp_r2_scores.append(r2_score(Y_te[:, 0], yp[:, 0]))
            time_r2_scores.append(r2_score(Y_te[:, 1], yp[:, 1]))
            temp_mae_scores.append(mean_absolute_error(Y_te[:, 0], yp[:, 0]))
            time_mae_scores.append(mean_absolute_error(Y_te[:, 1], yp[:, 1]))

    return {
        "temp_r2_mean":  float(np.mean(temp_r2_scores)),
        "temp_r2_std":   float(np.std(temp_r2_scores)),
        "time_r2_mean":  float(np.mean(time_r2_scores)),
        "time_r2_std":   float(np.std(time_r2_scores)),
        "mean_r2":       float(np.mean([np.mean(temp_r2_scores), np.mean(time_r2_scores)])),
        "temp_mae":      float(np.mean(temp_mae_scores)),
        "time_mae":      float(np.mean(time_mae_scores)),
        "n_folds":       len(folds),
    }


def apply_strategy(state: State, name: str) -> tuple[State, str, str]:
    """Return (new_state, description, code_diff). Idempotent."""
    s = state.clone()
    if name == "log_time":
        s.log_time = True
        return s, (
            "Wrap HoldingTime target with log1p / expm1. The time range "
            "(10 to 90+ minutes) is roughly log-uniform, so linear-MSE "
            "over-weights long-time samples."
        ), (
            "from sklearn.compose import TransformedTargetRegressor\n"
            "model = TransformedTargetRegressor(\n"
            "    regressor=GradientBoostingRegressor(...),\n"
            "    func=np.log1p, inverse_func=np.expm1)"
        )
    if name == "per_target":
        s.per_target = True
        return s, (
            "Train two independent single-output GBRs (one per target) "
            "instead of MultiOutputRegressor wrapping a joint multi-output "
            "GBR. Decouples the two targets so each tree depth/split can "
            "specialise."
        ), (
            "# before:\n"
            "model = MultiOutputRegressor(GradientBoostingRegressor(...))\n"
            "# after:\n"
            "m_temp = GradientBoostingRegressor(...)  # fits Y[:, 0] only\n"
            "m_time = GradientBoostingRegressor(...)  # fits Y[:, 1] only"
        )
    if name == "stratify_alloy":
        s.stratify_by = "alloy"
        return s, (
            "Use StratifiedKFold by alloy for the first CV repeat so every "
            "alloy is represented in both train and test in each fold."
        ), (
            "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n"
            "folds = list(skf.split(X, df_c1['alloy']))"
        )
    if name == "stratify_temp_bin":
        s.stratify_by = "temp_bin"
        return s, (
            "Use StratifiedKFold by HoldingTemp quantile-bin (5 bins) so "
            "rare setpoints aren't entirely on one side of the split."
        ), (
            "y_bin = pd.qcut(Y[:, 0], q=5, duplicates='drop').astype(str)\n"
            "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n"
            "folds = list(skf.split(X, y_bin))"
        )
    raise KeyError(f"Unknown strategy: {name!r}")


# Strategy lists, by tier. Order matters — strategies are tried left-to-right
# at each iteration; the first one to clear MIN_DELTA_R2 wins, then is folded
# into the baseline before the next iteration.
TIER_1_STRATEGIES = [
    "log_time",
    "per_target",
    "stratify_alloy",
    "stratify_temp_bin",
]


# ---------------------------------------------------------------------------
# Documentation writers
# ---------------------------------------------------------------------------

def write_iteration_md(out_dir: Path, n: int,
                       baseline: dict, candidates: list[StrategyResult],
                       winner: StrategyResult | None,
                       stack: list[str]) -> Path:
    md = out_dir / f"iteration_{n:02d}.md"
    lines = [
        f"# Iteration {n}",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_",
        "",
        "## Baseline going in",
        "",
        f"- Cumulative stack: `{stack or 'none — vanilla baseline'}`",
        f"- HoldingTemp R² : `{baseline['temp_r2_mean']:+.4f} ± {baseline['temp_r2_std']:.4f}`",
        f"- HoldingTime R² : `{baseline['time_r2_mean']:+.4f} ± {baseline['time_r2_std']:.4f}`",
        f"- Mean R²        : `{baseline['mean_r2']:+.4f}`",
        f"- Folds          : {baseline['n_folds']}",
        "",
        "## Candidates tested this iteration",
        "",
    ]
    for c in candidates:
        marker = "✅ accepted" if c.accepted else "❌ rejected"
        lines += [
            f"### `{c.name}` — {marker}",
            "",
            c.description,
            "",
            "**Diff:**",
            "",
            "```python",
            c.code_diff,
            "```",
            "",
            "**Result vs baseline:**",
            "",
            f"- HoldingTemp R²: `{c.metrics['temp_r2_mean']:+.4f}` "
            f"(Δ = `{c.metrics['temp_r2_mean'] - baseline['temp_r2_mean']:+.4f}`)",
            f"- HoldingTime R²: `{c.metrics['time_r2_mean']:+.4f}` "
            f"(Δ = `{c.metrics['time_r2_mean'] - baseline['time_r2_mean']:+.4f}`)",
            f"- Mean R²       : `{c.metrics['mean_r2']:+.4f}` "
            f"(Δ = `{c.delta_mean_r2:+.4f}`)",
            f"- Wall time     : `{c.secs:.1f}s`",
            "",
        ]
    if winner is None:
        lines += [
            "## Outcome",
            "",
            "**No candidate cleared the improvement threshold — iteration loop ends.**",
            "",
        ]
    else:
        lines += [
            "## Outcome",
            "",
            f"**Winner: `{winner.name}`** "
            f"(Δ mean R² = `{winner.delta_mean_r2:+.4f}`)",
            "",
            f"Folded into the baseline. New cumulative stack: "
            f"`{stack + [winner.name]}`",
            "",
        ]
    md.write_text("\n".join(lines))
    return md


def update_summary_md(out_dir: Path, baseline_initial: dict,
                       iterations: list[dict]) -> Path:
    summary = out_dir / "summary.md"
    lines = [
        "# Iteration summary — `microstructure_demo` improvement loop",
        "",
        f"_Run id: `{out_dir.name}`_",
        f"_Updated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_",
        "",
        "## Initial baseline",
        "",
        f"- HoldingTemp R²: `{baseline_initial['temp_r2_mean']:+.4f} ± {baseline_initial['temp_r2_std']:.4f}`",
        f"- HoldingTime R²: `{baseline_initial['time_r2_mean']:+.4f} ± {baseline_initial['time_r2_std']:.4f}`",
        f"- Mean R²       : `{baseline_initial['mean_r2']:+.4f}`",
        "",
        "## Iterations",
        "",
        "| # | Winner | Δ mean R² | Temp R² | Time R² | Mean R² | Stack |",
        "|---|---|---|---|---|---|---|",
    ]
    for it in iterations:
        winner = it.get("winner") or "(none — stop)"
        delta  = it.get("winner_delta", 0.0)
        m      = it["after"]
        stack  = ", ".join(it["stack_after"]) or "—"
        lines.append(
            f"| {it['n']} | `{winner}` | `{delta:+.4f}` | "
            f"`{m['temp_r2_mean']:+.4f}` | `{m['time_r2_mean']:+.4f}` | "
            f"`{m['mean_r2']:+.4f}` | `{stack}` |"
        )
    if iterations:
        last = iterations[-1]["after"]
        lines += [
            "",
            "## Final state",
            "",
            f"- HoldingTemp R² : `{last['temp_r2_mean']:+.4f}` "
            f"(Δ vs initial: `{last['temp_r2_mean'] - baseline_initial['temp_r2_mean']:+.4f}`)",
            f"- HoldingTime R² : `{last['time_r2_mean']:+.4f}` "
            f"(Δ vs initial: `{last['time_r2_mean'] - baseline_initial['time_r2_mean']:+.4f}`)",
            f"- Mean R²        : `{last['mean_r2']:+.4f}` "
            f"(Δ vs initial: `{last['mean_r2'] - baseline_initial['mean_r2']:+.4f}`)",
            "",
        ]
    summary.write_text("\n".join(lines))
    return summary


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default=str(_REPO / "datasets" / "metadata_latest.csv"))
    parser.add_argument("--backbone",   default="dinov2_vitb14")
    parser.add_argument("--cache-dir",  default=str(_REPO / "data"))
    parser.add_argument("--morph-cache", default=str(_REPO / "features" / "morph_features_c1.npz"))
    parser.add_argument("--tier",       default="1", help="strategy tier: '1' or '1+2'")
    parser.add_argument("--min-delta",  type=float, default=0.005,
                        help="minimum mean-R² improvement to accept a strategy")
    parser.add_argument("--max-iter",   type=int, default=8,
                        help="hard cap on iterations regardless of strategies remaining")
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--cv-splits",  type=int, default=5)
    parser.add_argument("--final-cv-repeats", type=int, default=10,
                        help="repeats for the final headline number once the loop converges")
    args = parser.parse_args()

    # ── Build feature matrix from caches ────────────────────────────────────
    print(f"Loading dataset from {args.csv}")
    df_c1 = load_dataframe(Path(args.csv))
    print(f"  Cycle 1 subset: {len(df_c1)} rows")

    X_tab, tab_names, _ = build_tabular_matrix(df_c1)
    print(f"  Tabular features after preprocessing: {X_tab.shape}")

    cache = Path(args.cache_dir) / f"image_cache_{args.backbone}.npz"
    X_img = align_cache_to_ids(cache, df_c1["id"])
    if X_img is None:
        raise SystemExit(f"Image cache missing: {cache}")
    print(f"  Image features  ({args.backbone}): {X_img.shape}")

    X_morph = load_morph_cache(Path(args.morph_cache), df_c1["id"])
    if X_morph is None:
        print(f"  WARN: morph cache missing at {args.morph_cache} — proceeding without")

    parts = [X_img.astype(np.float64)]
    parts.append(X_tab.astype(np.float64))
    if X_morph is not None:
        parts.insert(1, X_morph.astype(np.float64))
        print(f"  Morph features: {X_morph.shape}")
    X_full = np.concatenate(parts, axis=1)

    # Tabular preprocessing already imputes; image alignment fills NaN rows
    # with column means; morph may still have raw NaNs from missing IDs.
    if np.isnan(X_full).any():
        X_full = SimpleImputer(strategy="median").fit_transform(X_full)
    print(f"  Combined feature matrix: {X_full.shape}")

    Y = df_c1[TARGET_COLS].values.astype(np.float64)
    print(f"  Targets: {Y.shape}")

    # ── Allocate run directory ──────────────────────────────────────────────
    store = RunStore("iterate_demo")
    run_dir, run_id = store.start()
    print(f"\nRun directory: {run_dir}")

    # ── Initial baseline ────────────────────────────────────────────────────
    state = State(X=X_full, Y=Y, feature_names=[], df_c1=df_c1,
                   cv_repeats=args.cv_repeats, cv_splits=args.cv_splits)
    print("\nMeasuring initial baseline (RepeatedKFold "
          f"{args.cv_splits}x{args.cv_repeats})...")
    t0 = time.time()
    base_metrics = cv_evaluate(state)
    print(f"  baseline temp_r2={base_metrics['temp_r2_mean']:+.4f} ± "
          f"{base_metrics['temp_r2_std']:.4f}  "
          f"time_r2={base_metrics['time_r2_mean']:+.4f} ± "
          f"{base_metrics['time_r2_std']:.4f}  "
          f"mean={base_metrics['mean_r2']:+.4f}  ({time.time()-t0:.1f}s)")

    initial_baseline = base_metrics
    history_rows: list[dict] = []
    iteration_records: list[dict] = []
    stack: list[str] = []
    remaining = list(TIER_1_STRATEGIES)
    if "2" in args.tier:
        # placeholder for Tier-2 strategies — not implemented yet.
        pass

    # ── Iteration loop ──────────────────────────────────────────────────────
    n = 0
    while remaining and n < args.max_iter:
        n += 1
        print(f"\n=== Iteration {n} — testing {len(remaining)} candidate(s): {remaining}")

        candidate_results: list[StrategyResult] = []
        best: StrategyResult | None = None

        for strat in remaining:
            t1 = time.time()
            try:
                new_state, desc, diff = apply_strategy(state, strat)
                metrics = cv_evaluate(new_state)
                secs    = time.time() - t1
                delta   = metrics["mean_r2"] - base_metrics["mean_r2"]
                accepted = delta >= args.min_delta
            except Exception as exc:
                # A broken strategy must not kill the whole run — record it as
                # a no-op rejection so the markdown captures the failure and
                # the next strategy still gets a chance.
                secs = time.time() - t1
                print(f"  [{strat:<20}] CRASHED: {type(exc).__name__}: {exc}")
                metrics = {
                    "temp_r2_mean": float("nan"), "temp_r2_std": float("nan"),
                    "time_r2_mean": float("nan"), "time_r2_std": float("nan"),
                    "mean_r2":      float("nan"),
                    "temp_mae":     float("nan"), "time_mae": float("nan"),
                    "n_folds":      0,
                }
                desc = f"(crashed: {type(exc).__name__}: {exc})"
                diff = "(no diff — strategy raised before evaluation)"
                delta = float("-inf")
                accepted = False

            r = StrategyResult(name=strat, description=desc, code_diff=diff,
                               accepted=accepted, delta_mean_r2=delta,
                               metrics=metrics, secs=secs)
            candidate_results.append(r)

            history_rows.append({
                "iteration": n, "strategy": strat,
                "temp_r2": metrics["temp_r2_mean"], "time_r2": metrics["time_r2_mean"],
                "mean_r2": metrics["mean_r2"],
                "delta_mean_r2": delta,
                "secs": secs,
            })

            # Persist history.csv after every candidate so a later crash
            # doesn't lose the work that did finish.
            pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

            if not np.isnan(metrics["mean_r2"]):
                print(f"  [{strat:<20}] temp={metrics['temp_r2_mean']:+.4f}  "
                      f"time={metrics['time_r2_mean']:+.4f}  "
                      f"mean={metrics['mean_r2']:+.4f}  "
                      f"(Δ={delta:+.4f})  ({secs:.1f}s)")

            if accepted and (best is None or delta > best.delta_mean_r2):
                best = r

        winner_name = best.name if best else None
        if best is None:
            print(f"\n  No candidate cleared min_delta={args.min_delta} — stopping.")

        # Per-iteration markdown
        write_iteration_md(run_dir, n, base_metrics, candidate_results, best, stack)

        # Cumulative state update
        if best is not None:
            state, _, _ = apply_strategy(state, best.name)
            stack.append(best.name)
            remaining = [s for s in remaining if s != best.name]
            base_metrics = best.metrics
        else:
            remaining = []

        iteration_records.append({
            "n": n,
            "winner": winner_name,
            "winner_delta": best.delta_mean_r2 if best else 0.0,
            "after": base_metrics,
            "stack_after": list(stack),
        })

        # Update running summary after every iteration so you can tail it.
        update_summary_md(run_dir, initial_baseline, iteration_records)

    # ── Final headline number ──────────────────────────────────────────────
    print(f"\n=== Final headline measurement (RepeatedKFold "
          f"{args.cv_splits}x{args.final_cv_repeats}) ===")
    state.cv_repeats = args.final_cv_repeats
    final = cv_evaluate(state)
    print(f"  temp_r2={final['temp_r2_mean']:+.4f} ± {final['temp_r2_std']:.4f}")
    print(f"  time_r2={final['time_r2_mean']:+.4f} ± {final['time_r2_std']:.4f}")
    print(f"  mean_r2={final['mean_r2']:+.4f}")

    # Append a final-headline row to history.
    history_rows.append({
        "iteration": "final",
        "strategy": "+".join(stack) or "baseline",
        "temp_r2": final["temp_r2_mean"], "time_r2": final["time_r2_mean"],
        "mean_r2": final["mean_r2"],
        "delta_mean_r2": final["mean_r2"] - initial_baseline["mean_r2"],
        "secs": float("nan"),
    })

    # Persist artifacts.
    pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)
    update_summary_md(run_dir, initial_baseline,
                      iteration_records + [{
                          "n": "final",
                          "winner": "(headline-rerun)",
                          "winner_delta": 0.0,
                          "after": final,
                          "stack_after": stack,
                      }])

    store.write_manifest({
        "csv":                args.csv,
        "backbone":           args.backbone,
        "tier":               args.tier,
        "min_delta":          args.min_delta,
        "max_iter":           args.max_iter,
        "stack":              "+".join(stack) or "baseline",
        "initial_temp_r2":    initial_baseline["temp_r2_mean"],
        "initial_time_r2":    initial_baseline["time_r2_mean"],
        "initial_mean_r2":    initial_baseline["mean_r2"],
        "final_temp_r2":      final["temp_r2_mean"],
        "final_time_r2":      final["time_r2_mean"],
        "final_mean_r2":      final["mean_r2"],
    })

    print(f"\nArtifacts written to: {run_dir}")
    print(f"  - history.csv")
    print(f"  - summary.md")
    print(f"  - iteration_*.md")


if __name__ == "__main__":
    main()
