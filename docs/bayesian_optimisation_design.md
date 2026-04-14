# Bayesian Hyperparameter Optimisation Design

## Overview

Hyperparameter tuning is implemented in two places:

| Context | Notebook | Section | Scope |
|---|---|---|---|
| Full-dataset, all alloys | `microstructure_demo.ipynb` | §6b, cell 38 | Cycle 1 model, combined feature matrix |
| DP steel only, pipeline sweep | `pipeline_benchmark.ipynb` | §7, cells 18–20 | Tabular features, top-3 from regressor sweep |

Both use the same algorithmic design. This document describes it once and notes the
differences where they exist.

---

## Why Bayesian optimisation over grid/random search

The dataset is small (~82 rows for the full Cycle 1 set, ~54 rows for DP steel). Each
objective evaluation is a full `RepeatedKFold` cross-validation — on this data that
takes 2–8 seconds depending on the model. A grid search over 5 hyperparameters with
modest resolution (4 values each) is 4⁵ = 1024 evaluations, which is prohibitive.
Random search is cheaper but converges slowly when the optimal region is narrow.

Bayesian optimisation using a Tree-structured Parzen Estimator (TPE) builds a
probabilistic surrogate model of the objective after each trial and concentrates
subsequent trials in regions of high expected improvement. In practice this typically
finds configurations competitive with exhaustive search in 50–100 trials — a 10–20×
reduction in compute relative to grid search at the same resolution.

---

## Algorithm: Tree-structured Parzen Estimator (TPE)

TPE models `p(x | y < y*)` and `p(x | y >= y*)` separately, where `x` is a
hyperparameter configuration and `y*` is a quantile threshold on observed objective
values. New candidates are sampled by maximising the ratio of the two densities
(expected improvement proxy). This is the `TPESampler` in Optuna with default
`gamma=0.25` (top 25% defines the "good" distribution).

**Warm-up:** The first 20 trials use uniform random sampling (`n_startup_trials=20`)
to initialise the surrogate before switching to TPE. This prevents the early model
from overfitting to a handful of random points.

**Pruning:** `MedianPruner` is enabled with `n_startup_trials=10, n_warmup_steps=5`.
For models that support staged evaluation (not applicable here — the pruner is
included as a no-op guard for future compatibility with iterative estimators).

---

## Cross-validation protocol

Two CV objects are used:

```
CV_FAST = RepeatedKFold(n_splits=5, n_repeats=5,  random_state=42)   # search phase
CV_FULL = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)   # final eval
```

**Search phase (CV_FAST, 25 fits per trial):** Lighter CV reduces wall time per trial
without losing the variance-reduction benefit of repeated folds. On 82 samples a
single 5-fold split has high variance; 5 repeats already captures most of that.

**Final evaluation (CV_FULL, 50 fits):** The winning configuration from each model is
re-evaluated using the same `RepeatedKFold(5×10)` used throughout the rest of the
notebook. This ensures the Bayesian result is on the same scale as the untuned
baselines and can be directly compared.

**Scorer:** `make_scorer(r2_score, multioutput='uniform_average')` — averages R²
across both Cycle 1 targets (HoldingTemp, HoldingTime). A single scalar objective is
required by Optuna; uniform averaging treats both targets as equally important, which
matches the evaluation convention used everywhere else.

**Leakage:** The full dataset (`X_full_all`, `Y_full_all`) is used directly — no
separate test hold-out during the search phase. This is intentional: with 82 samples
holding out a test set during tuning would leave ~65 rows for CV and increase
variance further. The train/val/test split used for the downstream prediction
visualisation is a post-hoc held-out evaluation, not part of the optimisation loop.

---

## Model selection

### `microstructure_demo.ipynb`

Models to tune are selected dynamically at runtime:

```python
_cv_ranking = sorted(cv_scores.items(), key=lambda kv: kv[1].mean(), reverse=True)
_to_tune    = [name for name, _ in _cv_ranking[:2]]          # top-2 from CV
if trainer.best_model_name not in _to_tune:
    _to_tune.append(trainer.best_model_name)                  # always include trainer winner
_to_tune = list(dict.fromkeys(_to_tune))                      # deduplicate
```

This means the models tuned are not fixed at authoring time — they adapt to whichever
models performed best in the preceding CV cell. The trainer's best model is always
included as a guarantee that the model already in use gets a tuning pass.

### `pipeline_benchmark.ipynb`

Top-3 regressors from the §4 sweep are selected via `df_reg.head(3)['regressor']`.
The number is fixed at 3 (configurable via `TOP_N`) because the benchmark notebook
runs all 8 regressors in §4, and tuning all 8 would be expensive.

---

## Search spaces

Search bounds are set to be deliberately wide — the goal is to avoid artificially
constraining the search, not to encode prior knowledge about optimal values. Log-uniform
priors are used wherever a parameter spans orders of magnitude.

### Random Forest / Extra Trees

| Parameter | Type | Range | Prior |
|---|---|---|---|
| `n_estimators` | int | [50, 600] | uniform |
| `max_depth` | int | [3, 30] | uniform |
| `min_samples_leaf` | int | [1, 10] | uniform |
| `max_features` | categorical | `sqrt`, `log2`, `0.3`, `0.5` | uniform |
| `bootstrap` | categorical | `True`, `False` | uniform |

RF and ExtraTrees share a search space because they have identical hyperparameters.
The ensemble class is determined by the model name, not the space definition.

### Gradient Boosting Regressor (GBR)

| Parameter | Type | Range | Prior |
|---|---|---|---|
| `n_estimators` | int | [50, 500] | uniform |
| `learning_rate` | float | [0.001, 0.5] | **log-uniform** |
| `max_depth` | int | [2, 8] | uniform |
| `subsample` | float | [0.4, 1.0] | uniform |
| `min_samples_leaf` | int | [1, 20] | uniform |
| `max_features` | categorical | `sqrt`, `log2`, `None` | uniform |

`learning_rate` is log-uniform: values between 0.001 and 0.01 are qualitatively
different from values between 0.1 and 0.5, and uniform sampling would under-explore
the low end of the range where GBR often performs best on small datasets.

### AdaBoost Regressor (ABR)

| Parameter | Type | Range | Prior |
|---|---|---|---|
| `n_estimators` | int | [50, 400] | uniform |
| `learning_rate` | float | [0.01, 2.0] | **log-uniform** |
| `max_depth` | int | [1, 8] | uniform (base estimator) |

`max_depth` controls the base `DecisionTreeRegressor`, not AdaBoost itself.
It is extracted before constructing the regressor:

```python
max_depth = trial.suggest_int('max_depth', 1, 8)
...
AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=max_depth), ...)
```

### Support Vector Regressor (SVR)

**RBF kernel:**

| Parameter | Type | Range | Prior |
|---|---|---|---|
| `C` | float | [0.01, 1000] | **log-uniform** |
| `epsilon` | float | [0.001, 10] | **log-uniform** |
| `gamma` | categorical | `scale`, `auto` | uniform |

**Linear kernel:**

| Parameter | Type | Range | Prior |
|---|---|---|---|
| `C` | float | [0.01, 100] | **log-uniform** |
| `epsilon` | float | [0.001, 5] | **log-uniform** |

Both `C` and `epsilon` are log-uniform. The regularisation strength `C` is the
canonical example: `C=0.1` vs `C=1.0` is a qualitative difference in the model;
`C=100` vs `C=101` is not.

### K-Nearest Neighbours (KNN)

| Parameter | Type | Range | Prior |
|---|---|---|---|
| `n_neighbors` | int | [2, min(15, n//5)] | uniform |
| `weights` | categorical | `uniform`, `distance` | uniform |
| `p` | int | [1, 2] | uniform |

`n_neighbors` upper bound is clipped to `n_samples // 5` to prevent degenerate
configurations where k spans a large fraction of the dataset.
`p=1` is Manhattan distance; `p=2` is Euclidean.

---

## Multi-output wrapping

sklearn's `GradientBoostingRegressor`, `AdaBoostRegressor`, `SVR`, and
`KNeighborsRegressor` are single-output estimators. They are wrapped in
`MultiOutputRegressor`, which fits one independent model per target column.
`RandomForestRegressor` and `ExtraTreesRegressor` natively support multi-output
and are not wrapped.

For GBR and ABR, `StandardScaler(with_mean=False)` is included in a `Pipeline`
inside the wrapper. This is consistent with how the model trainer constructs these
estimators and ensures the scaling applied during search matches production.

---

## `_FixedTrial` helper

After the search completes, the best parameter configuration needs to be used to
construct a fresh model instance for final evaluation and fitting. Optuna does not
provide a direct "reconstruct from params dict" API, so a minimal duck-typed
`_FixedTrial` class is used:

```python
class _FixedTrial:
    def __init__(self, p):                       self._p = p
    def suggest_int(self, n, *a, **kw):          return self._p[n]
    def suggest_float(self, n, *a, **kw):        return self._p[n]
    def suggest_categorical(self, n, *a, **kw):  return self._p[n]
```

Passing `_FixedTrial(study.best_params)` to `_build()` produces a model with
the exact hyperparameter values selected by Optuna, without re-running any trials.
This pattern guarantees the final model is identical to what was evaluated.

---

## Outputs

### Runtime outputs (printed)

```
Optimising RF (80 trials)...
  best CV R² = 0.8312  (47s)
  params    = {'n_estimators': 312, 'max_depth': 11, ...}

Final evaluation — RepeatedKFold(5×10):
  RF            untuned=0.8010  tuned=0.8312±0.0541  Δ=+0.0302 ⭐
  ABR           untuned=0.7950  tuned=0.8105±0.0614  Δ=+0.0155

Best tuned model : RF  CV R² = 0.8312
```

### `runs/best_hyperparams.json`

One entry per tuned model:

```json
{
  "RF": {
    "best_cv_r2": 0.8312,
    "params": {
      "n_estimators": 312,
      "max_depth": 11,
      "min_samples_leaf": 2,
      "max_features": 0.3,
      "bootstrap": true
    }
  }
}
```

### `runs/bayes_tuning.png`

Two-panel figure:
- **Left panel:** grouped bar chart comparing untuned vs Bayesian-tuned mean CV R²
  per model, with Δ annotated in green (improvement) or red (regression).
- **Right panel:** best-so-far optimisation curve per model across 80 trials —
  shows the convergence rate and whether more trials would be useful.

### FAnova parameter importance

After final evaluation, `optuna.importance.get_param_importances()` runs a
functional ANOVA decomposition over the completed trials to estimate the fraction
of objective variance explained by each hyperparameter. This is printed to stdout:

```
RF:
  n_estimators             0.412  ████████████
  max_depth                0.298  █████████
  min_samples_leaf         0.171  █████
  max_features             0.089  ██
  bootstrap                0.030  █
```

High importance on a single parameter (e.g. `learning_rate` for GBR) is a signal
that the search space for that parameter should be examined — the optimal region may
sit at the boundary of the defined range.

### Metrics log integration (`runs/metrics_log.csv`)

The logging cell (cell 53 in `microstructure_demo.ipynb`) reads `best_tuned_r2` and
`bayes_eval` from the Bayesian cell and auto-tags the run:

```
bayes_best=RF,r2=0.8312,delta=+0.0302
```

If the tuned CV R² exceeds the untuned estimate, `c1_cv_r2_mean` in the log row is
overwritten with the tuned value so the log reflects the best achievable estimate
for that run.

---

## Variable dependencies

```
Cell 37 (CV)
  → cv_scores: dict[str, ndarray]   # untuned CV scores per model
  → X_full_all, Y_full_all           # full dataset arrays
  → _multi_r2                        # scorer

Cell 38 (Bayes)  [reads all of the above]
  → bayes_studies: dict[str, Study]  # one Optuna study per model
  → bayes_eval: dict[str, dict]      # tuned metrics per model
  → best_tuned_name: str
  → best_tuned_r2: float
  → best_tuned_params: dict
  → best_tuned_model: fitted estimator

Cell 53 (Logger) [reads bayes_eval, best_tuned_name, best_tuned_r2]
  → runs/metrics_log.csv             # appended
  → auto-tag: "bayes_best=...,r2=...,delta=..."
```

Cells 39–52 (pruning, SMOGN, predictions, C1+2 extension) run between the Bayesian
cell and the logger. They do not depend on Bayesian outputs and do not overwrite
`bayes_eval` or `best_tuned_model`.

---

## Configuration

| Constant | Location | Default | Effect |
|---|---|---|---|
| `N_TRIALS` | cell 38 | `80` | Trials per model. Increase for more thorough search. |
| `TOP_N` | benchmark cell 18 | `3` | Models tuned in pipeline_benchmark.ipynb |
| `CV_FAST` | cell 38 | `RKF(5,5)` | CV during search. Increase repeats for less noisy objective. |
| `CV_FULL` | cell 38 | `RKF(5,10)` | CV for final evaluation. Must match the CV in cell 37. |
| `n_startup_trials` | TPESampler | `20` | Random trials before TPE activates. Raise for wider exploration on first run. |

---

## Limitations

**Single-objective only.** Both targets (HoldingTemp, HoldingTime) are collapsed into
a single R² scalar via `multioutput='uniform_average'`. A configuration that is
excellent for one target but poor for the other can score well overall. Multi-objective
Optuna (`direction=['maximize', 'maximize']`) would surface the Pareto front but
requires downstream logic to select a single configuration from it.

**No preprocessing co-optimisation.** The search fixes preprocessing at whatever was
applied in cell 15 (demo notebook) or the best config from §3 (benchmark notebook).
Jointly optimising preprocessing and model hyperparameters would require fitting the
preprocessor inside each CV fold of each trial — roughly a 5× increase in compute.
The pipeline benchmark addresses this as a separate stage (§3) rather than coupling
it to the model search.

**Small dataset variance.** On 82 samples even `RepeatedKFold(5×10)` has non-trivial
variance (~±0.05 R²). The "best" configuration found by Optuna may not be
meaningfully better than the second-best. The Δ column in outputs should be
interpreted cautiously when |Δ| < 0.02.

**No warm-starting across runs.** Each execution of cell 38 starts a fresh Optuna
study. Previous trials are not loaded. If the notebook is re-run after changing the
feature matrix (e.g. adding a new backbone cache), the search starts from scratch,
which is correct behaviour — the objective has changed.

---

## Standalone tuning workflow

### The problem

Running Bayesian optimisation inside the pipeline benchmark or main demo notebook
couples a slow operation (100 trials × 25 CV fits ≈ 5–15 minutes) to a routine
pipeline run. Every re-run of the pipeline would re-tune from scratch or skip tuning,
neither of which is ideal.

### Solution: `bayes_tuning.ipynb` + `runs/hyperparams.json`

```
bayes_tuning.ipynb        (run when you want to tune)
        │
        ▼
runs/hyperparams.json     (canonical store, scope-keyed)
        │
        ├──▶  pipeline_benchmark.ipynb  §4 regressor cell reads at startup
        └──▶  microstructure_demo.ipynb §6b CV cell reads at startup
```

### `src/hyperparams.py` — the store API

| Function | Description |
|---|---|
| `save(scope, models_dict, ...)` | Write / merge one scope into the store |
| `load(scope)` | Return `{model_name: {params, tuned_cv_r2, ...}}` or `{}` |
| `best_model(scope)` | Return `(name, params)` for the top model |
| `has_params(scope, model_name)` | Check before loading |
| `list_scopes()` | All scopes currently stored |
| `summary()` | Human-readable overview of the store |

### Store schema

```json
{
  "dp_steel": {
    "saved_at":             "2026-04-14 11:23:01",
    "git_commit":           "8273fd7",
    "n_trials":             100,
    "cv_protocol":          "RepeatedKFold(n_splits=5, n_repeats=5) ...",
    "feature_matrix_shape": [54, 38],
    "best_model":           "RF",
    "models": {
      "RF":  {"best_cv_r2": 0.80, "tuned_cv_r2": 0.83, "delta": 0.03,
              "params": {"n_estimators": 312, "max_depth": 11, ...}},
      "GBR": { ... }
    }
  },
  "all_alloys": { ... }
}
```

Multiple scopes coexist in the same file. Re-running `bayes_tuning.ipynb` with the
same scope overwrites only that scope; other scopes are preserved.

### Scopes

| Scope | Dataset filter | Written by | Read by |
|---|---|---|---|
| `dp_steel` | DP / dual-phase only | `bayes_tuning.ipynb` or `pipeline_benchmark.ipynb §7` | `pipeline_benchmark.ipynb §4` |
| `all_alloys` | Full dataset | `bayes_tuning.ipynb` or `microstructure_demo.ipynb §6b` | `microstructure_demo.ipynb §6b` |

### How the pipeline consumes saved params

**`pipeline_benchmark.ipynb` §4 (cell 10):**

```python
_saved = hp.load('dp_steel')
# REGRESSORS dict uses saved params where available, defaults otherwise
```

Each model in `REGRESSORS` calls `_from_saved(name)` which returns the tuned
`params` dict if it exists, `None` otherwise. The kwargs are unpacked directly into
the estimator constructor. Models without saved params (e.g. `KNN_5`, `KNN_3`) are
constructed with their default values unchanged.

A per-model annotation (`[saved] ✓` / `default`) is printed alongside CV scores so
it is always clear which models are using tuned vs default params.

**`microstructure_demo.ipynb` §6b (cell 37):**

```python
_saved_params = hp.load('all_alloys')
```

The loaded dict is available for use in the CV and Bayesian cells. Cell 38 appends
its results back to the store via `hp.save('all_alloys', ...)`.

### Running tuning independently

```bash
cd notebooks
jupyter notebook bayes_tuning.ipynb
```

1. Set `TUNING_SCOPE` in §1 (`"dp_steel"` or `"all_alloys"`)
2. Optionally adjust `N_TRIALS`, `TOP_N`, preprocessing config
3. Run all cells
4. Results persist to `runs/hyperparams.json`
5. Next pipeline run picks them up automatically — no further action needed
