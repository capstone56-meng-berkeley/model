# Iteration 2

_Generated: 2026-05-04 13:54:30 PDT_

## Baseline going in

- Cumulative stack: `['stratify_alloy']`
- Folds: 5

| target | R² | MAE | RMSE |
|---|---|---|---|
| HoldingTemp | `+0.7106 ± 0.0967` | `17.55` | `27.75` |
| HoldingTime | `+0.7760 ± 0.2708` | `2.30` | `6.89` |
| **mean R²** | `+0.7433` | | |

## Candidates tested this iteration

### `log_time` — ❌ rejected

Wrap HoldingTime target with log1p / expm1. The time range (10 to 90+ minutes) is roughly log-uniform, so linear-MSE over-weights long-time samples.

**Diff:**

```python
from sklearn.compose import TransformedTargetRegressor
model = TransformedTargetRegressor(
    regressor=GradientBoostingRegressor(...),
    func=np.log1p, inverse_func=np.expm1)
```

**Per-target metrics (Δ vs baseline):**

| target | R² | Δ R² | MAE | Δ MAE | RMSE | Δ RMSE |
|---|---|---|---|---|---|---|
| HoldingTemp | `+0.7106` | `+0.0000` | `17.55` | `+0.00` | `27.75` | `+0.00` |
| HoldingTime | `+0.7537` | `-0.0224` | `2.11` | `-0.18` | `7.29` | `+0.39` |
| **mean R²** | `+0.7321` | `-0.0112` | | | | |

_Wall time: `107.3s`_

### `per_target` — ❌ rejected

Train two independent single-output GBRs (one per target) instead of MultiOutputRegressor wrapping a joint multi-output GBR. Decouples the two targets so each tree depth/split can specialise.

**Diff:**

```python
# before:
model = MultiOutputRegressor(GradientBoostingRegressor(...))
# after:
m_temp = GradientBoostingRegressor(...)  # fits Y[:, 0] only
m_time = GradientBoostingRegressor(...)  # fits Y[:, 1] only
```

**Per-target metrics (Δ vs baseline):**

| target | R² | Δ R² | MAE | Δ MAE | RMSE | Δ RMSE |
|---|---|---|---|---|---|---|
| HoldingTemp | `+0.7106` | `+0.0000` | `17.55` | `+0.00` | `27.75` | `+0.00` |
| HoldingTime | `+0.7760` | `+0.0000` | `2.30` | `+0.00` | `6.89` | `+0.00` |
| **mean R²** | `+0.7433` | `+0.0000` | | | | |

_Wall time: `107.7s`_

### `stratify_temp_bin` — ❌ rejected

Use StratifiedKFold by HoldingTemp quantile-bin (5 bins) so rare setpoints aren't entirely on one side of the split.

**Diff:**

```python
y_bin = pd.qcut(Y[:, 0], q=5, duplicates='drop').astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y_bin))
```

**Per-target metrics (Δ vs baseline):**

| target | R² | Δ R² | MAE | Δ MAE | RMSE | Δ RMSE |
|---|---|---|---|---|---|---|
| HoldingTemp | `+0.6655` | `-0.0451` | `18.47` | `+0.92` | `30.49` | `+2.74` |
| HoldingTime | `+0.7846` | `+0.0086` | `2.38` | `+0.09` | `6.77` | `-0.12` |
| **mean R²** | `+0.7250` | `-0.0183` | | | | |

_Wall time: `107.6s`_

## Outcome

**No candidate cleared the improvement threshold — iteration loop ends.**
