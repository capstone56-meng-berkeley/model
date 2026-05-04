# Iteration 3

_Generated: 2026-05-04 12:59:07 PDT_

## Baseline going in

- Cumulative stack: `['log_time', 'stratify_alloy']`
- Folds: 15

| target | R² | MAE | RMSE |
|---|---|---|---|
| HoldingTemp | `+0.7163 ± 0.0772` | `18.42` | `30.12` |
| HoldingTime | `+0.8000 ± 0.2558` | `1.95` | `7.68` |
| **mean R²** | `+0.7581` | | |

## Candidates tested this iteration

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
| HoldingTemp | `+0.7163` | `+0.0000` | `18.42` | `+0.00` | `30.12` | `+0.00` |
| HoldingTime | `+0.8000` | `+0.0000` | `1.95` | `+0.00` | `7.68` | `+0.00` |
| **mean R²** | `+0.7581` | `+0.0000` | | | | |

_Wall time: `388.9s`_

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
| HoldingTemp | `+0.7059` | `-0.0104` | `18.48` | `+0.06` | `30.57` | `+0.45` |
| HoldingTime | `+0.7999` | `-0.0001` | `2.03` | `+0.08` | `7.80` | `+0.12` |
| **mean R²** | `+0.7529` | `-0.0052` | | | | |

_Wall time: `384.4s`_

## Outcome

**No candidate cleared the improvement threshold — iteration loop ends.**
