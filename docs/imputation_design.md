# Imputation Design: Chemical Composition Features

## Overview

Missing values in chemical composition columns are not all alike. The imputation
strategy is split by the *mechanism* of missingness rather than applying a single
approach uniformly. Using the wrong strategy for a given mechanism either destroys
information (imputing a structurally-absent element) or introduces noise (flagging
a randomly-missing measurement as a design feature).

---

## Column groups and strategies

### Group 1 — Structural absence: `Ti`, `Nb`, `V`

**Strategy: zero-fill + binary presence indicator (`{col}_present`)**

These are deliberate microalloying additions. When a steel does not contain Ti, Nb,
or V it is because the alloy was not designed to include them — the true concentration
is 0 wt%. The value is not missing at random; it is absent by design.

- `{col}_present = 1` → element was added; the recorded value is meaningful
- `{col}_present = 0` → element was not added; zero-fill is the physically correct value

The indicator carries alloy-family information the model would otherwise lose. A
plain carbon steel and a Ti-microalloyed steel are fundamentally different alloy
classes; the presence flag encodes that boundary directly.

**Why not MICE here:** MICE would impute a plausible non-zero concentration for
samples where the element was never added. This is physically wrong and would
obscure the alloy-family signal.

**Why not indicators for all columns:** See the rejected alternative section below.

---

### Group 2 — Random missingness in correlated elements: `Cr`, `Mo`, `S`, `Ni`, `Al`

**Strategy: MICE (Multiple Imputation by Chained Equations)**

These elements are present in many alloys at varying concentrations but are not
always reported in the dataset. Their absence reflects a data collection gap, not
a design choice. Crucially, their concentrations are physically correlated — higher
Cr often co-occurs with higher Mo and Ni in engineering steels — so the full
multivariate distribution carries information that single-column imputation discards.

MICE fits an iterative regression across all five columns simultaneously, recovering
plausible values consistent with the observed correlations. This produces more
informative imputed values than median imputation, which collapses the distribution
to a single point and breaks the inter-element correlation structure.

**Implementation:** `sklearn.impute.IterativeImputer` with `max_iter=10`,
`random_state=42`, `skip_complete=True`. Fitted on the training split only;
`transform()` is applied to val and test separately to prevent leakage.

**Why not indicators here:** Adding a `cr_present` indicator while simultaneously
running MICE sends contradictory signals — MICE says "recover the true value" and
the indicator says "flag that the value was missing". The indicator would also be
collinear with post-imputation values and add noise on a small dataset.

---

### Group 3 — Rarely missing, no inter-element correlation: `C`, `Mn`, `Si`, `Cu`, `P`, `Fe`

**Strategy: median imputation**

These columns are rarely missing in the dataset (typically <10%). Median imputation
is sufficient at low missingness rates and introduces minimal bias. There is no
strong physical correlation between, say, C content and Cu content that MICE could
exploit, so the added complexity of MICE is not justified.

---

## Rejected alternative: indicators for all composition columns

A uniform approach — adding `{col}_present` for every chemical column — was
considered and rejected for the following reasons:

1. **Near-constant features for common elements.** `C`, `Mn`, `Si`, `Fe` are
   present in virtually every sample. A `c_present` indicator would be 1 for
   >95% of rows: zero variance, no signal, pure noise.

2. **Contradicts MICE for Group 2.** Running MICE and adding an indicator
   simultaneously is internally inconsistent. MICE is premised on MAR — the value
   exists but was not recorded. An indicator is premised on the value being
   meaningfully absent. Both cannot be true at once.

3. **Feature budget on a small dataset.** With 88 samples and 14 composition
   columns, adding 14 binary indicators would nearly double the feature count
   from chemistry alone. At ~6 samples per feature in a Random Forest, near-useless
   binary features increase variance without adding predictive signal.

4. **Ambiguity between zero and missing.** After zero-fill, `Al=0.0` (a valid
   measured value) and `Al=NaN` (not reported) both produce `al_present=0`. The
   indicator cannot distinguish a true zero from an unreported value, making it
   misleading for columns where zero is a physically valid measurement.

The right trigger for adding an indicator to a column is evidence that its
missingness correlates with a distinct alloy family — detectable via a co-occurrence
analysis or a `missingness ~ heat_treatment_type` groupby. Without that evidence,
the default is median or MICE depending on the mechanism above.

---

## Leakage prevention

The preprocessor (`FeaturePreprocessor`) applies all imputation inside `fit()`,
which is called only on the training split. `transform()` applies the fitted
imputer state to val and test sets. MICE in particular must not see test-set rows
during fitting, as it uses the full matrix to infer missing values — fitting on
the full dataset would allow test-set chemistry to influence training-set imputed
values.

See `run_training.py` for the split-before-fit ordering, and
`src/preprocessing/pipeline.py` for the pre-pass architecture (indicators first,
then MICE, then per-column handlers).

---

## Configuration

The column assignments are defined in `run_training.py` and `main.py`:

```python
MICE_COLUMNS      = ["cr", "mo", "s", "ni", "al"]   # Group 2
INDICATOR_COLUMNS = ["ti", "nb", "v"]                # Group 1
# Group 3 columns use median fill via MissingDataConfig.numeric_fill_strategy
```

MICE iteration count is controlled via `config.json`:

```json
"missing_data": {
  "mice_max_iter": 10
}
```

---

## Validation

`notebooks/imputation_validation.ipynb` benchmarks three strategies against
5-fold cross-validated R² and MAE on a Random Forest regressor:

| Strategy | Description |
|---|---|
| A — Baseline | Median imputation for all columns |
| B — MICE only | MICE for Group 2, median for rest |
| C — MICE + indicators | MICE for Group 2, zero-fill + indicator for Group 1 |

The notebook also plots imputed value distributions (observed vs median-fill vs
MICE-fill) and indicator prevalence rates for Ti, Nb, V.
