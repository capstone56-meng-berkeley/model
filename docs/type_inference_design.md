# Type Inference and Handler Architecture

## Overview

The preprocessing pipeline uses a registry-based, per-column handler architecture.
Each column is assigned a type (automatically inferred or explicitly overridden),
then processed through a three-stage pipeline: **impute → encode → scale**.

All components — imputers, encoders, scalers, and type handlers — are registered
by string key. New strategies can be added without modifying the pipeline
orchestrator.

---

## The Registry Pattern

Defined in `src/registry.py`. `Registry` is a base class with a `_registry` dict
per subclass and three class-method entry points:

| Method | Purpose |
|--------|---------|
| `@cls.register("name")` | Decorator that registers a class under a key |
| `cls.get("name")` | Returns the class or `None` |
| `cls.get_or_raise("name")` | Returns the class or raises `ValueError` with available keys |
| `cls.create("name", *args, **kwargs)` | Instantiates the registered class |
| `cls.list_available()` | Returns all registered keys |

Double-registration raises immediately (`ValueError: Component 'X' already
registered`), preventing silent shadowing.

Four registries are used in the pipeline:

| Registry | Location | Registered strategies |
|----------|----------|----------------------|
| `TypeHandlerRegistry` | `type_handlers.py` | `numeric`, `categorical`, `text`, `unique_string`, `boolean`, `datetime` |
| `ImputerRegistry` | `imputers.py` | `mean`, `median`, `mode`, `constant`, `forward_fill`, `unknown` (+ `mice` via `MICEImputer`) |
| `EncoderRegistry` | `encoders.py` | `passthrough`, `onehot`, `label`, `ordinal`, `binary`, `tfidf` |
| `ScalerRegistry` | `scalers.py` | `none`, `standard`, `minmax`, `robust`, `maxabs` |

---

## Type Constants

Defined in `src/preprocessing/type_handlers.py`:

| Constant | Value | Mapped to dtype(s) |
|----------|-------|--------------------|
| `TYPE_NUMERIC` | `"numeric"` | `int8/16/32/64`, `uint*`, `float16/32/64` |
| `TYPE_CATEGORICAL` | `"categorical"` | `category` dtype; object with low uniqueness |
| `TYPE_TEXT` | `"text"` | object, high uniqueness (>90%) + long strings (avg > 50 chars) |
| `TYPE_UNIQUE_STRING` | `"unique_string"` | object, high uniqueness (>90%) + short strings |
| `TYPE_BOOLEAN` | `"boolean"` | `bool` dtype |
| `TYPE_DATETIME` | `"datetime"` | `datetime64` |

---

## Auto-Inference Logic

Implemented in `FeaturePreprocessor._infer_column_type()` and
`_infer_object_type()` in `src/preprocessing/pipeline.py`.

### Dtype-based dispatch (`_infer_column_type`)

Rules are evaluated in priority order:

```
bool dtype            → TYPE_BOOLEAN
numeric dtype         → TYPE_NUMERIC
datetime64 dtype      → TYPE_DATETIME
category dtype        → TYPE_CATEGORICAL
object/string dtype   → _infer_object_type()  (heuristics)
anything else         → TYPE_CATEGORICAL       (fallback)
```

### Object-dtype heuristics (`_infer_object_type`)

For `object`, `string`, or `str` dtypes, two statistics are computed on the
non-null values:

- **Uniqueness ratio** = `nunique / n_non_null`
- **Average string length** = mean of `str.len()` across non-null values

Decision tree:

```
uniqueness_ratio > 0.9
    avg_length > 50  → TYPE_TEXT          (free-form content: descriptions, notes)
    avg_length ≤ 50  → TYPE_UNIQUE_STRING (identifiers, batch codes, names)
uniqueness_ratio ≤ 0.9  → TYPE_CATEGORICAL
```

An empty column (all null) defaults to `TYPE_CATEGORICAL`.

---

## Column Type Overrides

`FeaturePreprocessor` accepts an optional `column_types: Dict[str, str]` argument
that bypasses auto-inference for named columns:

```python
COLUMN_TYPE_OVERRIDES = {
    "num_cycles":          "numeric",
    "heat_treatment_type": "categorical",
    "id":                  "unique_string",
    ...
}
preprocessor = FeaturePreprocessor(config, column_types=COLUMN_TYPE_OVERRIDES)
```

**Drift check**: during `fit()`, the pipeline verifies that every key in
`column_types` exists in the DataFrame. A mismatch raises immediately:

```
ValueError: column_types override(s) refer to columns not present in the
DataFrame: ['old_column_name']. Update config.json column_types to match
the current dataset schema.
```

This surfaces schema drift (renamed or removed columns) before the pipeline
produces silently incorrect output.

**Scoping**: pass only the subset of overrides that apply to the current
feature set. In the notebook, `active_overrides` is filtered to
`{k: v for k, v in COLUMN_TYPE_OVERRIDES.items() if k in feature_columns}` so
the drift check does not fire on columns intentionally excluded from the run.

---

## Handler Pipeline: impute → encode → scale

Each handler (`BaseTypeHandler`) wraps three components in a fixed order.
`BaseTypeHandler` defines the interface (`fit`, `transform`, `get_feature_names`);
concrete handlers are registered and instantiated via `TypeHandlerRegistry`.

The three components are themselves abstract (`BaseImputer`, `BaseEncoder`,
`BaseScaler`), each with a `fit`/`transform` contract. All are optional; a
handler can pass `None` for components it does not need.

### Handler defaults and configuration

`_create_handler()` in `pipeline.py` selects strategies based on:
- `column_type` — determines which registry entries are valid
- `missing_ratio` — controls which imputer strategy is chosen
- `config` (`PreprocessingConfig`) — carries user-configured strategy names

**Imputer selection by missing ratio:**

| Condition | Imputer chosen |
|-----------|---------------|
| `missing_ratio ≤ row_fill_threshold` | `numeric_fill_strategy` (default `"median"`) for numeric; `categorical_fill_strategy` (default `"mode"`) for others |
| `missing_ratio > row_fill_threshold`, `mid_range_strategy == "fill"` | Same as above |
| `missing_ratio > row_fill_threshold`, `mid_range_strategy == "flag"` | `"constant"` (TODO: add indicator feature in this branch) |
| `missing_ratio > column_drop_threshold` | Column dropped before handler is created |

Note: `missing_ratio` is computed from a pre-pass snapshot (`original_missing =
df.isna().mean()`) taken before indicator zero-fill and MICE run, so the drop
threshold reflects true original missingness rather than post-imputation sparsity.

---

## Type Handlers

### `NumericHandler` (registered as `"numeric"`)

| Component | Default | Override param |
|-----------|---------|---------------|
| Imputer | `median` | `impute_strategy` |
| Encoder | `passthrough` | — |
| Scaler | `standard` | `scale_method` |

`fit()` calls `pd.to_numeric(series, errors='coerce')` before fitting imputer,
encoder, and scaler in order. `transform()` repeats the same coercion so
non-numeric strings are treated as NaN rather than raising.

---

### `CategoricalHandler` (registered as `"categorical"`)

| Component | Default | Override param |
|-----------|---------|---------------|
| Imputer | `mode` | `impute_strategy` |
| Encoder | `onehot` | `encode_method` |
| Scaler | `none` | — |

**NaN-safe string conversion**: both `fit()` and `transform()` use:

```python
series.where(series.isna(), series.astype(str))
```

This preserves `NaN` as `NaN` — not as the string `"nan"`. The previous
`.astype(str).replace('nan', np.nan)` approach would silently convert any
real category value spelled `"nan"` into missing.

No scaler is applied — one-hot columns are already binary.

---

### `TextHandler` (registered as `"text"`)

| Component | Default | Override param |
|-----------|---------|---------------|
| Imputer | `constant("")` | — |
| Encoder | `tfidf` | `encode_method` |
| Scaler | `none` | — |

When `encode_method="skip"`, both `fit()` and `transform()` return immediately
with zero output features. This is the correct behavior for free-text columns
(e.g. `article_url`) that should not contribute features.

---

### `UniqueStringHandler` (registered as `"unique_string"`)

| Component | Default | Override param |
|-----------|---------|---------------|
| Imputer | `unknown` | — |
| Encoder | `skip` | `encode_method` |
| Scaler | `none` | — |

Default `encode_method` is `"skip"` — identifier columns (`id`, `alloy`,
`original_image`) carry no ordinal or semantic signal useful to a regressor.
The previous default was `"label"`, which assigned arbitrary integers to IDs
and injected meaningless ordinal information.

---

### `BooleanHandler` (registered as `"boolean"`)

| Component | Default | Override param |
|-----------|---------|---------------|
| Imputer | `mode` | `impute_strategy` |
| Encoder | `passthrough` | — |
| Scaler | `none` | — |

**Explicit bool map** via `_map_bool()`:

```python
_BOOL_MAP = {
    True: 1, False: 0,
    'True': 1, 'False': 0,
    'true': 1, 'false': 0,
    '1': 1, '0': 0,
    1: 1, 0: 0,
}
```

Values not in the map produce `NaN` and trigger a `warnings.warn` with the
specific unmapped values and their count. Previously, unmapped values silently
became `NaN` with no indication of data quality issues.

---

### `DatetimeHandler` (registered as `"datetime"`)

Decomposes a datetime column into four numeric features:
`{col}_year`, `{col}_month`, `{col}_day`, `{col}_dayofweek`.

| Component | Default | Notes |
|-----------|---------|-------|
| Imputer | `forward_fill` | Applied before decomposition |
| Encoder | `None` | Decomposition happens inline |
| Scaler | `none` | — |

`pd.to_datetime(series, errors='coerce')` converts the series first; unparseable
values become `NaT`, which `forward_fill` then resolves using the preceding
timestamp. The resulting `dt` accessor is applied without `fillna(0)` — the
previous approach produced `year=1970` for any remaining nulls, which is not a
real date.

---

## Encoders

| Key | Class | Output | Use case |
|-----|-------|--------|----------|
| `passthrough` | `PassthroughEncoder` | `(n, 1)` float64 | Numeric and boolean columns |
| `onehot` | `OneHotEncoder` | `(n, k)` binary | Categorical columns (default) |
| `label` | `LabelEncoder` | `(n, 1)` int as float64 | Categorical with many levels or ordinal-by-index |
| `ordinal` | `OrdinalEncoder` | `(n, 1)` float64 | Categorical with known order; falls back to sorted order |
| `binary` | `BinaryEncoder` | `(n, ceil(log2(k+1)))` binary bits | Compact categorical encoding |
| `tfidf` | `TfidfEncoder` | `(n, max_features)` float64 | Free-form text |

`OneHotEncoder` respects `max_categories` — categories beyond the limit are
silently dropped at fit time. Unknown categories at transform time are ignored
(all-zero row) when `handle_unknown="ignore"` (default).

`LabelEncoder` maps unknown values to `len(mapping)` (one beyond the last known
integer), making unseen categories distinguishable from known ones.

---

## Scalers

| Key | Class | Method | Use case |
|-----|-------|--------|----------|
| `none` | `NoScaler` | Identity | Categorical, boolean, datetime outputs |
| `standard` | `StandardScaler` | `(x - mean) / std` | Default for numeric; assumes approximate normality |
| `minmax` | `MinMaxScaler` | `(x - min) / (max - min)` | When bounded range is required |
| `robust` | `RobustScaler` | `(x - median) / IQR` | Numeric columns with outliers |
| `maxabs` | `MaxAbsScaler` | `x / max(|x|)` | Sparse data; preserves zero |

All scalers handle zero-variance columns (constant features) by setting the
denominator to 1.0, returning the input unchanged rather than raising.

---

## Pre-pass Architecture

Two pre-passes run before any per-column handler sees the data, ordered to
prevent leakage between them:

```
Pre-pass 1 (indicators):  Ti, Nb, V → {col}_present + zero-fill
                          ↓
Pre-pass 2 (MICE):        Cr, Mo, S, Ni, Al imputed jointly
                          ↓
Per-column handlers:      all remaining columns
```

Indicators must precede MICE so that MICE never encounters NaN in the indicator
columns during its iterative regression. The missing-ratio snapshot is taken
before both pre-passes so the column-drop threshold reflects original sparsity.

See `docs/imputation_design.md` for the rationale behind this split.

---

## Adding a New Type or Strategy

**New imputer/encoder/scaler:**

```python
@ImputerRegistry.register("my_strategy")
class MyImputer(BaseImputer):
    def fit(self, series): ...
    def transform(self, series): ...
```

No changes to `pipeline.py` — the registry resolves the key at runtime.

**New type handler:**

```python
@TypeHandlerRegistry.register("my_type")
class MyHandler(BaseTypeHandler):
    def fit(self, series): ...
    def transform(self, series): ...
```

Then add `"my_type"` to the type-constant checks in `_infer_column_type()` or
assign it via `column_types` override. No other changes needed.

**New column in the dataset:**

The pipeline auto-infers types for new columns. If the inference is wrong, add
an explicit entry to `COLUMN_TYPE_OVERRIDES` in `run_training.py` / `main.py` /
the notebook using the sanitized column name.
