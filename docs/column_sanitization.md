# Column Name Sanitization

## Problem

The source dataset (`metadata_latest.csv`) contains column names that are
unusable as-is in downstream code:

- Irregular whitespace: `'Cycle3_HoldingTemp   (°C)'`, `'Cycle1_   Qtype'`
- Special characters: `°`, `%`, `+`, `/`, `.`
- Mixed case: `'Heat treatment type'`, `'Cycle1_Rolling'`
- Parenthesised units with inconsistent spacing: `'Cycle1_CRate (°C/s)'` vs `'Cycle3_CRate   (°C/s)'`
- No separator before unit: `'Cycle1_Qtime(min)'`

These make column references brittle — a trailing space or a degree symbol in a
string literal causes a silent `KeyError`. The original codebase worked around
this with a `find_column()` fuzzy-matcher that normalised whitespace at every
lookup site. This is fragile: every new lookup needed the same workaround, and
it could still silently match the wrong column if two names normalised to the
same string.

---

## Solution

Sanitize all column names **once, immediately after loading**, and use the
sanitized names exclusively from that point forward. No fuzzy matching is needed
anywhere downstream.

```python
from src.column_sanitizer import sanitize_dataframe
df = pd.read_csv(path, header=1)
df = sanitize_dataframe(df)   # all subsequent code uses sanitized names
```

This is applied in:
- `run_training.py` — `load_data()`, right after `pd.read_csv` / Sheets load
- `main.py` — after `data_loader.load_data()`
- `notebooks/microstructure_demo.ipynb` — cell immediately after `read_csv`

---

## Sanitization rules

Implemented in `src/column_sanitizer.py`, function `sanitize_column(name)`.
Rules are applied in order:

| Step | Rule | Example |
|------|------|---------|
| 1 | Strip leading/trailing whitespace | `' Si '` → `'Si'` |
| 2 | Lowercase | `'Heat treatment type'` → `'heat treatment type'` |
| 3 | Insert space before `(` attached to a word char | `'Qtime(min)'` → `'Qtime (min)'` |
| 4 | Replace unit patterns with readable tokens | `'(°C/s)'` → `'degc_s'` |
| 5 | Replace all remaining non-`[a-z0-9]` with `_` | `'heat treatment type'` → `'heat_treatment_type'` |
| 6 | Collapse runs of underscores | `'cycle1___qtype'` → `'cycle1_qtype'` |
| 7 | Strip leading/trailing underscores | `'_si_'` → `'si'` |

### Unit token map

Units are replaced with readable tokens before the generic stripping step so
that context is preserved in the sanitized name.

| Pattern | Token | Example |
|---------|-------|---------|
| `(°C/s)` | `degc_s` | `cycle1_crate_degc_s` |
| `(C/s)` | `degc_s` | `cycle2_crate_degc_s` |
| `(°C)` | `degc` | `cycle3_holdingtemp_degc` |
| `(C)` | `degc` | `cycle1_holdingtemp_degc` |
| `(C/min)` | `degc_min` | `cycle1_hrate_degc_min` |
| `(min)` | `min` | `cycle1_holdingtime_min` |
| `(%)` | `pct` | — |

`(wt.%)` is **not** in the token map. Compositional columns in the source
dataset no longer carry that suffix — element symbols are bare (`C`, `Mn`,
`Cr`, etc.) and sanitize to their lowercase equivalents (`c`, `mn`, `cr`).

---

## Full column mapping

| Original | Sanitized |
|----------|-----------|
| `Alloy` | `alloy` |
| `Article url` | `article_url` |
| `ID` | `id` |
| `Original_Image` | `original_image` |
| `AI_Cleaned_Image` | `ai_cleaned_image` |
| `Augumented_Data` | `augumented_data` |
| `C` | `c` |
| `Mn` | `mn` |
| `Si` | `si` |
| `Cr` | `cr` |
| `P` | `p` |
| `S` | `s` |
| `Mo` | `mo` |
| `Cu` | `cu` |
| `Ni` | `ni` |
| `Al` | `al` |
| `Nb` | `nb` |
| `V` | `v` |
| `Ti` | `ti` |
| `Ti+Nb+V` | `ti_nb_v` |
| `B` | `b` |
| `Ce+La` | `ce_la` |
| `Fe` | `fe` |
| `Pixel_Size` | `pixel_size` |
| `Heat treatment type` | `heat_treatment_type` |
| `Num_Cycles` | `num_cycles` |
| `Cycle0_Rolling` | `cycle0_rolling` |
| `Cycle0_Rtemp` | `cycle0_rtemp` |
| `Cycle0_ RPercentage` | `cycle0_rpercentage` |
| `Cycle1_HRate (C/min)` | `cycle1_hrate_degc_min` |
| `Cycle1_HoldingTemp (C)` | `cycle1_holdingtemp_degc` |
| `Cycle1_HoldingTime (min)` | `cycle1_holdingtime_min` |
| `Cycle1_CRate (°C/s)` | `cycle1_crate_degc_s` |
| `Cycle1_Rolling` | `cycle1_rolling` |
| `Cycle1_Rtemp` | `cycle1_rtemp` |
| `Cycle1_ RPercentage` | `cycle1_rpercentage` |
| `Cycle1_   Qtype` | `cycle1_qtype` |
| `Cycle1_Qtemp` | `cycle1_qtemp` |
| `Cycle1_Qtime(min)` | `cycle1_qtime_min` |
| `Cycle2_HRate (C/min)` | `cycle2_hrate_degc_min` |
| `Cycle2_HoldingTemp (C)` | `cycle2_holdingtemp_degc` |
| `Cycle2_HoldingTime (min)` | `cycle2_holdingtime_min` |
| `Cycle2_CRate (C/s)` | `cycle2_crate_degc_s` |
| `Cycle2_  rolling` | `cycle2_rolling` |
| `Cycle2_Rtemp (C)` | `cycle2_rtemp_degc` |
| `Cycle2_ RPercentage` | `cycle2_rpercentage` |
| `Cycle2_  Qtype` | `cycle2_qtype` |
| `Cycle2_Qtemp  (C)` | `cycle2_qtemp_degc` |
| `Cycle2_Qtime   (min)` | `cycle2_qtime_min` |
| `Cycle3_HRate (C/min)` | `cycle3_hrate_degc_min` |
| `Cycle3_HoldingTemp   (°C)` | `cycle3_holdingtemp_degc` |
| `Cycle3_HoldingTime     (min)` | `cycle3_holdingtime_min` |
| `Cycle3_CRate   (°C/s)` | `cycle3_crate_degc_s` |
| `Cycle3_rolling` | `cycle3_rolling` |
| `Cycle3_Rtemp` | `cycle3_rtemp` |
| `Cycle3_ RPercentage` | `cycle3_rpercentage` |
| `Cycle3_  Qtype` | `cycle3_qtype` |
| `Cycle3_Qtemp` | `cycle3_qtemp` |
| `Cycle3_Qtime` | `cycle3_qtime` |
| `Cycle4_HRate (C/min)` | `cycle4_hrate_degc_min` |
| `Cycle4_HoldingTemp   (°C)` | `cycle4_holdingtemp_degc` |
| `Cycle4_HoldingTime     (min)` | `cycle4_holdingtime_min` |
| `Cycle4_CRate   (°C/s)` | `cycle4_crate_degc_s` |
| `Cycle4_rolling` | `cycle4_rolling` |
| `Cycle4_Rtemp` | `cycle4_rtemp` |
| `Cycle4_ RPercentage` | `cycle4_rpercentage` |
| `Cycle4_  Qtype` | `cycle4_qtype` |
| `Cycle4_Qtemp` | `cycle4_qtemp` |
| `Cycle4_Qtime` | `cycle4_qtime` |

---

## Collision detection

`build_rename_map()` checks that no two distinct original names produce the
same sanitized name. If a collision is detected it raises immediately:

```
ValueError: Column name collision after sanitization:
  'Cycle2_Rtemp (C)' and 'Cycle2_Rtemp_C' both map to 'cycle2_rtemp_degc'.
  Resolve the ambiguity in the source data before proceeding.
```

This surfaces schema drift early rather than silently merging two columns into
one. The check runs on every load, so a new dataset version with a renamed
column will fail loudly at the sanitization step rather than propagating
incorrect data into training.

---

## Adding a new dataset version

If the source dataset gains new columns or renames existing ones:

1. Load the new CSV and call `build_rename_map(df.columns)` — fix any
   collisions reported
2. Update `CHEMICAL_COLUMNS`, `MICE_COLUMNS`, `INDICATOR_COLUMNS`, and
   `COLUMN_TYPE_OVERRIDES` in `run_training.py`, `main.py`, and the notebook
   to use the new sanitized names
3. Update `TARGET_COLUMNS_TO_EXCLUDE` in the notebook if new target columns
   were added
4. Update the mapping table in this document

The sanitizer itself requires no changes for new columns — the rules are
general-purpose. Only the downstream name references need updating.
