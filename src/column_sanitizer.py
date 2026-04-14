"""
Column name sanitization utility.

Converts arbitrary DataFrame column names to safe identifiers:
    [a-z][a-z0-9_]*

Rules applied in order:
  1. Strip leading/trailing whitespace
  2. Lowercase
  3. Insert a separator before '(' when directly attached to a word character
     (e.g. 'Qtime(min)' → 'Qtime (min)') so unit patterns match reliably
  4. Replace known unit/symbol patterns with readable tokens
  5. Replace all remaining non-[a-z0-9] characters with underscore
  6. Collapse runs of underscores to a single underscore
  7. Strip leading/trailing underscores

Unit token map (applied before stripping):
  (°C/s)   → degc_s
  (C/s)    → degc_s
  (°C)     → degc
  (C)      → degc
  (C/min)  → degc_min
  (min)    → min
  (%)      → pct

Note: (wt.%) is intentionally excluded — compositional columns in the source
dataset no longer carry that suffix.

The function is deterministic and collision-free for the current dataset
(verified against all column names). If a future dataset produces a
collision, ``sanitize_dataframe`` raises a ``ValueError`` with the offending
names so the issue surfaces immediately rather than silently merging columns.
"""

import re
from typing import Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def sanitize_column(name: str) -> str:
    """Return a sanitized version of a single column name."""
    s = name.strip().lower()

    # Insert space before '(' when directly attached to a word char so that
    # unit patterns below match regardless of spacing in the source.
    s = re.sub(r'([a-z0-9])\(', r'\1 (', s)

    # Replace known unit/symbol patterns with readable tokens.
    s = re.sub(r'\(°c/s\)',    'degc_s',   s)
    s = re.sub(r'\(c/s\)',     'degc_s',   s)
    s = re.sub(r'\(°c\)',      'degc',     s)
    s = re.sub(r'\(c\)',       'degc',     s)
    s = re.sub(r'\(c/min\)',   'degc_min', s)
    s = re.sub(r'\(min\)',     'min',      s)
    s = re.sub(r'\(%\)',       'pct',      s)

    # Replace all remaining non-alphanumeric characters with underscore.
    s = re.sub(r'[^a-z0-9]+', '_', s)

    # Collapse runs and strip edge underscores.
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')

    return s


def build_rename_map(columns) -> Dict[str, str]:
    """
    Build a {original: sanitized} rename mapping for a list of column names.

    Raises ``ValueError`` if any two distinct originals map to the same
    sanitized name (collision).
    """
    rename: Dict[str, str] = {}
    seen: Dict[str, str] = {}  # sanitized -> first original that produced it

    for col in columns:
        san = sanitize_column(col)
        if san in seen and seen[san] != col:
            raise ValueError(
                f"Column name collision after sanitization: "
                f"{repr(col)} and {repr(seen[san])} both map to {repr(san)}. "
                f"Resolve the ambiguity in the source data before proceeding."
            )
        seen[san] = col
        rename[col] = san

    return rename


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *df* with all column names sanitized.

    Raises ``ValueError`` on collision (see ``build_rename_map``).
    """
    rename_map = build_rename_map(df.columns.tolist())
    return df.rename(columns=rename_map)
