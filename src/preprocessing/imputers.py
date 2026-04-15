"""Imputer implementations for missing value handling."""

from typing import Any, List

import pandas as pd

from ..registry import Registry
from .base import BaseImputer


class ImputerRegistry(Registry):
    """Registry for missing value imputation strategies."""
    _registry = {}


@ImputerRegistry.register("mean")
class MeanImputer(BaseImputer):
    """Fill missing values with mean of the column."""

    def fit(self, series: pd.Series) -> 'MeanImputer':
        self._fit_value = series.mean()
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        self._check_fitted()
        return series.fillna(self._fit_value)


@ImputerRegistry.register("median")
class MedianImputer(BaseImputer):
    """Fill missing values with median of the column."""

    def fit(self, series: pd.Series) -> 'MedianImputer':
        self._fit_value = series.median()
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        self._check_fitted()
        return series.fillna(self._fit_value)


@ImputerRegistry.register("mode")
class ModeImputer(BaseImputer):
    """Fill missing values with mode (most frequent) of the column."""

    def fit(self, series: pd.Series) -> 'ModeImputer':
        mode_values = series.mode()
        self._fit_value = mode_values.iloc[0] if len(mode_values) > 0 else None
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        self._check_fitted()
        if self._fit_value is not None:
            return series.fillna(self._fit_value)
        return series


@ImputerRegistry.register("constant")
class ConstantImputer(BaseImputer):
    """Fill missing values with a constant value."""

    def __init__(self, value: Any = 0, **kwargs):
        super().__init__(**kwargs)
        self._fit_value = value

    def fit(self, series: pd.Series) -> 'ConstantImputer':
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        self._check_fitted()
        return series.fillna(self._fit_value)


@ImputerRegistry.register("zero")
class ZeroImputer(ConstantImputer):
    """Fill missing values with zero. Alias for ConstantImputer(value=0)."""

    def __init__(self, **kwargs):
        super().__init__(value=0, **kwargs)


@ImputerRegistry.register("forward_fill")
class ForwardFillImputer(BaseImputer):
    """Fill missing values with previous valid value."""

    def fit(self, series: pd.Series) -> 'ForwardFillImputer':
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        self._check_fitted()
        return series.ffill()


@ImputerRegistry.register("backward_fill")
class BackwardFillImputer(BaseImputer):
    """Fill missing values with next valid value."""

    def fit(self, series: pd.Series) -> 'BackwardFillImputer':
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        self._check_fitted()
        return series.bfill()


@ImputerRegistry.register("unknown")
class UnknownImputer(ConstantImputer):
    """Fill missing categorical values with 'Unknown'. Alias for ConstantImputer(value='Unknown')."""

    def __init__(self, unknown_value: str = "Unknown", **kwargs):
        super().__init__(value=unknown_value, **kwargs)


@ImputerRegistry.register("interpolate")
class InterpolateImputer(BaseImputer):
    """Fill missing values using interpolation."""

    def __init__(self, method: str = "linear", **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def fit(self, series: pd.Series) -> 'InterpolateImputer':
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        self._check_fitted()
        return series.interpolate(method=self.method)


@ImputerRegistry.register("mice")
class MICEImputer(BaseImputer):
    """
    Multiple Imputation by Chained Equations (MICE).

    Wraps sklearn's IterativeImputer. Unlike other imputers this operates on
    a full numeric DataFrame, not a single Series.  Direct per-column use via
    fit(series)/transform(series) raises NotImplementedError — wire it through
    FeaturePreprocessor(mice_columns=[...]) which calls fit_df/transform_df.
    """

    def __init__(self, max_iter: int = 10, random_state: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self.random_state = random_state
        self._imputer = None
        self._columns: List[str] = []

    # ------------------------------------------------------------------
    # Per-series API — intentionally unsupported
    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> 'MICEImputer':
        raise NotImplementedError(
            "MICEImputer requires the full DataFrame. "
            "Use FeaturePreprocessor(mice_columns=[...]) instead."
        )

    def transform(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError(
            "MICEImputer requires the full DataFrame. "
            "Use FeaturePreprocessor(mice_columns=[...]) instead."
        )

    # ------------------------------------------------------------------
    # DataFrame API — called by FeaturePreprocessor
    # ------------------------------------------------------------------
    def fit_df(self, df: pd.DataFrame, columns: List[str]) -> 'MICEImputer':
        """
        Fit the iterative imputer on the given columns of df.

        Args:
            df: Full feature DataFrame (only numeric columns used)
            columns: Columns to impute via MICE
        """
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        self._columns = columns
        numeric_df = df[columns].apply(pd.to_numeric, errors='coerce')
        self._imputer = IterativeImputer(
            max_iter=self.max_iter,
            random_state=self.random_state,
            skip_complete=True,
        )
        self._imputer.fit(numeric_df)
        self._fitted = True
        return self

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the MICE columns and return an updated DataFrame.

        Args:
            df: DataFrame containing at least self._columns

        Returns:
            Copy of df with MICE columns filled
        """
        self._check_fitted()
        numeric_df = df[self._columns].apply(pd.to_numeric, errors='coerce')
        imputed = self._imputer.transform(numeric_df)
        df = df.copy()
        df[self._columns] = imputed
        return df


def get_imputer(strategy: str, **kwargs) -> BaseImputer:
    """Factory function to get an imputer by strategy name."""
    return ImputerRegistry.create(strategy, **kwargs)
