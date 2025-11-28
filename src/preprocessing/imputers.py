"""Imputer implementations for missing value handling."""

from typing import Any, Optional

import numpy as np
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
        if not self._fitted:
            raise RuntimeError("Imputer not fitted. Call fit() first.")
        return series.fillna(self._fit_value)


@ImputerRegistry.register("median")
class MedianImputer(BaseImputer):
    """Fill missing values with median of the column."""

    def fit(self, series: pd.Series) -> 'MedianImputer':
        self._fit_value = series.median()
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Imputer not fitted. Call fit() first.")
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
        if not self._fitted:
            raise RuntimeError("Imputer not fitted. Call fit() first.")
        if self._fit_value is not None:
            return series.fillna(self._fit_value)
        return series


@ImputerRegistry.register("constant")
class ConstantImputer(BaseImputer):
    """Fill missing values with a constant value."""

    def __init__(self, value: Any = 0, **kwargs):
        """
        Initialize constant imputer.

        Args:
            value: Constant value to fill missing values with
        """
        super().__init__(**kwargs)
        self._fit_value = value

    def fit(self, series: pd.Series) -> 'ConstantImputer':
        # No fitting needed for constant
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        return series.fillna(self._fit_value)


@ImputerRegistry.register("zero")
class ZeroImputer(BaseImputer):
    """Fill missing values with zero."""

    def fit(self, series: pd.Series) -> 'ZeroImputer':
        self._fit_value = 0
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        return series.fillna(0)


@ImputerRegistry.register("forward_fill")
class ForwardFillImputer(BaseImputer):
    """Fill missing values with previous valid value."""

    def fit(self, series: pd.Series) -> 'ForwardFillImputer':
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        return series.ffill()


@ImputerRegistry.register("backward_fill")
class BackwardFillImputer(BaseImputer):
    """Fill missing values with next valid value."""

    def fit(self, series: pd.Series) -> 'BackwardFillImputer':
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        return series.bfill()


@ImputerRegistry.register("unknown")
class UnknownImputer(BaseImputer):
    """Fill missing categorical values with 'Unknown' string."""

    def __init__(self, unknown_value: str = "Unknown", **kwargs):
        super().__init__(**kwargs)
        self._fit_value = unknown_value

    def fit(self, series: pd.Series) -> 'UnknownImputer':
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        return series.fillna(self._fit_value)


@ImputerRegistry.register("interpolate")
class InterpolateImputer(BaseImputer):
    """Fill missing values using interpolation."""

    def __init__(self, method: str = "linear", **kwargs):
        """
        Initialize interpolation imputer.

        Args:
            method: Interpolation method ('linear', 'quadratic', 'cubic', etc.)
        """
        super().__init__(**kwargs)
        self.method = method

    def fit(self, series: pd.Series) -> 'InterpolateImputer':
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        return series.interpolate(method=self.method)


def get_imputer(strategy: str, **kwargs) -> BaseImputer:
    """
    Factory function to get an imputer by strategy name.

    Args:
        strategy: Imputer strategy name
        **kwargs: Additional arguments for the imputer

    Returns:
        Imputer instance
    """
    return ImputerRegistry.create(strategy, **kwargs)
