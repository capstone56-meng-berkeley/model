"""Scaler implementations for feature normalization."""

from typing import Optional

import numpy as np

from ..registry import Registry
from .base import BaseScaler


class ScalerRegistry(Registry):
    """Registry for scaling strategies."""
    _registry = {}


@ScalerRegistry.register("none")
class NoScaler(BaseScaler):
    """Pass-through scaler (no scaling applied)."""

    def fit(self, X: np.ndarray) -> 'NoScaler':
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X


@ScalerRegistry.register("standard")
class StandardScaler(BaseScaler):
    """Standardize features by removing mean and scaling to unit variance."""

    def __init__(self, with_mean: bool = True, with_std: bool = True, **kwargs):
        """
        Initialize standard scaler.

        Args:
            with_mean: If True, center data by removing mean
            with_std: If True, scale data to unit variance
        """
        super().__init__(**kwargs)
        self.with_mean = with_mean
        self.with_std = with_std
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.with_mean:
            self._mean = np.nanmean(X, axis=0)
        else:
            self._mean = np.zeros(X.shape[1])

        if self.with_std:
            self._std = np.nanstd(X, axis=0)
            # Avoid division by zero
            self._std[self._std == 0] = 1.0
        else:
            self._std = np.ones(X.shape[1])

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return (X - self._mean) / self._std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X * self._std + self._mean


@ScalerRegistry.register("minmax")
class MinMaxScaler(BaseScaler):
    """Scale features to a given range (default 0-1)."""

    def __init__(
        self,
        feature_range: tuple = (0, 1),
        **kwargs
    ):
        """
        Initialize min-max scaler.

        Args:
            feature_range: Desired range of transformed data (min, max)
        """
        super().__init__(**kwargs)
        self.feature_range = feature_range
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._min = np.nanmin(X, axis=0)
        self._max = np.nanmax(X, axis=0)

        # Calculate scale
        data_range = self._max - self._min
        data_range[data_range == 0] = 1.0  # Avoid division by zero

        feature_min, feature_max = self.feature_range
        self._scale = (feature_max - feature_min) / data_range

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        feature_min = self.feature_range[0]
        X_scaled = (X - self._min) * self._scale + feature_min

        return X_scaled

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        feature_min = self.feature_range[0]
        return (X - feature_min) / self._scale + self._min


@ScalerRegistry.register("robust")
class RobustScaler(BaseScaler):
    """Scale features using statistics robust to outliers (median and IQR)."""

    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple = (25.0, 75.0),
        **kwargs
    ):
        """
        Initialize robust scaler.

        Args:
            with_centering: If True, center data using median
            with_scaling: If True, scale data using IQR
            quantile_range: Quantile range for computing IQR
        """
        super().__init__(**kwargs)
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self._median: Optional[np.ndarray] = None
        self._iqr: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'RobustScaler':
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.with_centering:
            self._median = np.nanmedian(X, axis=0)
        else:
            self._median = np.zeros(X.shape[1])

        if self.with_scaling:
            q_low, q_high = self.quantile_range
            percentiles = np.nanpercentile(X, [q_low, q_high], axis=0)
            self._iqr = percentiles[1] - percentiles[0]
            # Avoid division by zero
            self._iqr[self._iqr == 0] = 1.0
        else:
            self._iqr = np.ones(X.shape[1])

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return (X - self._median) / self._iqr

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X * self._iqr + self._median


@ScalerRegistry.register("maxabs")
class MaxAbsScaler(BaseScaler):
    """Scale features by their maximum absolute value."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_abs: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'MaxAbsScaler':
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._max_abs = np.nanmax(np.abs(X), axis=0)
        # Avoid division by zero
        self._max_abs[self._max_abs == 0] = 1.0

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X / self._max_abs

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X * self._max_abs


def get_scaler(method: str, **kwargs) -> BaseScaler:
    """
    Factory function to get a scaler by method name.

    Args:
        method: Scaler method name
        **kwargs: Additional arguments for the scaler

    Returns:
        Scaler instance
    """
    return ScalerRegistry.create(method, **kwargs)
