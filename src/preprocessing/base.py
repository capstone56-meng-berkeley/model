"""Abstract base classes for preprocessing components."""

from abc import ABC, abstractmethod
from typing import List, Any, Optional

import numpy as np
import pandas as pd


class BaseImputer(ABC):
    """
    Base class for missing value imputers.

    Subclasses must implement fit() and transform().
    """

    def __init__(self, **kwargs):
        """Initialize imputer with optional parameters."""
        self._fitted = False
        self._fit_value: Any = None

    @abstractmethod
    def fit(self, series: pd.Series) -> 'BaseImputer':
        """
        Fit the imputer on training data.

        Args:
            series: Pandas Series to fit on

        Returns:
            self for chaining
        """
        pass

    @abstractmethod
    def transform(self, series: pd.Series) -> pd.Series:
        """
        Transform data using fitted imputer.

        Args:
            series: Pandas Series to transform

        Returns:
            Transformed Series with missing values filled
        """
        pass

    def fit_transform(self, series: pd.Series) -> pd.Series:
        """Fit and transform in one step."""
        return self.fit(series).transform(series)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Imputer not fitted. Call fit() first.")


class BaseEncoder(ABC):
    """
    Base class for feature encoders.

    Subclasses must implement fit(), transform(), and get_feature_names().
    """

    def __init__(self, **kwargs):
        """Initialize encoder with optional parameters."""
        self._fitted = False
        self._feature_names: List[str] = []

    @abstractmethod
    def fit(self, series: pd.Series) -> 'BaseEncoder':
        """
        Fit the encoder on training data.

        Args:
            series: Pandas Series to fit on

        Returns:
            self for chaining
        """
        pass

    @abstractmethod
    def transform(self, series: pd.Series) -> np.ndarray:
        """
        Transform data using fitted encoder.

        Args:
            series: Pandas Series to transform

        Returns:
            Encoded numpy array (may have multiple columns)
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get output feature names after encoding.

        Returns:
            List of feature names
        """
        pass

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(series).transform(series)


class BaseScaler(ABC):
    """
    Base class for feature scalers.

    Subclasses must implement fit() and transform().
    """

    def __init__(self, **kwargs):
        """Initialize scaler with optional parameters."""
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseScaler':
        """
        Fit the scaler on training data.

        Args:
            X: Numpy array to fit on (n_samples, n_features)

        Returns:
            self for chaining
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Numpy array to transform

        Returns:
            Scaled array
        """
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data.

        Args:
            X: Scaled array

        Returns:
            Original scale array
        """
        raise NotImplementedError("inverse_transform not implemented")


class BaseTypeHandler(ABC):
    """
    Base class for column type handlers.

    Handles the full preprocessing pipeline for a specific data type.
    """

    def __init__(
        self,
        column_name: str,
        imputer: Optional[BaseImputer] = None,
        encoder: Optional[BaseEncoder] = None,
        scaler: Optional[BaseScaler] = None,
        **kwargs
    ):
        """
        Initialize type handler.

        Args:
            column_name: Name of the column being handled
            imputer: Optional imputer for missing values
            encoder: Optional encoder for transformation
            scaler: Optional scaler for normalization
        """
        self.column_name = column_name
        self.imputer = imputer
        self.encoder = encoder
        self.scaler = scaler
        self._fitted = False
        self._feature_names: List[str] = []

    @abstractmethod
    def fit(self, series: pd.Series) -> 'BaseTypeHandler':
        """
        Fit all components on training data.

        Args:
            series: Pandas Series to fit on

        Returns:
            self for chaining
        """
        pass

    @abstractmethod
    def transform(self, series: pd.Series) -> np.ndarray:
        """
        Transform data using fitted components.

        Args:
            series: Pandas Series to transform

        Returns:
            Transformed numpy array
        """
        pass

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(series).transform(series)

    def get_feature_names(self) -> List[str]:
        """Get output feature names."""
        return self._feature_names

    @classmethod
    def get_default_imputer(cls, strategy: str) -> Optional['BaseImputer']:
        """Get default imputer for this type."""
        return None

    @classmethod
    def get_default_encoder(cls, method: str) -> Optional['BaseEncoder']:
        """Get default encoder for this type."""
        return None

    @classmethod
    def get_default_scaler(cls, method: str) -> Optional['BaseScaler']:
        """Get default scaler for this type."""
        return None
