"""Type handler implementations for different column data types."""

import warnings
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from ..registry import Registry
from .base import BaseTypeHandler, BaseImputer, BaseEncoder, BaseScaler
from .imputers import ImputerRegistry
from .encoders import EncoderRegistry
from .scalers import ScalerRegistry

TYPE_NUMERIC = "numeric"        # int8/16/32/64, uint8/16/32/64, float16/32/64
TYPE_CATEGORICAL = "categorical"  # category dtype or object with few unique values
TYPE_TEXT = "text"              # object with high uniqueness and long avg length
TYPE_UNIQUE_STRING = "unique_string"  # object with high uniqueness, short values (IDs, names)
TYPE_BOOLEAN = "boolean"        # bool dtype
TYPE_DATETIME = "datetime"      # datetime64


class TypeHandlerRegistry(Registry):
    """Registry for column type handlers."""
    _registry = {}


@TypeHandlerRegistry.register(TYPE_NUMERIC)
class NumericHandler(BaseTypeHandler):
    """Handler for numeric (continuous/discrete) columns."""

    def __init__(
        self,
        column_name: str,
        imputer: Optional[BaseImputer] = None,
        encoder: Optional[BaseEncoder] = None,
        scaler: Optional[BaseScaler] = None,
        impute_strategy: str = "mean",
        scale_method: str = "standard",
        **kwargs
    ):
        """
        Initialize numeric handler.

        Args:
            column_name: Name of the column
            imputer: Custom imputer (or use default based on strategy)
            encoder: Custom encoder (default: passthrough)
            scaler: Custom scaler (or use default based on method)
            impute_strategy: Imputation strategy if no imputer provided
            scale_method: Scaling method if no scaler provided
        """
        # Set defaults if not provided
        if imputer is None:
            imputer = ImputerRegistry.create(impute_strategy)
        if encoder is None:
            encoder = EncoderRegistry.create("passthrough", column_name=column_name)
        if scaler is None:
            scaler = ScalerRegistry.create(scale_method)

        super().__init__(column_name, imputer, encoder, scaler, **kwargs)

    def fit(self, series: pd.Series) -> 'NumericHandler':
        # Convert to numeric
        numeric_series = pd.to_numeric(series, errors='coerce')

        # Fit imputer
        if self.imputer:
            self.imputer.fit(numeric_series)

        # Apply imputation for encoder fitting
        imputed = self.imputer.transform(numeric_series) if self.imputer else numeric_series

        # Fit encoder
        if self.encoder:
            self.encoder.fit(imputed)

        # Get encoded for scaler fitting
        encoded = self.encoder.transform(imputed) if self.encoder else imputed.values.reshape(-1, 1)

        # Fit scaler
        if self.scaler:
            self.scaler.fit(encoded)

        self._feature_names = self.encoder.get_feature_names() if self.encoder else [self.column_name]
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Handler not fitted. Call fit() first.")

        # Convert to numeric
        numeric_series = pd.to_numeric(series, errors='coerce')

        # Impute
        imputed = self.imputer.transform(numeric_series) if self.imputer else numeric_series

        # Encode
        encoded = self.encoder.transform(imputed) if self.encoder else imputed.values.reshape(-1, 1)

        # Scale
        scaled = self.scaler.transform(encoded) if self.scaler else encoded

        return scaled


@TypeHandlerRegistry.register(TYPE_CATEGORICAL)
class CategoricalHandler(BaseTypeHandler):
    """Handler for categorical columns."""

    def __init__(
        self,
        column_name: str,
        imputer: Optional[BaseImputer] = None,
        encoder: Optional[BaseEncoder] = None,
        scaler: Optional[BaseScaler] = None,
        impute_strategy: str = "mode",
        encode_method: str = "onehot",
        max_categories: int = 50,
        **kwargs
    ):
        """
        Initialize categorical handler.

        Args:
            column_name: Name of the column
            imputer: Custom imputer (default: mode)
            encoder: Custom encoder (default: onehot)
            scaler: Custom scaler (default: none for categorical)
            impute_strategy: Imputation strategy
            encode_method: Encoding method ('onehot' or 'label')
            max_categories: Max categories before switching to label encoding
        """
        if imputer is None:
            imputer = ImputerRegistry.create(impute_strategy)
        if encoder is None:
            encoder = EncoderRegistry.create(
                encode_method,
                column_name=column_name,
                max_categories=max_categories
            )
        if scaler is None:
            scaler = ScalerRegistry.create("none")  # No scaling for categorical

        super().__init__(column_name, imputer, encoder, scaler, **kwargs)

    @staticmethod
    def _to_str_series(series: pd.Series) -> pd.Series:
        """Convert to string while preserving NaN as NaN (not the string 'nan')."""
        return series.where(series.isna(), series.astype(str))

    def fit(self, series: pd.Series) -> 'CategoricalHandler':
        str_series = self._to_str_series(series)

        # Fit imputer
        if self.imputer:
            self.imputer.fit(str_series)

        # Apply imputation
        imputed = self.imputer.transform(str_series) if self.imputer else str_series

        # Fit encoder
        if self.encoder:
            self.encoder.fit(imputed)

        self._feature_names = self.encoder.get_feature_names() if self.encoder else [self.column_name]
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Handler not fitted. Call fit() first.")

        str_series = self._to_str_series(series)

        # Impute
        imputed = self.imputer.transform(str_series) if self.imputer else str_series

        # Encode
        encoded = self.encoder.transform(imputed) if self.encoder else imputed.values.reshape(-1, 1)

        return encoded


@TypeHandlerRegistry.register(TYPE_TEXT)
class TextHandler(BaseTypeHandler):
    """Handler for free-form text columns."""

    def __init__(
        self,
        column_name: str,
        imputer: Optional[BaseImputer] = None,
        encoder: Optional[BaseEncoder] = None,
        scaler: Optional[BaseScaler] = None,
        encode_method: str = "tfidf",
        max_features: int = 100,
        **kwargs
    ):
        """
        Initialize text handler.

        Args:
            column_name: Name of the column
            imputer: Custom imputer (default: empty string)
            encoder: Custom encoder (default: tfidf)
            scaler: Custom scaler (default: none)
            encode_method: Encoding method ('tfidf' or 'skip')
            max_features: Max TF-IDF features
        """
        if imputer is None:
            imputer = ImputerRegistry.create("constant", value="")
        if encoder is None and encode_method != "skip":
            encoder = EncoderRegistry.create(
                encode_method,
                column_name=column_name,
                max_features=max_features
            )
        if scaler is None:
            scaler = ScalerRegistry.create("none")

        super().__init__(column_name, imputer, encoder, scaler, **kwargs)
        self._skip = encode_method == "skip"

    def fit(self, series: pd.Series) -> 'TextHandler':
        if self._skip:
            self._feature_names = []
            self._fitted = True
            return self

        # Convert to string
        str_series = series.fillna("").astype(str)

        # Fit encoder (imputation is just empty string)
        if self.encoder:
            self.encoder.fit(str_series)

        self._feature_names = self.encoder.get_feature_names() if self.encoder else []
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Handler not fitted. Call fit() first.")

        if self._skip:
            return np.array([]).reshape(len(series), 0)

        # Convert to string
        str_series = series.fillna("").astype(str)

        # Encode
        encoded = self.encoder.transform(str_series) if self.encoder else np.array([])

        return encoded


@TypeHandlerRegistry.register(TYPE_UNIQUE_STRING)
class UniqueStringHandler(BaseTypeHandler):
    """Handler for unique identifier columns (sample IDs, batch numbers)."""

    def __init__(
        self,
        column_name: str,
        imputer: Optional[BaseImputer] = None,
        encoder: Optional[BaseEncoder] = None,
        scaler: Optional[BaseScaler] = None,
        encode_method: str = "skip",
        **kwargs
    ):
        """
        Initialize unique string handler.

        Args:
            column_name: Name of the column
            encode_method: 'label' or 'skip' (default: 'skip' — IDs carry no ordinal signal)
        """
        if imputer is None:
            imputer = ImputerRegistry.create("unknown")
        if encoder is None and encode_method != "skip":
            encoder = EncoderRegistry.create(encode_method, column_name=column_name)
        if scaler is None:
            scaler = ScalerRegistry.create("none")

        super().__init__(column_name, imputer, encoder, scaler, **kwargs)
        self._skip = encode_method == "skip"

    def fit(self, series: pd.Series) -> 'UniqueStringHandler':
        if self._skip:
            self._feature_names = []
            self._fitted = True
            return self

        str_series = series.where(series.isna(), series.astype(str))

        if self.imputer:
            self.imputer.fit(str_series)

        imputed = self.imputer.transform(str_series) if self.imputer else str_series

        if self.encoder:
            self.encoder.fit(imputed)

        self._feature_names = self.encoder.get_feature_names() if self.encoder else []
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Handler not fitted. Call fit() first.")

        if self._skip:
            return np.array([]).reshape(len(series), 0)

        str_series = series.where(series.isna(), series.astype(str))

        imputed = self.imputer.transform(str_series) if self.imputer else str_series
        encoded = self.encoder.transform(imputed) if self.encoder else np.array([])

        return encoded


@TypeHandlerRegistry.register(TYPE_BOOLEAN)
class BooleanHandler(BaseTypeHandler):
    """Handler for boolean columns."""

    _BOOL_MAP: Dict[Any, int] = {
        True: 1, False: 0,
        'True': 1, 'False': 0,
        'true': 1, 'false': 0,
        '1': 1, '0': 0,
        1: 1, 0: 0,
    }

    def __init__(
        self,
        column_name: str,
        imputer: Optional[BaseImputer] = None,
        encoder: Optional[BaseEncoder] = None,
        scaler: Optional[BaseScaler] = None,
        impute_strategy: str = "mode",
        **kwargs
    ):
        if imputer is None:
            imputer = ImputerRegistry.create(impute_strategy)
        if encoder is None:
            encoder = EncoderRegistry.create("passthrough", column_name=column_name)
        if scaler is None:
            scaler = ScalerRegistry.create("none")

        super().__init__(column_name, imputer, encoder, scaler, **kwargs)

    def _map_bool(self, series: pd.Series) -> pd.Series:
        mapped = series.map(self._BOOL_MAP)
        unmapped = series.notna() & mapped.isna()
        if unmapped.any():
            bad_vals = series[unmapped].unique().tolist()
            warnings.warn(
                f"BooleanHandler '{self.column_name}': {unmapped.sum()} value(s) not in "
                f"bool map will be treated as missing: {bad_vals}"
            )
        return mapped.astype(float)

    def fit(self, series: pd.Series) -> 'BooleanHandler':
        bool_series = self._map_bool(series)

        if self.imputer:
            self.imputer.fit(bool_series)

        imputed = self.imputer.transform(bool_series) if self.imputer else bool_series

        if self.encoder:
            self.encoder.fit(imputed)

        self._feature_names = self.encoder.get_feature_names() if self.encoder else [self.column_name]
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Handler not fitted. Call fit() first.")

        bool_series = self._map_bool(series)

        imputed = self.imputer.transform(bool_series) if self.imputer else bool_series
        encoded = self.encoder.transform(imputed) if self.encoder else imputed.values.reshape(-1, 1)

        return encoded


@TypeHandlerRegistry.register(TYPE_DATETIME)
class DatetimeHandler(BaseTypeHandler):
    """Handler for datetime columns — extracts year, month, day, dayofweek as numeric features."""

    def __init__(
        self,
        column_name: str,
        imputer=None,
        encoder=None,
        scaler=None,
        **kwargs
    ):
        # Forward-fill is the most sensible default for a missing timestamp:
        # use the surrounding temporal context rather than a sentinel value.
        if imputer is None:
            imputer = ImputerRegistry.create("forward_fill")
        if scaler is None:
            scaler = ScalerRegistry.create("none")
        super().__init__(column_name, imputer, encoder, scaler, **kwargs)

    def fit(self, series: pd.Series) -> 'DatetimeHandler':
        dt = pd.to_datetime(series, errors='coerce')
        if self.imputer:
            self.imputer.fit(dt)
        self._feature_names = [
            f"{self.column_name}_year",
            f"{self.column_name}_month",
            f"{self.column_name}_day",
            f"{self.column_name}_dayofweek",
        ]
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Handler not fitted. Call fit() first.")

        dt = pd.to_datetime(series, errors='coerce')
        if self.imputer:
            dt = self.imputer.transform(dt)
        return np.column_stack([
            dt.dt.year.astype(np.float64),
            dt.dt.month.astype(np.float64),
            dt.dt.day.astype(np.float64),
            dt.dt.dayofweek.astype(np.float64),
        ])


def get_type_handler(
    column_type: str,
    column_name: str,
    **kwargs
) -> BaseTypeHandler:
    """
    Factory function to get a type handler.

    Args:
        column_type: Type of the column ('numeric', 'categorical', etc.)
        column_name: Name of the column
        **kwargs: Additional arguments for the handler

    Returns:
        Type handler instance
    """
    return TypeHandlerRegistry.create(column_type, column_name=column_name, **kwargs)
