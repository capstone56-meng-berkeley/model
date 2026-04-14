"""Feature preprocessing pipeline orchestrator."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import MissingDataConfig, ScalingConfig, EncodingConfig, PreprocessingConfig
from .base import BaseTypeHandler
from .type_handlers import TypeHandlerRegistry, get_type_handler
from .type_handlers import (
    TYPE_NUMERIC,
    TYPE_CATEGORICAL,
    TYPE_TEXT,
    TYPE_UNIQUE_STRING,
    TYPE_BOOLEAN,
    TYPE_DATETIME,
)


class FeaturePreprocessor:
    """
    Extensible preprocessing pipeline for tabular features.

    Handles:
    - Missing data (column drop, row fill based on thresholds)
    - Type-specific processing via registered handlers
    - Encoding and scaling
    """

    def __init__(self, config: PreprocessingConfig, column_types: Optional[Dict[str, str]] = None):
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing configuration
            column_types: Optional dict of {column_name: type_string} overrides.
                          Bypasses auto-inference for named columns.
                          Valid types: 'numeric', 'categorical', 'text',
                          'unique_string', 'boolean', 'datetime'.
        """
        self.config = config
        self._column_type_overrides: Dict[str, str] = column_types or {}
        self._handlers: Dict[str, BaseTypeHandler] = {}
        self._dropped_columns: List[str] = []
        self._feature_names: List[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> 'FeaturePreprocessor':
        """
        Fit the preprocessing pipeline on training data.

        Args:
            df: DataFrame with training data
            columns: List of columns to process

        Returns:
            self for chaining
        """
        self._handlers = {}
        self._dropped_columns = []
        self._feature_names = []
        columns = df.columns.tolist()

        for col in columns:
            series = df[col]

            # Check missing ratio
            missing_ratio = series.isna().sum() / len(series)

            # Drop column if too many missing
            if missing_ratio > self.config.missing_data.column_drop_threshold:
                print(f"  Dropping column '{col}': {missing_ratio:.1%} missing "
                      f"(threshold: {self.config.missing_data.column_drop_threshold:.0%})")
                self._dropped_columns.append(col)
                continue

            # Resolve column type: explicit override takes priority over auto-inference
            if col in self._column_type_overrides:
                col_type = self._column_type_overrides[col]
                print(f"  {col}: {series.dtype} → {col_type} (override)")
            else:
                col_type = self._infer_column_type(series)
                print(f"  {col}: {series.dtype} → {col_type}")

            # Build handler with appropriate config
            handler = self._create_handler(col, col_type, missing_ratio)

            # Fit handler
            handler.fit(series)
            self._handlers[col] = handler

            # Collect feature names
            self._feature_names.extend(handler.get_feature_names())

        self._fitted = True
        print(f"  Fitted {len(self._handlers)} columns, "
              f"dropped {len(self._dropped_columns)}, "
              f"output features: {len(self._feature_names)}")

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted pipeline.

        Args:
            df: DataFrame to transform

        Returns:
            Numpy array of preprocessed features
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        if len(self._handlers) == 0:
            return np.array([]).reshape(len(df), 0)

        # Transform each column
        transformed_parts = []

        for col, handler in self._handlers.items():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            series = df[col]
            transformed = handler.transform(series)
            transformed_parts.append(transformed)

        # Concatenate all parts
        return np.hstack(transformed_parts)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def _infer_column_type(self, series: pd.Series) -> str:
        """
        Infer column type from pandas dtype.

        Args:
            series: Pandas Series to infer type from

        Returns:
            Inferred type constant (TYPE_NUMERIC, TYPE_CATEGORICAL, etc.)
        """
        dtype = series.dtype
        dtype_name = str(dtype)

        # Boolean
        if pd.api.types.is_bool_dtype(dtype):
            return TYPE_BOOLEAN

        # Numeric types (int8/16/32/64, uint8/16/32/64, float16/32/64)
        if pd.api.types.is_numeric_dtype(dtype):
            return TYPE_NUMERIC

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return TYPE_DATETIME

        # Categorical dtype
        if pd.api.types.is_categorical_dtype(dtype):
            return TYPE_CATEGORICAL

        # Object/string dtype - use heuristics
        if dtype_name in ("object", "string", "str"):
            return self._infer_object_type(series)

        # Default fallback
        return TYPE_CATEGORICAL

    def _infer_object_type(self, series: pd.Series) -> str:
        """
        Infer type for object/string columns using heuristics.

        Args:
            series: Pandas Series with object dtype

        Returns:
            Inferred type constant (TYPE_CATEGORICAL, TYPE_TEXT, or TYPE_UNIQUE_STRING)
        """
        non_null = series.dropna()
        n_unique = series.nunique()
        n_total = len(non_null)

        if n_total == 0:
            return TYPE_CATEGORICAL

        unique_ratio = n_unique / n_total
        avg_length = non_null.astype(str).str.len().mean()

        # High uniqueness (>90% unique values)
        if unique_ratio > 0.9:
            if avg_length > 50:
                return TYPE_TEXT  # Long text content
            return TYPE_UNIQUE_STRING  # IDs, names, etc.

        # Low to medium uniqueness -> categorical
        return TYPE_CATEGORICAL

    def _create_handler(
        self,
        column_name: str,
        column_type: str,
        missing_ratio: float
    ) -> BaseTypeHandler:
        """
        Create a type handler with appropriate configuration.

        Args:
            column_name: Name of the column
            column_type: Type of the column
            missing_ratio: Ratio of missing values

        Returns:
            Configured type handler
        """
        # Determine impute strategy based on missing ratio and config
        if missing_ratio <= self.config.missing_data.row_fill_threshold:
            # Low missing: use configured strategy
            if column_type == TYPE_NUMERIC:
                impute_strategy = self.config.missing_data.numeric_fill_strategy
            else:
                impute_strategy = self.config.missing_data.categorical_fill_strategy
        else:
            # Mid-range missing: use mid_range_strategy
            strategy = self.config.missing_data.mid_range_strategy
            if strategy == "fill":
                if column_type == TYPE_NUMERIC:
                    impute_strategy = self.config.missing_data.numeric_fill_strategy
                else:
                    impute_strategy = self.config.missing_data.categorical_fill_strategy
            elif strategy == "flag":
                # TODO: Add missing indicator feature
                impute_strategy = "constant"
            else:
                impute_strategy = "constant"

        # Build handler kwargs
        kwargs = {
            "column_name": column_name,
            "impute_strategy": impute_strategy,
        }

        # Add type-specific config
        if column_type == TYPE_NUMERIC:
            kwargs["scale_method"] = self.config.scaling.method if self.config.scaling.enabled else "none"
        elif column_type == TYPE_CATEGORICAL:
            kwargs["encode_method"] = self.config.encoding.categorical
            kwargs["max_categories"] = self.config.encoding.max_categories
        elif column_type == TYPE_TEXT:
            kwargs["encode_method"] = self.config.encoding.text
        elif column_type == TYPE_UNIQUE_STRING:
            kwargs["encode_method"] = self.config.encoding.unique_string

        return get_type_handler(column_type, **kwargs)

    def get_feature_names(self) -> List[str]:
        """Get output feature names."""
        return self._feature_names

    def get_dropped_columns(self) -> List[str]:
        """Get list of columns that were dropped due to missing data."""
        return self._dropped_columns

    def get_handler(self, column_name: str) -> Optional[BaseTypeHandler]:
        """Get the handler for a specific column."""
        return self._handlers.get(column_name)


def preprocess_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    config: Optional[PreprocessingConfig] = None
) -> Tuple[np.ndarray, FeaturePreprocessor]:
    """
    Convenience function to preprocess features.

    Column types are automatically inferred from DataFrame dtypes.

    Args:
        df: DataFrame with data
        feature_columns: List of columns to process
        config: Optional preprocessing config

    Returns:
        Tuple of (preprocessed array, fitted preprocessor)
    """
    if config is None:
        config = PreprocessingConfig()

    preprocessor = FeaturePreprocessor(config)
    X = preprocessor.fit_transform(df[feature_columns])

    return X, preprocessor
