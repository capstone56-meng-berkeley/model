"""Feature preprocessing pipeline orchestrator."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .base import BaseTypeHandler
from .type_handlers import TypeHandlerRegistry, get_type_handler


@dataclass
class MissingDataConfig:
    """Configuration for missing data handling."""
    column_drop_threshold: float = 0.95  # Drop column if >95% missing
    row_fill_threshold: float = 0.10     # Fill if <10% missing
    numeric_fill_strategy: str = "mean"  # mean, median, zero
    categorical_fill_strategy: str = "mode"  # mode, unknown
    mid_range_strategy: str = "fill"  # drop_rows, fill, flag


@dataclass
class ScalingConfig:
    """Configuration for feature scaling."""
    method: str = "standard"  # standard, minmax, robust, none
    enabled: bool = True


@dataclass
class EncodingConfig:
    """Configuration for feature encoding."""
    categorical: str = "onehot"  # onehot, label
    text: str = "tfidf"  # tfidf, skip
    unique_string: str = "label"  # label, skip
    max_categories: int = 50


@dataclass
class PreprocessingConfig:
    """Full preprocessing configuration."""
    column_types: Dict[str, str] = field(default_factory=dict)
    missing_data: MissingDataConfig = field(default_factory=MissingDataConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)


class FeaturePreprocessor:
    """
    Extensible preprocessing pipeline for tabular features.

    Handles:
    - Missing data (column drop, row fill based on thresholds)
    - Type-specific processing via registered handlers
    - Encoding and scaling
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self._handlers: Dict[str, BaseTypeHandler] = {}
        self._dropped_columns: List[str] = []
        self._feature_names: List[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'FeaturePreprocessor':
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

        for col in columns:
            if col not in df.columns:
                print(f"  Warning: Column '{col}' not found in DataFrame, skipping")
                continue

            series = df[col]

            # Check missing ratio
            missing_ratio = series.isna().sum() / len(series)

            # Drop column if too many missing
            if missing_ratio > self.config.missing_data.column_drop_threshold:
                print(f"  Dropping column '{col}': {missing_ratio:.1%} missing "
                      f"(threshold: {self.config.missing_data.column_drop_threshold:.0%})")
                self._dropped_columns.append(col)
                continue

            # Get column type
            col_type = self.config.column_types.get(col, "numeric")

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

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df, columns).transform(df)

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
            if column_type == "numeric":
                impute_strategy = self.config.missing_data.numeric_fill_strategy
            else:
                impute_strategy = self.config.missing_data.categorical_fill_strategy
        else:
            # Mid-range missing: use mid_range_strategy
            strategy = self.config.missing_data.mid_range_strategy
            if strategy == "fill":
                if column_type == "numeric":
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
        if column_type == "numeric":
            kwargs["scale_method"] = self.config.scaling.method if self.config.scaling.enabled else "none"
        elif column_type == "categorical":
            kwargs["encode_method"] = self.config.encoding.categorical
            kwargs["max_categories"] = self.config.encoding.max_categories
        elif column_type == "text":
            kwargs["encode_method"] = self.config.encoding.text
        elif column_type == "unique_string":
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
    column_types: Dict[str, str],
    config: Optional[PreprocessingConfig] = None
) -> Tuple[np.ndarray, FeaturePreprocessor]:
    """
    Convenience function to preprocess features.

    Args:
        df: DataFrame with data
        feature_columns: List of columns to process
        column_types: Mapping of column name to type
        config: Optional preprocessing config

    Returns:
        Tuple of (preprocessed array, fitted preprocessor)
    """
    if config is None:
        config = PreprocessingConfig(column_types=column_types)
    else:
        config.column_types = column_types

    preprocessor = FeaturePreprocessor(config)
    X = preprocessor.fit_transform(df, feature_columns)

    return X, preprocessor
