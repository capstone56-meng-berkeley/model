"""Encoder implementations for feature transformation."""

from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from ..registry import Registry
from .base import BaseEncoder


class EncoderRegistry(Registry):
    """Registry for encoding strategies."""
    _registry = {}


@EncoderRegistry.register("passthrough")
class PassthroughEncoder(BaseEncoder):
    """Pass numeric values through without encoding."""

    def __init__(self, column_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self.column_name = column_name

    def fit(self, series: pd.Series) -> 'PassthroughEncoder':
        self._feature_names = [self.column_name or series.name or "feature"]
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        return series.values.reshape(-1, 1).astype(np.float64)

    def get_feature_names(self) -> List[str]:
        return self._feature_names


@EncoderRegistry.register("onehot")
class OneHotEncoder(BaseEncoder):
    """One-hot encode categorical values."""

    def __init__(
        self,
        column_name: str = "",
        handle_unknown: str = "ignore",
        max_categories: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize one-hot encoder.

        Args:
            column_name: Name of the column being encoded
            handle_unknown: How to handle unknown categories ('ignore' or 'error')
            max_categories: Maximum number of categories (switch to label if exceeded)
        """
        super().__init__(**kwargs)
        self.column_name = column_name
        self.handle_unknown = handle_unknown
        self.max_categories = max_categories
        self._categories: List[str] = []

    def fit(self, series: pd.Series) -> 'OneHotEncoder':
        unique_values = series.dropna().unique()
        self._categories = sorted([str(v) for v in unique_values])

        # Limit categories if specified
        if self.max_categories and len(self._categories) > self.max_categories:
            self._categories = self._categories[:self.max_categories]

        prefix = self.column_name or series.name or "cat"
        self._feature_names = [f"{prefix}_{cat}" for cat in self._categories]
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        # Create one-hot matrix
        n_samples = len(series)
        n_categories = len(self._categories)
        result = np.zeros((n_samples, n_categories), dtype=np.float64)

        for i, val in enumerate(series):
            if pd.isna(val):
                continue
            str_val = str(val)
            if str_val in self._categories:
                cat_idx = self._categories.index(str_val)
                result[i, cat_idx] = 1.0
            elif self.handle_unknown == "error":
                raise ValueError(f"Unknown category: {str_val}")
            # else: ignore - row stays all zeros

        return result

    def get_feature_names(self) -> List[str]:
        return self._feature_names


@EncoderRegistry.register("label")
class LabelEncoder(BaseEncoder):
    """Encode categorical values as integers."""

    def __init__(self, column_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self.column_name = column_name
        self._mapping: Dict[str, int] = {}
        self._unknown_value: int = -1

    def fit(self, series: pd.Series) -> 'LabelEncoder':
        unique_values = series.dropna().unique()
        sorted_values = sorted([str(v) for v in unique_values])
        self._mapping = {v: i for i, v in enumerate(sorted_values)}
        self._unknown_value = len(self._mapping)

        name = self.column_name or series.name or "label"
        self._feature_names = [name]
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        result = np.zeros(len(series), dtype=np.float64)
        for i, val in enumerate(series):
            if pd.isna(val):
                result[i] = self._unknown_value
            else:
                str_val = str(val)
                result[i] = self._mapping.get(str_val, self._unknown_value)

        return result.reshape(-1, 1)

    def get_feature_names(self) -> List[str]:
        return self._feature_names

    def inverse_transform(self, encoded: np.ndarray) -> List[str]:
        """Convert encoded values back to original labels."""
        inverse_mapping = {v: k for k, v in self._mapping.items()}
        result = []
        for val in encoded.flatten():
            int_val = int(val)
            result.append(inverse_mapping.get(int_val, "unknown"))
        return result


@EncoderRegistry.register("ordinal")
class OrdinalEncoder(BaseEncoder):
    """Encode categorical values with explicit ordering."""

    def __init__(
        self,
        column_name: str = "",
        categories: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize ordinal encoder.

        Args:
            column_name: Name of the column
            categories: Ordered list of categories (low to high)
        """
        super().__init__(**kwargs)
        self.column_name = column_name
        self._categories = categories or []
        self._mapping: Dict[str, int] = {}

    def fit(self, series: pd.Series) -> 'OrdinalEncoder':
        if self._categories:
            # Use provided ordering
            self._mapping = {str(v): i for i, v in enumerate(self._categories)}
        else:
            # Fall back to sorted order
            unique_values = series.dropna().unique()
            sorted_values = sorted([str(v) for v in unique_values])
            self._mapping = {v: i for i, v in enumerate(sorted_values)}

        name = self.column_name or series.name or "ordinal"
        self._feature_names = [name]
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        result = np.zeros(len(series), dtype=np.float64)
        unknown_value = len(self._mapping)

        for i, val in enumerate(series):
            if pd.isna(val):
                result[i] = np.nan
            else:
                str_val = str(val)
                result[i] = self._mapping.get(str_val, unknown_value)

        return result.reshape(-1, 1)

    def get_feature_names(self) -> List[str]:
        return self._feature_names


@EncoderRegistry.register("binary")
class BinaryEncoder(BaseEncoder):
    """Encode categorical values as binary representation."""

    def __init__(self, column_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self.column_name = column_name
        self._mapping: Dict[str, int] = {}
        self._n_bits: int = 0

    def fit(self, series: pd.Series) -> 'BinaryEncoder':
        unique_values = series.dropna().unique()
        sorted_values = sorted([str(v) for v in unique_values])
        self._mapping = {v: i for i, v in enumerate(sorted_values)}

        # Calculate number of bits needed
        n_categories = len(self._mapping)
        self._n_bits = max(1, int(np.ceil(np.log2(n_categories + 1))))

        prefix = self.column_name or series.name or "bin"
        self._feature_names = [f"{prefix}_bit{i}" for i in range(self._n_bits)]
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        result = np.zeros((len(series), self._n_bits), dtype=np.float64)

        for i, val in enumerate(series):
            if pd.isna(val):
                continue
            str_val = str(val)
            int_val = self._mapping.get(str_val, 0)
            # Convert to binary representation
            for bit in range(self._n_bits):
                result[i, bit] = (int_val >> bit) & 1

        return result

    def get_feature_names(self) -> List[str]:
        return self._feature_names


@EncoderRegistry.register("tfidf")
class TfidfEncoder(BaseEncoder):
    """Encode text using TF-IDF vectorization."""

    def __init__(
        self,
        column_name: str = "",
        max_features: int = 100,
        ngram_range: tuple = (1, 1),
        **kwargs
    ):
        """
        Initialize TF-IDF encoder.

        Args:
            column_name: Name of the column
            max_features: Maximum number of features
            ngram_range: N-gram range (min, max)
        """
        super().__init__(**kwargs)
        self.column_name = column_name
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizer = None

    def fit(self, series: pd.Series) -> 'TfidfEncoder':
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )

        # Convert to strings and handle NaN
        text_data = series.fillna("").astype(str)
        self._vectorizer.fit(text_data)

        prefix = self.column_name or series.name or "tfidf"
        vocab = self._vectorizer.get_feature_names_out()
        self._feature_names = [f"{prefix}_{word}" for word in vocab]
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        text_data = series.fillna("").astype(str)
        sparse_result = self._vectorizer.transform(text_data)
        return sparse_result.toarray()

    def get_feature_names(self) -> List[str]:
        return self._feature_names


def get_encoder(method: str, **kwargs) -> BaseEncoder:
    """
    Factory function to get an encoder by method name.

    Args:
        method: Encoder method name
        **kwargs: Additional arguments for the encoder

    Returns:
        Encoder instance
    """
    return EncoderRegistry.create(method, **kwargs)
