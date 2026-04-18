"""Tests for src/preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.config import EncodingConfig, MissingDataConfig, PreprocessingConfig, ScalingConfig
from src.preprocessing import FeaturePreprocessor


def make_cfg(
    drop_threshold=0.95,
    fill_strategy="median",
    scale="standard",
    encode="onehot",
):
    return PreprocessingConfig(
        missing_data=MissingDataConfig(
            column_drop_threshold=drop_threshold,
            row_fill_threshold=1.0,
            numeric_fill_strategy=fill_strategy,
        ),
        scaling=ScalingConfig(method=scale, enabled=(scale != "none")),
        encoding=EncodingConfig(categorical=encode, max_categories=20),
    )


@pytest.fixture()
def simple_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "num_a": rng.normal(100, 20, 30).tolist(),
        "num_b": rng.normal(5, 1, 30).tolist(),
        "cat_x": (["steel", "aluminium", "copper"] * 10),
    })


@pytest.fixture()
def df_with_missing():
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, 20).tolist()
    vals[3] = float("nan")
    vals[7] = float("nan")
    return pd.DataFrame({"a": vals, "b": rng.normal(1, 0.5, 20).tolist()})


class TestFitTransform:
    def test_returns_ndarray(self, simple_df):
        prep = FeaturePreprocessor(make_cfg())
        X = prep.fit_transform(simple_df)
        assert isinstance(X, np.ndarray)

    def test_no_nan_output(self, df_with_missing):
        prep = FeaturePreprocessor(make_cfg())
        X = prep.fit_transform(df_with_missing)
        assert not np.isnan(X).any()

    def test_row_count_preserved(self, simple_df):
        prep = FeaturePreprocessor(make_cfg())
        X = prep.fit_transform(simple_df)
        assert X.shape[0] == len(simple_df)

    def test_categorical_encoded(self, simple_df):
        prep = FeaturePreprocessor(make_cfg(encode="onehot"))
        X = prep.fit_transform(simple_df)
        # 2 numeric + at least 2 onehot cols (3 categories)
        assert X.shape[1] >= 4

    def test_drop_high_missing_column(self):
        df = pd.DataFrame({
            "good": [1.0, 2.0, 3.0, 4.0, 5.0],
            "bad":  [float("nan")] * 5,  # 100% missing
        })
        prep = FeaturePreprocessor(make_cfg(drop_threshold=0.90))
        X = prep.fit_transform(df)
        # "bad" column should be dropped → only 1 column remains
        assert X.shape[1] == 1


class TestTransform:
    def test_consistent_with_fit(self, simple_df):
        prep = FeaturePreprocessor(make_cfg())
        X_train = prep.fit_transform(simple_df.iloc[:20])
        X_test = prep.transform(simple_df.iloc[20:])
        assert X_train.shape[1] == X_test.shape[1]

    def test_transform_before_fit_raises(self, simple_df):
        prep = FeaturePreprocessor(make_cfg())
        with pytest.raises(RuntimeError):
            prep.transform(simple_df)


class TestScaling:
    def test_standard_scale_zero_mean(self, simple_df):
        numeric_df = simple_df[["num_a", "num_b"]]
        prep = FeaturePreprocessor(make_cfg(scale="standard", encode="onehot"))
        X = prep.fit_transform(numeric_df)
        # Standard scaled columns should have near-zero mean
        assert abs(X[:, 0].mean()) < 0.1

    def test_no_scaling(self, simple_df):
        numeric_df = simple_df[["num_a", "num_b"]]
        prep_scaled = FeaturePreprocessor(make_cfg(scale="standard"))
        prep_raw = FeaturePreprocessor(make_cfg(scale="none"))
        X_scaled = prep_scaled.fit_transform(numeric_df.copy())
        X_raw = prep_raw.fit_transform(numeric_df.copy())
        # Raw should have larger variance than scaled
        assert X_raw[:, 0].std() > X_scaled[:, 0].std()
