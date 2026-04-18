"""Tests for src/features.py — alignment and build_feature_matrix logic.

These tests mock the cache files rather than requiring real images or GPU,
so they run cleanly in CI.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import FeaturePipeline, _align_cache_to_ids, _normalise_id


# ---------------------------------------------------------------------------
# _normalise_id
# ---------------------------------------------------------------------------

class TestNormaliseId:
    def test_lowercase(self):
        assert _normalise_id("ABC") == "abc"

    def test_collapse_hyphens(self):
        assert _normalise_id("a-b-c") == "a_b_c"

    def test_collapse_spaces(self):
        assert _normalise_id("a b  c") == "a_b_c"

    def test_strip(self):
        assert _normalise_id("  abc  ") == "abc"

    def test_mixed(self):
        assert _normalise_id(" A-B C ") == "a_b_c"


# ---------------------------------------------------------------------------
# _align_cache_to_ids
# ---------------------------------------------------------------------------

def _write_npz(path, ids, feat_dim=4, seed=0):
    """Write a minimal .npz cache for testing."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(len(ids), feat_dim)).astype(np.float32)
    filenames = np.array([f"{rid}_F_0.jpg" for rid in ids])
    np.savez(str(path), X=X, filenames=filenames)
    return X


class TestAlignCacheToIds:
    def test_perfect_match(self, tmp_path):
        ids = ["sample_a", "sample_b", "sample_c"]
        X_cache = _write_npz(tmp_path / "cache.npz", ids)
        df_ids = pd.Series(ids)
        X_out, n_matched = _align_cache_to_ids(tmp_path / "cache.npz", df_ids)
        assert X_out.shape == (3, 4)
        assert n_matched == 3

    def test_partial_match(self, tmp_path):
        cache_ids = ["sample_a", "sample_b"]
        _write_npz(tmp_path / "cache.npz", cache_ids)
        df_ids = pd.Series(["sample_a", "sample_b", "sample_c"])  # c has no image
        X_out, n_matched = _align_cache_to_ids(tmp_path / "cache.npz", df_ids)
        assert n_matched == 2
        assert X_out.shape[0] == 3  # row count matches df

    def test_no_nan_in_output(self, tmp_path):
        _write_npz(tmp_path / "cache.npz", ["a"])
        df_ids = pd.Series(["a", "b"])  # b has no image
        X_out, _ = _align_cache_to_ids(tmp_path / "cache.npz", df_ids)
        assert not np.isnan(X_out).any()

    def test_case_insensitive_match(self, tmp_path):
        _write_npz(tmp_path / "cache.npz", ["Sample_A"])
        df_ids = pd.Series(["sample_a"])
        _, n_matched = _align_cache_to_ids(tmp_path / "cache.npz", df_ids)
        assert n_matched == 1

    def test_hyphen_normalisation(self, tmp_path):
        _write_npz(tmp_path / "cache.npz", ["sample-a"])
        df_ids = pd.Series(["sample_a"])
        _, n_matched = _align_cache_to_ids(tmp_path / "cache.npz", df_ids)
        assert n_matched == 1

    def test_multi_image_averaged(self, tmp_path):
        """Two images for same row → their features should be averaged."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(2, 4)).astype(np.float32)
        filenames = np.array(["row1_F_0.jpg", "row1_F_1.jpg"])
        np.savez(str(tmp_path / "cache.npz"), X=X, filenames=filenames)
        df_ids = pd.Series(["row1"])
        X_out, n_matched = _align_cache_to_ids(tmp_path / "cache.npz", df_ids)
        expected = X.mean(axis=0)
        np.testing.assert_allclose(X_out[0], expected, rtol=1e-5)
        assert n_matched == 1


# ---------------------------------------------------------------------------
# FeaturePipeline.load_image_features
# ---------------------------------------------------------------------------

class TestLoadImageFeatures:
    def test_returns_none_when_no_cache(self, tmp_path):
        fp = FeaturePipeline(data_dir=tmp_path, temp_dir=tmp_path / "imgs",
                             features_dir=tmp_path / "feats")
        df_ids = pd.Series(["a", "b"])
        result = fp.load_image_features("nonexistent_backbone", df_ids)
        assert result is None

    def test_returns_array_when_cache_exists(self, tmp_path):
        fp = FeaturePipeline(data_dir=tmp_path, temp_dir=tmp_path / "imgs",
                             features_dir=tmp_path / "feats")
        _write_npz(fp.cnn_cache_path("test_bb"), ["row1", "row2"], feat_dim=8)
        df_ids = pd.Series(["row1", "row2"])
        X = fp.load_image_features("test_bb", df_ids)
        assert X is not None
        assert X.shape == (2, 8)

    def test_output_row_count_matches_df(self, tmp_path):
        fp = FeaturePipeline(data_dir=tmp_path, temp_dir=tmp_path / "imgs",
                             features_dir=tmp_path / "feats")
        _write_npz(fp.cnn_cache_path("bb"), ["r1", "r2"], feat_dim=4)
        df_ids = pd.Series(["r1", "r2", "r3", "r4"])  # 4 rows, only 2 in cache
        X = fp.load_image_features("bb", df_ids)
        assert X.shape[0] == 4


# ---------------------------------------------------------------------------
# FeaturePipeline.build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_tabular_only_when_no_caches(self, tmp_path):
        fp = FeaturePipeline(data_dir=tmp_path, temp_dir=tmp_path / "imgs",
                             features_dir=tmp_path / "feats")
        X_tab = np.ones((5, 10), dtype=np.float64)
        df_ids = pd.Series(["a", "b", "c", "d", "e"])
        X_full, log = fp.build_feature_matrix(X_tab, "missing_bb", df_ids)
        assert X_full.shape == (5, 10)
        assert "tabular" in log

    def test_concatenates_image_and_tabular(self, tmp_path):
        fp = FeaturePipeline(data_dir=tmp_path, temp_dir=tmp_path / "imgs",
                             features_dir=tmp_path / "feats")
        _write_npz(fp.cnn_cache_path("bb"), ["a", "b", "c"], feat_dim=6)
        X_tab = np.ones((3, 4), dtype=np.float64)
        df_ids = pd.Series(["a", "b", "c"])
        X_full, log = fp.build_feature_matrix(X_tab, "bb", df_ids)
        assert X_full.shape == (3, 10)   # 6 + 4
        assert "image" in log
        assert "tabular" in log

    def test_stream_log_total(self, tmp_path):
        fp = FeaturePipeline(data_dir=tmp_path, temp_dir=tmp_path / "imgs",
                             features_dir=tmp_path / "feats")
        _write_npz(fp.cnn_cache_path("bb"), ["a"], feat_dim=5)
        X_tab = np.ones((1, 3), dtype=np.float64)
        X_full, log = fp.build_feature_matrix(X_tab, "bb", pd.Series(["a"]))
        assert "= 8 total" in log   # 5 + 3

    def test_output_dtype_float64(self, tmp_path):
        fp = FeaturePipeline(data_dir=tmp_path, temp_dir=tmp_path / "imgs",
                             features_dir=tmp_path / "feats")
        X_tab = np.ones((3, 4), dtype=np.float32)
        df_ids = pd.Series(["a", "b", "c"])
        X_full, _ = fp.build_feature_matrix(X_tab, "no_bb", df_ids)
        assert X_full.dtype == np.float64
