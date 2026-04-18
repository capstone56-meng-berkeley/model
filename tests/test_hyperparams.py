"""Tests for src/hyperparams.py"""
import json
from pathlib import Path

import pytest

from src import hyperparams as hp


@pytest.fixture()
def store(tmp_path):
    """Return a temp path for the hyperparams store."""
    return tmp_path / "hyperparams.json"


MODELS_A = {
    "RF": {
        "best_cv_r2": 0.80,
        "tuned_cv_r2": 0.83,
        "delta": 0.03,
        "params": {"n_estimators": 200, "max_depth": 10},
    },
    "GBR": {
        "best_cv_r2": 0.78,
        "tuned_cv_r2": 0.81,
        "delta": 0.03,
        "params": {"n_estimators": 150, "learning_rate": 0.05},
    },
}

MODELS_B = {
    "ABR": {
        "best_cv_r2": 0.75,
        "tuned_cv_r2": 0.77,
        "delta": 0.02,
        "params": {"n_estimators": 100, "learning_rate": 1.0},
    },
}


class TestSave:
    def test_creates_file(self, store):
        hp.save("scope_a", MODELS_A, store_path=store)
        assert store.exists()

    def test_written_json_is_valid(self, store):
        hp.save("scope_a", MODELS_A, store_path=store)
        data = json.loads(store.read_text())
        assert "scope_a" in data

    def test_returns_path(self, store):
        result = hp.save("scope_a", MODELS_A, store_path=store)
        assert Path(result) == store

    def test_merge_preserves_other_scopes(self, store):
        hp.save("scope_a", MODELS_A, store_path=store)
        hp.save("scope_b", MODELS_B, store_path=store)
        data = json.loads(store.read_text())
        assert "scope_a" in data
        assert "scope_b" in data

    def test_overwrite_same_scope(self, store):
        hp.save("scope_a", MODELS_A, store_path=store)
        updated = {"RF": {**MODELS_A["RF"], "tuned_cv_r2": 0.99}}
        hp.save("scope_a", updated, store_path=store)
        models = hp.load("scope_a", store_path=store)
        assert models["RF"]["tuned_cv_r2"] == 0.99

    def test_best_model_field_set(self, store):
        hp.save("scope_a", MODELS_A, store_path=store)
        data = json.loads(store.read_text())
        # RF has higher tuned_cv_r2 (0.83 vs 0.81)
        assert data["scope_a"]["best_model"] == "RF"

    def test_metadata_fields(self, store):
        hp.save("scope_a", MODELS_A, n_trials=50, cv_protocol="RKF(5,5)",
                feature_matrix_shape=(82, 38), store_path=store)
        data = json.loads(store.read_text())["scope_a"]
        assert data["n_trials"] == 50
        assert data["cv_protocol"] == "RKF(5,5)"
        assert data["feature_matrix_shape"] == [82, 38]


class TestLoad:
    def test_missing_scope_returns_empty(self, store):
        hp.save("scope_a", MODELS_A, store_path=store)
        assert hp.load("nonexistent", store_path=store) == {}

    def test_missing_file_returns_empty(self, store):
        assert hp.load("scope_a", store_path=store) == {}

    def test_loads_models_dict(self, store):
        hp.save("scope_a", MODELS_A, store_path=store)
        models = hp.load("scope_a", store_path=store)
        assert "RF" in models
        assert models["RF"]["params"]["n_estimators"] == 200


class TestBestModel:
    def test_returns_best(self, store):
        hp.save("scope_a", MODELS_A, store_path=store)
        name, params = hp.best_model("scope_a", store_path=store)
        assert name == "RF"
        assert params["n_estimators"] == 200

    def test_missing_scope_returns_none(self, store):
        name, params = hp.best_model("nonexistent", store_path=store)
        assert name is None
        assert params is None


class TestHasParams:
    def test_true_when_present(self, store):
        hp.save("scope_a", MODELS_A, store_path=store)
        assert hp.has_params("scope_a", "RF", store_path=store)

    def test_false_when_missing_model(self, store):
        hp.save("scope_a", MODELS_A, store_path=store)
        assert not hp.has_params("scope_a", "SVR", store_path=store)

    def test_false_when_missing_scope(self, store):
        assert not hp.has_params("nonexistent", "RF", store_path=store)


class TestListScopes:
    def test_empty_store(self, store):
        assert hp.list_scopes(store_path=store) == []

    def test_returns_all_scopes(self, store):
        hp.save("scope_a", MODELS_A, store_path=store)
        hp.save("scope_b", MODELS_B, store_path=store)
        scopes = hp.list_scopes(store_path=store)
        assert set(scopes) == {"scope_a", "scope_b"}


class TestSummary:
    def test_empty_store(self, store):
        result = hp.summary(store_path=store)
        assert "empty" in result.lower()

    def test_contains_scope(self, store):
        hp.save("dp_steel", MODELS_A, store_path=store)
        result = hp.summary(store_path=store)
        assert "dp_steel" in result
        assert "RF" in result
