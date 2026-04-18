"""Tests for src/run_store.py"""
import csv
import json
import time

import pytest

from src.run_store import RunStore


@pytest.fixture()
def store(tmp_path):
    return RunStore("pipeline", base_dir=tmp_path)


class TestStart:
    def test_creates_run_dir(self, store):
        run_dir, run_id = store.start()
        assert run_dir.exists()
        assert run_dir.is_dir()

    def test_run_id_format(self, store):
        _, run_id = store.start()
        # YYYYMMDD_HHMMSS_xxxx
        parts = run_id.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8   # date
        assert len(parts[1]) == 6   # time
        assert len(parts[2]) == 4   # hex suffix

    def test_run_dir_under_family(self, store, tmp_path):
        run_dir, run_id = store.start()
        assert run_dir.parent == tmp_path / "pipeline"

    def test_two_starts_distinct(self, store):
        dir1, id1 = store.start()
        time.sleep(0.01)
        dir2, id2 = store.start()
        assert id1 != id2
        assert dir1 != dir2

    def test_properties_raise_before_start(self, tmp_path):
        s = RunStore("demo", base_dir=tmp_path)
        with pytest.raises(RuntimeError):
            _ = s.run_dir
        with pytest.raises(RuntimeError):
            _ = s.run_id


class TestWriteManifest:
    def test_creates_manifest_json(self, store):
        store.start()
        store.write_manifest({"score": 0.85})
        assert (store.run_dir / "manifest.json").exists()

    def test_manifest_contains_data(self, store):
        store.start()
        store.write_manifest({"score": 0.85, "model": "RF"})
        data = json.loads((store.run_dir / "manifest.json").read_text())
        assert data["score"] == 0.85
        assert data["model"] == "RF"

    def test_manifest_auto_fields(self, store):
        store.start()
        store.write_manifest({})
        data = json.loads((store.run_dir / "manifest.json").read_text())
        assert "run_id" in data
        assert "saved_at" in data
        assert "name" in data

    def test_incremental_merge(self, store):
        store.start()
        store.write_manifest({"a": 1})
        store.write_manifest({"b": 2})
        data = json.loads((store.run_dir / "manifest.json").read_text())
        assert data["a"] == 1
        assert data["b"] == 2

    def test_returns_path(self, store):
        store.start()
        p = store.write_manifest({})
        assert p == store.run_dir / "manifest.json"


class TestCopyArtifact:
    def test_copies_file(self, store, tmp_path):
        store.start()
        src = tmp_path / "model.joblib"
        src.write_bytes(b"fake model bytes")
        dest = store.copy_artifact(src)
        assert dest.exists()
        assert dest.read_bytes() == b"fake model bytes"

    def test_dest_name_override(self, store, tmp_path):
        store.start()
        src = tmp_path / "file.txt"
        src.write_text("hello")
        dest = store.copy_artifact(src, dest_name="renamed.txt")
        assert dest.name == "renamed.txt"

    def test_dest_in_run_dir(self, store, tmp_path):
        store.start()
        src = tmp_path / "f.txt"
        src.write_text("x")
        dest = store.copy_artifact(src)
        assert dest.parent == store.run_dir


class TestAppendHistory:
    def test_creates_history_csv(self, store):
        store.start()
        store.append_history({"metric": 0.9})
        assert store.history_path.exists()

    def test_writes_header(self, store):
        store.start()
        store.append_history({"r2": 0.85, "model": "RF"})
        with open(store.history_path, newline="") as f:
            reader = csv.DictReader(f)
            assert "r2" in reader.fieldnames
            assert "model" in reader.fieldnames

    def test_multiple_rows(self, store):
        store.start()
        store.append_history({"r2": 0.80})
        store.start()
        store.append_history({"r2": 0.85})
        with open(store.history_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_auto_run_id_in_row(self, store):
        store.start()
        store.append_history({"x": 1})
        with open(store.history_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["run_id"] == store.run_id

    def test_schema_evolution(self, store):
        """New columns in later rows should not corrupt earlier rows."""
        store.start()
        store.append_history({"a": 1})
        store.start()
        store.append_history({"a": 2, "b": 99})
        with open(store.history_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[1]["b"] == "99"

    def test_history_path_property(self, store, tmp_path):
        assert store.history_path == tmp_path / "pipeline" / "history.csv"
