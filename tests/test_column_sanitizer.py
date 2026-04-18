"""Tests for src/column_sanitizer.py"""
import re

import pandas as pd

from src.column_sanitizer import sanitize_column, sanitize_dataframe


class TestSanitizeColumn:
    def test_lowercase(self):
        assert sanitize_column("HoldingTemp") == "holdingtemp"

    def test_strip_whitespace(self):
        assert sanitize_column("  holding temp  ") == "holding_temp"

    def test_degc_unit(self):
        assert sanitize_column("Cycle1_HoldingTemp (C)") == "cycle1_holdingtemp_degc"

    def test_degc_symbol(self):
        assert sanitize_column("Temp (°C)") == "temp_degc"

    def test_min_unit(self):
        assert sanitize_column("HoldingTime (min)") == "holdingtime_min"

    def test_pct_unit(self):
        assert sanitize_column("Carbon (%)") == "carbon_pct"

    def test_degc_per_s(self):
        assert sanitize_column("CoolingRate (C/s)") == "coolingrate_degc_s"

    def test_collapse_underscores(self):
        assert sanitize_column("a__b___c") == "a_b_c"

    def test_strip_leading_trailing_underscores(self):
        assert sanitize_column("_col_") == "col"

    def test_special_chars_replaced(self):
        result = sanitize_column("col-name/value")
        assert re.match(r"^[a-z0-9_]+$", result)

    def test_idempotent(self):
        col = "cycle1_holdingtemp_degc"
        assert sanitize_column(col) == col

    def test_empty_string(self):
        # should not raise
        result = sanitize_column("")
        assert isinstance(result, str)

    def test_bracket_attached(self):
        # 'Qtime(min)' — bracket directly attached
        result = sanitize_column("Qtime(min)")
        assert "min" in result


class TestSanitizeDataframe:
    def test_renames_columns(self):
        df = pd.DataFrame({"HoldingTemp (C)": [1], "HoldingTime (min)": [2]})
        out = sanitize_dataframe(df)
        assert "holdingtemp_degc" in out.columns
        assert "holdingtime_min" in out.columns

    def test_preserves_data(self):
        df = pd.DataFrame({"Val (C)": [42.0]})
        out = sanitize_dataframe(df)
        assert out.iloc[0, 0] == 42.0

    def test_collision_raises(self):
        # 'A (C)' and 'A (°C)' both sanitize to 'a_degc' — must raise
        import pytest
        df = pd.DataFrame({"A (C)": [1], "A (°C)": [2]})
        with pytest.raises(ValueError, match="collision"):
            sanitize_dataframe(df)

    def test_returns_dataframe(self):
        df = pd.DataFrame({"X": [1, 2]})
        out = sanitize_dataframe(df)
        assert isinstance(out, pd.DataFrame)
