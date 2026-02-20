"""
Unit tests for machine_data_extraction module functions.

These tests focus on the core functionality of data extraction functions,
using mocks for external dependencies (NXCALS, LSA) to avoid network dependencies.
Each function is tested in isolation with controlled inputs.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from omc3.machine_data_extraction.nxcals_knobs import NXCALSResult, get_knob_vals, get_raw_vars


class TestNXCALSResult:
    """Tests for NXCALSResult dataclass and its methods."""

    def test_create_nxcals_result(self):
        """Test creating an NXCALSResult object."""
        now = datetime.now(timezone.utc)
        result = NXCALSResult(
            name="kqt12.a12b1",
            value=0.5,
            datetime=now,
            pc_name="RPMBB.UA12.RQT12.A12B1",
        )
        assert result.name == "kqt12.a12b1"
        assert result.value == 0.5
        assert result.pc_name == "RPMBB.UA12.RQT12.A12B1"

    def test_nxcals_result_to_madx_format(self):
        """Test converting NXCALSResult to MAD-X format string."""
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = NXCALSResult(
            name="kqt12.a12b1",
            value=0.123456789,
            datetime=now,
            pc_name="RPMBB.UA12.RQT12.A12B1",
        )
        madx_str = result.to_madx()

        # Check format contains key elements
        assert "kqt12.a12b1" in madx_str
        assert re.search(r"1\.234567890*e-01", madx_str, re.IGNORECASE)
        assert "RPMBB.UA12.RQT12.A12B1" in madx_str
        assert "2025-01-01" in madx_str

    def test_nxcals_result_to_series(self):
        """Test converting NXCALSResult to pandas Series."""
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        result = NXCALSResult(
            name="kqt12.a12b1",
            value=0.5,
            datetime=now,
            pc_name="RPMBB.UA12.RQT12.A12B1",
        )
        series = result.to_series()

        assert series["MADX"] == "kqt12.a12b1"
        assert series["VALUE"] == 0.5
        assert series["PC_NAME"] == "RPMBB.UA12.RQT12.A12B1"
        assert "TIME" in series

    def test_nxcals_result_to_tfs_dataframe(self):
        """Test converting list of NXCALSResult to TFS DataFrame."""
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        results = [
            NXCALSResult("kqt12.a12b1", 0.5, now, "PC1"),
            NXCALSResult("kqt12.a23b1", 0.6, now, "PC2"),
        ]

        df = NXCALSResult.to_tfs(results, now.to_pydatetime(), beam=1)
        assert len(df) == 2
        assert "EXTRACTION_TIME" in df.headers
        assert "BEAM" in df.headers
        assert df.headers["BEAM"] == 1
        assert list(df["MADX"]) == ["kqt12.a12b1", "kqt12.a23b1"]


class TestGetRawVars:
    """Tests for get_raw_vars function."""

    @patch('omc3.machine_data_extraction.nxcals_knobs.builders')
    def test_get_raw_vars_success(self, mock_builders):
        """Test successful retrieval of raw variables from NXCALS."""
        # Setup mock Spark and NXCALS data
        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.__getitem__ = MagicMock(side_effect=lambda i: {
            0: "RPMBB.UA12.RQT12.A12B1:I_MEAS",
            1: 123.45,
            2: 1704110400000000000,  # nanoseconds
            "nxcals_variable_name": "RPMBB.UA12.RQT12.A12B1:I_MEAS",
            "nxcals_value": 123.45,
            "nxcals_timestamp": 1704110400000000000,
        }[i])

        mock_df.take.return_value = [mock_row]
        mock_df.select.return_value.collect.return_value = [mock_row]

        mock_builder = MagicMock()
        mock_builder.variables.return_value = mock_builder
        mock_builder.system.return_value = mock_builder
        mock_builder.nameLike.return_value = mock_builder
        mock_builder.timeWindow.return_value = mock_builder
        mock_builder.build.return_value = mock_df

        mock_builders.DataQuery.builder.return_value = mock_builder

        # Call function
        spark = MagicMock()
        time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        results = get_raw_vars(spark, time, "RPMBB.UA%.RQT%.A%B1:I_MEAS", latest_only=False)

        # Assertions
        assert len(results) == 1
        assert results[0].name == "RPMBB.UA12.RQT12.A12B1:I_MEAS"
        assert results[0].value == 123.45

    @patch('omc3.machine_data_extraction.nxcals_knobs.builders')
    def test_get_raw_vars_no_timezone_raises_error(self, mock_builders):
        """Test that get_raw_vars raises error for naive datetime."""
        spark = MagicMock()
        time = datetime(2024, 1, 1, 12, 0, 0)  # No timezone

        with pytest.raises(ValueError, match="timezone-aware"):
            get_raw_vars(spark, time, "RPMBB.UA%.RQT%.A%B1:I_MEAS")

    @patch('omc3.machine_data_extraction.nxcals_knobs.builders')
    def test_get_raw_vars_no_data_raises_error(self, mock_builders):
        """Test that get_raw_vars raises RuntimeError when no data found."""
        mock_df = MagicMock()
        mock_df.take.return_value = []  # No data

        mock_builder = MagicMock()
        mock_builder.variables.return_value = mock_builder
        mock_builder.system.return_value = mock_builder
        mock_builder.nameLike.return_value = mock_builder
        mock_builder.timeWindow.return_value = mock_builder
        mock_builder.build.return_value = mock_df

        mock_builders.DataQuery.builder.return_value = mock_builder

        spark = MagicMock()
        time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        with pytest.raises(RuntimeError, match="No data found"):
            get_raw_vars(spark, time, "RPMBB.UA%.RQT%.A%B1:I_MEAS")

    @patch('omc3.machine_data_extraction.nxcals_knobs.window')
    @patch('omc3.machine_data_extraction.nxcals_knobs.functions')
    @patch('omc3.machine_data_extraction.nxcals_knobs.builders')
    def test_get_raw_vars_latest_only(self, mock_builders, mock_functions, mock_window):
        """Test latest_only path uses windowing without Spark context."""
        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.__getitem__ = MagicMock(side_effect=lambda i: {
            "nxcals_variable_name": "RPMBB.UA12.RQT12.A12B1:I_MEAS",
            "nxcals_value": 123.45,
            "nxcals_timestamp": 1704110400000000000,
        }[i])

        mock_df.take.return_value = [mock_row]
        mock_df.select.return_value.collect.return_value = [mock_row]

        mock_builder = MagicMock()
        mock_builder.variables.return_value = mock_builder
        mock_builder.system.return_value = mock_builder
        mock_builder.nameLike.return_value = mock_builder
        mock_builder.timeWindow.return_value = mock_builder
        mock_builder.build.return_value = mock_df

        mock_builders.DataQuery.builder.return_value = mock_builder

        mock_window_spec = MagicMock()
        mock_window.Window.partitionBy.return_value.orderBy.return_value = mock_window_spec
        mock_functions.col.return_value.desc.return_value = MagicMock()
        mock_functions.row_number.return_value.over.return_value = MagicMock()

        mock_df.withColumn.return_value.filter.return_value = mock_df

        spark = MagicMock()
        time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        results = get_raw_vars(spark, time, "RPMBB.UA%.RQT%.A%B1:I_MEAS", latest_only=True)

        assert len(results) == 1
        mock_df.withColumn.assert_called_once()
        mock_df.withColumn.return_value.filter.assert_called_once()


@pytest.mark.usefixtures("mock_pjlsa")
class TestGetKnobVals:
    """Tests for get_knob_vals function."""

    @patch('omc3.machine_data_extraction.nxcals_knobs.pjlsa')
    @patch('omc3.machine_data_extraction.nxcals_knobs.get_raw_vars')
    @patch('omc3.machine_data_extraction.nxcals_knobs.get_energy')
    def test_get_knob_vals_basic(self, mock_get_energy, mock_get_raw_vars, mock_pjlsa):
        """Test basic knob value retrieval."""
        # Setup mocks
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        raw_var = NXCALSResult(
            "RPMBB.UA12.RQT12.A12B1:I_MEAS",
            100.0,
            now,
            "RPMBB.UA12.RQT12.A12B1"
        )
        mock_get_raw_vars.return_value = [raw_var]
        mock_get_energy.return_value = (7000.0, now)

        # Mock LSA client and K-value calculation
        mock_lsa_client = MagicMock()
        mock_pjlsa.LSAClient.return_value = mock_lsa_client

        with patch('omc3.machine_data_extraction.nxcals_knobs.lsa_utils.calc_k_from_iref') as mock_calc_k:
            mock_calc_k.return_value = {"RPMBB.UA12.RQT12.A12B1": 0.5}

            with patch('omc3.machine_data_extraction.nxcals_knobs.map_pc_name_to_madx') as mock_map:
                mock_map.return_value = "kqt12.a12b1"

                spark = MagicMock()
                time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
                patterns = ["RPMBB.UA%.RQT%.A%B1:I_MEAS"]
                expected_knobs = {"kqt12.a12b1"}

                results = get_knob_vals(
                    spark, time, beam=1, patterns=patterns,
                    expected_knobs=expected_knobs
                )

                assert len(results) == 1
                assert results[0].name == "kqt12.a12b1"
                assert results[0].value == 0.5

    @patch('omc3.machine_data_extraction.nxcals_knobs.pjlsa')
    @patch('omc3.machine_data_extraction.nxcals_knobs.get_raw_vars')
    @patch('omc3.machine_data_extraction.nxcals_knobs.get_energy')
    def test_get_knob_vals_multiple_patterns(self, mock_get_energy, mock_get_raw_vars, mock_pjlsa):
        """Test knob retrieval with multiple patterns."""
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")

        # Mock returns for two patterns
        mock_get_raw_vars.side_effect = [
            [NXCALSResult("RPMBB.UA12.RQT12.A12B1:I_MEAS", 100.0, now, "PC1")],
            [NXCALSResult("RPMBB.UA12.RQT12.A23B1:I_MEAS", 110.0, now, "PC2")],
        ]
        mock_get_energy.return_value = (7000.0, now)

        mock_lsa_client = MagicMock()
        mock_pjlsa.LSAClient.return_value = mock_lsa_client

        with patch('omc3.machine_data_extraction.nxcals_knobs.lsa_utils.calc_k_from_iref') as mock_calc_k:
            mock_calc_k.return_value = {
                "RPMBB.UA12.RQT12.A12B1": 0.5,
                "RPMBB.UA12.RQT12.A23B1": 0.6,
            }

            with patch('omc3.machine_data_extraction.nxcals_knobs.map_pc_name_to_madx') as mock_map:
                # Setup side effects for multiple calls
                def map_side_effect(key):
                    mapping = {
                        "RPMBB.UA12.RQT12.A12B1:I_MEAS": "kqt12.a12b1",
                        "RPMBB.UA12.RQT12.A23B1:I_MEAS": "kqt12.a23b1",
                        "RPMBB.UA12.RQT12.A12B1": "kqt12.a12b1",
                        "RPMBB.UA12.RQT12.A23B1": "kqt12.a23b1",
                    }
                    return mapping.get(key, key)

                mock_map.side_effect = map_side_effect

                spark = MagicMock()
                time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
                patterns = ["pattern1", "pattern2"]
                expected_knobs = {"kqt12.a12b1", "kqt12.a23b1"}

                results = get_knob_vals(
                    spark, time, beam=1, patterns=patterns,
                    expected_knobs=expected_knobs
                )

                assert len(results) == 2
                result_names = {r.name for r in results}
                assert result_names == expected_knobs

    @patch('omc3.machine_data_extraction.nxcals_knobs.pjlsa')
    @patch('omc3.machine_data_extraction.nxcals_knobs.get_raw_vars')
    @patch('omc3.machine_data_extraction.nxcals_knobs.get_energy')
    def test_get_knob_vals_missing_knobs_warning(self, mock_get_energy, mock_get_raw_vars, mock_pjlsa):
        """Test that missing knobs generate warnings."""
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        raw_var = NXCALSResult(
            "RPMBB.UA12.RQT12.A12B1:I_MEAS",
            100.0,
            now,
            "PC1"
        )
        mock_get_raw_vars.return_value = [raw_var]
        mock_get_energy.return_value = (7000.0, now)

        mock_lsa_client = MagicMock()
        mock_pjlsa.LSAClient.return_value = mock_lsa_client

        with patch('omc3.machine_data_extraction.nxcals_knobs.lsa_utils.calc_k_from_iref') as mock_calc_k:
            mock_calc_k.return_value = {"RPMBB.UA12.RQT12.A12B1": 0.5}

            with patch('omc3.machine_data_extraction.nxcals_knobs.map_pc_name_to_madx') as mock_map:
                mock_map.return_value = "kqt12.a12b1"

                # Request more knobs than available
                spark = MagicMock()
                time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
                patterns = ["RPMBB.UA%.RQT%.A%B1:I_MEAS"]
                expected_knobs = {"kqt12.a12b1", "kqt12.a23b1"}  # Second doesn't exist

                results = get_knob_vals(
                    spark, time, beam=1, patterns=patterns,
                    expected_knobs=expected_knobs
                )

                # Should return only available knob
                assert len(results) == 1
                assert results[0].name == "kqt12.a12b1"
