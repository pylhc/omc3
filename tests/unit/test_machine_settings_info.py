"""
Unit tests for machine_settings_info script and related functions.

Tests the data retrieval, processing, and output generation for machine settings information.
Includes tests for BeamProcess, Optics, and Knob extraction functions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import tfs

from omc3.machine_data_extraction.data_classes import (
    BeamProcessInfo,
    FillInfo,
    MachineSettingsInfo,
    OpticsInfo,
    TrimHistories,
    TrimHistoryHeader,
)
from omc3.scripts.machine_settings_info import get_info


class TestBeamProcessInfo:
    """Tests for BeamProcessInfo dataclass."""

    def test_create_beamprocess_info(self):
        """Test creating a BeamProcessInfo object."""
        now = datetime.now(timezone.utc)
        bp = BeamProcessInfo(
            name="RAMP",
            accelerator="lhc",
            context_category="PHYSICS",
            start_time=now,
            category="CYCLE",
            description="LHC ramp cycle",
        )
        assert bp.name == "RAMP"
        assert bp.accelerator == "lhc"
        assert bp.context_category == "PHYSICS"

    def test_beamprocess_info_from_java_mock(self):
        """Test creating BeamProcessInfo from mocked Java object."""
        # Create a mock Java object
        mock_java_bp = MagicMock()
        mock_java_bp.getName.return_value = "RAMP"
        mock_java_bp.getAccelerator.return_value.getName.return_value = "lhc"
        mock_java_bp.getContextCategory.return_value.toString.return_value = "PHYSICS"
        mock_java_bp.getCategory.return_value.toString.return_value = "CYCLE"
        mock_java_bp.getDescription.return_value = "LHC ramp cycle"
        mock_java_bp.getStartTime.return_value = 1704110400000  # milliseconds

        bp = BeamProcessInfo.from_java_beamprocess(mock_java_bp)

        assert bp.name == "RAMP"
        assert bp.accelerator == "lhc"
        assert bp.context_category == "PHYSICS"
        assert bp.category == "CYCLE"


class TestOpticsInfo:
    """Tests for OpticsInfo dataclass."""

    def test_create_optics_info(self):
        """Test creating an OpticsInfo object."""
        now = datetime.now(timezone.utc)
        optics = OpticsInfo(
            name="OPTICSYEAR1",
            id="001",
            start_time=now,
            accelerator="lhc",
        )
        assert optics.name == "OPTICSYEAR1"
        assert optics.id == "001"
        assert optics.accelerator == "lhc"


class TestFillInfo:
    """Tests for FillInfo dataclass."""

    def test_create_fill_info(self):
        """Test creating a FillInfo object."""
        now = datetime.now(timezone.utc)
        fill = FillInfo(
            no=12345,
            accelerator="lhc",
            start_time=now,
        )
        assert fill.no == 12345
        assert fill.accelerator == "lhc"
        assert fill.start_time == now

    def test_fill_info_hashable(self):
        """Test that FillInfo is hashable (can be added to sets)."""
        now = datetime.now(timezone.utc)
        fill1 = FillInfo(no=123, accelerator="lhc", start_time=now)
        fill2 = FillInfo(no=123, accelerator="lhc", start_time=now)

        # Should be able to add to set
        fill_set = {fill1, fill2}
        assert len(fill_set) == 1  # Same hash, should be deduplicated


class TestTrimHistories:
    """Tests for TrimHistories dataclass."""

    def test_create_trim_histories(self):
        """Test creating a TrimHistories object."""
        now = datetime.now(timezone.utc)
        trims = {}  # Empty trims dict

        trim_hist = TrimHistories(
            beamprocess="RAMP",
            start_time=now - timedelta(hours=1),
            end_time=now,
            accelerator="lhc",
            trims=trims,
        )

        assert trim_hist.beamprocess == "RAMP"
        assert trim_hist.accelerator == "lhc"
        assert hasattr(trim_hist, "headers")

    def test_trim_histories_tfs_conversion(self):
        """Test converting TrimHistories to TFS format."""
        now = datetime.now(timezone.utc)

        # Create mock trim data (would normally be TrimTuple from pjlsa)
        trims = {
            "knob1": MagicMock(time=[1, 2, 3], data=[0.1, 0.2, 0.3]),
            "knob2": MagicMock(time=[4, 5, 6], data=[0.4, 0.5, 0.6]),
        }

        trim_hist = TrimHistories(
            beamprocess="RAMP",
            start_time=now - timedelta(hours=1),
            end_time=now,
            accelerator="lhc",
            trims=trims,
        )

        # Convert to TFS files dict
        tfs_dict = trim_hist.to_tfs_dict()

        # Should have one TFS dataframe per knob
        assert len(tfs_dict) == 2
        assert "knob1" in tfs_dict
        assert "knob2" in tfs_dict

        # Each should be a TfsDataFrame
        for knob_name, df in tfs_dict.items():
            assert isinstance(df, tfs.TfsDataFrame)
            # Check headers are set correctly
            assert TrimHistoryHeader.BEAMPROCESS in df.headers
            assert df.headers[TrimHistoryHeader.BEAMPROCESS] == "RAMP"


class TestMachineSettingsInfo:
    """Tests for MachineSettingsInfo dataclass."""

    def test_create_empty_machine_settings_info(self):
        """Test creating a minimal MachineSettingsInfo object."""
        now = datetime.now(timezone.utc)
        info = MachineSettingsInfo(
            time=now,
            accelerator="lhc",
        )

        assert info.time == now
        assert info.accelerator == "lhc"
        assert info.fill is None
        assert info.beamprocess is None
        assert info.optics is None

    def test_create_complete_machine_settings_info(self):
        """Test creating a fully populated MachineSettingsInfo object."""
        now = datetime.now(timezone.utc)

        fill = FillInfo(no=12345, accelerator="lhc", start_time=now)
        bp = BeamProcessInfo(
            name="RAMP",
            accelerator="lhc",
            context_category="PHYSICS",
            start_time=now,
            category="CYCLE",
            description="Ramp cycle",
        )
        optics = OpticsInfo(name="OPTICSYEAR1", id="001", start_time=now)

        info = MachineSettingsInfo(
            time=now,
            accelerator="lhc",
            fill=fill,
            beamprocess=bp,
            optics=optics,
        )

        assert info.fill and info.fill.no == 12345
        assert info.beamprocess and info.beamprocess.name == "RAMP"
        assert info.optics and info.optics.name == "OPTICSYEAR1"


class TestMachineSettingsInfoOutput:
    """Tests for machine_settings_info output generation."""

    def test_summary_dataframe_generation(self):
        """Test generating summary DataFrame from MachineSettingsInfo."""
        from omc3.scripts.machine_settings_info import _summary_df

        now = datetime.now(timezone.utc)
        fill = FillInfo(no=12345, accelerator="lhc", start_time=now)
        bp = BeamProcessInfo(
            name="RAMP",
            accelerator="lhc",
            context_category="PHYSICS",
            start_time=now,
            category="CYCLE",
            description="Ramp cycle",
        )

        trims = {
            "knob1": 0.5,
            "knob2": 0.6,
        }

        info = MachineSettingsInfo(
            time=now,
            accelerator="lhc",
            fill=fill,
            beamprocess=bp,
            trims=trims,
        )

        df = _summary_df(info)

        # Check structure
        assert isinstance(df, tfs.TfsDataFrame)
        assert len(df) == 2
        assert "KNOB" in df.columns
        assert "VALUE" in df.columns

        # Check headers
        assert "ACCELERATOR" in df.headers
        assert df.headers["ACCELERATOR"] == "lhc"
        assert "BEAMPROCESS" in df.headers
        assert df.headers["BEAMPROCESS"] == "RAMP"
        assert "FILL" in df.headers

    def test_summary_dataframe_with_optics(self):
        """Test summary DataFrame includes optics information."""
        from omc3.scripts.machine_settings_info import _summary_df

        now = datetime.now(timezone.utc)
        fill = FillInfo(no=12345, accelerator="lhc", start_time=now)
        bp = BeamProcessInfo(
            name="RAMP",
            accelerator="lhc",
            context_category="PHYSICS",
            start_time=now,
            category="CYCLE",
            description="Ramp cycle",
        )
        optics = OpticsInfo(name="OPTICSYEAR1", id="001", start_time=now)

        info = MachineSettingsInfo(
            time=now,
            accelerator="lhc",
            fill=fill,
            beamprocess=bp,
            optics=optics,
            trims={"knob1": 0.5},
        )

        df = _summary_df(info)

        # Check optics in headers
        assert "OPTICS" in df.headers
        assert df.headers["OPTICS"] == "OPTICSYEAR1"
        assert "OPTICS_START" in df.headers


class TestMachineSettingsInfoFunctions:
    """Tests for helper functions in machine_settings_info."""

    @patch("omc3.scripts.machine_settings_info.get_beamprocess_with_fill_at_time")
    @patch("omc3.scripts.machine_settings_info.get_optics_for_beamprocess_at_time")
    def test_get_optics_handling(self, mock_get_optics, mock_get_bp):
        """Test that get_optics handles errors gracefully."""
        from omc3.scripts.machine_settings_info import _get_optics

        # Mock the LSA client
        mock_lsa_client = MagicMock()
        now = datetime.now(timezone.utc)

        # Create mock BeamProcessInfo
        bp_info = BeamProcessInfo(
            name="RAMP",
            accelerator="lhc",
            context_category="PHYSICS",
            start_time=now,
            category="CYCLE",
            description="Ramp cycle",
        )

        # Test successful case
        optics_info = OpticsInfo(name="OPTICSYEAR1", id="001", start_time=now)
        mock_get_optics.return_value = optics_info

        result = _get_optics(mock_lsa_client, now, bp_info)
        assert result and result.name == "OPTICSYEAR1"

        # Test error case - returns None
        mock_get_optics.side_effect = ValueError("No optics found")
        result = _get_optics(mock_lsa_client, now, bp_info)
        assert result is None

    @patch("omc3.scripts.machine_settings_info._get_optics")
    @patch("omc3.scripts.machine_settings_info._get_trim_history")
    @patch("omc3.scripts.machine_settings_info._get_knob_definitions")
    def test_get_info_with_knob_extraction(self, mock_get_defs, mock_get_trim, mock_get_optics):
        """Test get_info function with knob extraction."""
        with patch("omc3.scripts.machine_settings_info._get_clients") as mock_clients:
            mock_spark = MagicMock()
            mock_lsa_client = MagicMock()
            mock_clients.return_value = (mock_spark, mock_lsa_client)

            with patch(
                "omc3.scripts.machine_settings_info.get_beamprocess_with_fill_at_time"
            ) as mock_bp:
                now = datetime.now(timezone.utc)
                fill = FillInfo(no=12345, accelerator="lhc", start_time=now)
                bp = BeamProcessInfo(
                    name="RAMP",
                    accelerator="lhc",
                    context_category="PHYSICS",
                    start_time=now,
                    category="CYCLE",
                    description="Ramp cycle",
                )

                mock_bp.return_value = (fill, bp)
                mock_get_optics.return_value = None
                mock_get_trim.return_value = MagicMock()

                # Call with knobs = None (no knob extraction)
                result = get_info(
                    {
                        "time": "now",
                        "timedelta": None,
                        "data_retrieval_days": 0.25,
                        "knobs": None,
                        "accel": "lhc",
                        "output_dir": None,
                        "knob_definitions": False,
                        "log": False,
                    }
                )

                assert result.accelerator == "lhc"
                assert result.fill.no == 12345
                assert result.beamprocess.name == "RAMP"

    def test_get_trim_history_all_keyword(self):
        from omc3.scripts.machine_settings_info import _get_trim_history

        now = datetime.now(timezone.utc)
        bp = BeamProcessInfo(
            name="RAMP",
            accelerator="lhc",
            context_category="PHYSICS",
            start_time=now,
            category="CYCLE",
            description="Ramp cycle",
        )

        with patch("omc3.scripts.machine_settings_info.get_trim_history") as mock_get_trim:
            _get_trim_history(MagicMock(), ["all"], now, 0.25, bp)

        assert mock_get_trim.call_args.kwargs["knobs"] == []

    def test_get_knob_definitions_returns_none_without_optics(self):
        from omc3.scripts.machine_settings_info import _get_knob_definitions

        info = MachineSettingsInfo(time=datetime.now(timezone.utc), accelerator="lhc")
        result = _get_knob_definitions(MagicMock(), info)
        assert result is None

    def test_get_knob_definitions_returns_none_without_trims(self):
        from omc3.scripts.machine_settings_info import _get_knob_definitions

        now = datetime.now(timezone.utc)
        info = MachineSettingsInfo(
            time=now,
            accelerator="lhc",
            optics=OpticsInfo(name="OPTICSYEAR1", id="001", start_time=now),
            trim_histories=None,
        )
        result = _get_knob_definitions(MagicMock(), info)
        assert result is None

    def test_get_clients_initializes_spark_and_lsa(self, monkeypatch):
        from omc3.scripts import machine_settings_info

        mock_spark = MagicMock()
        mock_builder = MagicMock()
        mock_builder.get_or_create.return_value = mock_spark
        mock_pjlsa = MagicMock()
        mock_lsa_client = MagicMock()
        mock_pjlsa.LSAClient.return_value = mock_lsa_client

        monkeypatch.setattr(machine_settings_info, "spark_session_builder", mock_builder)
        monkeypatch.setattr(machine_settings_info, "pjlsa", mock_pjlsa)

        spark, lsa = machine_settings_info._get_clients()

        assert spark is mock_spark
        assert lsa is mock_lsa_client
        mock_spark.sparkContext.setLogLevel.assert_called_once_with("WARN")


class TestMachineSettingsInfoMoreOutput:
    """Additional output branch tests."""

    def test_write_output_with_trim_histories_sets_headers(self, tmp_path):
        from omc3.scripts.machine_settings_info import _write_output

        now = datetime.now(timezone.utc)
        trim_histories = MagicMock(spec=TrimHistories)
        trim_histories.headers = {}
        trim_histories.to_tfs_dict.return_value = {"knob1": tfs.TfsDataFrame()}

        info = MachineSettingsInfo(
            time=now,
            accelerator="lhc",
            fill=FillInfo(no=12345, accelerator="lhc", start_time=now),
            optics=OpticsInfo(name="OPTICSYEAR1", id="001", start_time=now),
            trim_histories=trim_histories,
            trims={"knob1": 0.5},
        )

        _write_output(tmp_path, info)

        assert trim_histories.headers[TrimHistoryHeader.OPTICS] == "OPTICSYEAR1"
        assert trim_histories.headers[TrimHistoryHeader.FILL] == 12345


class TestLoggingAndOutput:
    """Tests for logging and file output functionality."""

    @patch("omc3.scripts.machine_settings_info.LOGGER.info")
    def test_log_info_output(self, mock_log_info):
        """Test that machine info is logged correctly."""
        from omc3.scripts.machine_settings_info import _log_info

        now = datetime.now(timezone.utc)
        fill = FillInfo(no=12345, accelerator="lhc", start_time=now)
        bp = BeamProcessInfo(
            name="RAMP",
            accelerator="lhc",
            context_category="PHYSICS",
            start_time=now,
            category="CYCLE",
            description="Ramp cycle",
        )

        info = MachineSettingsInfo(
            time=now,
            accelerator="lhc",
            fill=fill,
            beamprocess=bp,
        )

        _log_info(info)

        mock_log_info.assert_called_once()
        logged_text = mock_log_info.call_args.args[0]
        assert "Summary" in logged_text
        assert "RAMP" in logged_text

    def test_write_output_creates_files(self, tmp_path):
        """Test that _write_output creates expected output files."""
        from omc3.scripts.machine_settings_info import _write_output

        now = datetime.now(timezone.utc)
        fill = FillInfo(no=12345, accelerator="lhc", start_time=now)
        bp = BeamProcessInfo(
            name="RAMP",
            accelerator="lhc",
            context_category="PHYSICS",
            start_time=now,
            category="CYCLE",
            description="Ramp cycle",
        )

        trims = {"knob1": 0.5, "knob2": 0.6}

        info = MachineSettingsInfo(
            time=now,
            accelerator="lhc",
            fill=fill,
            beamprocess=bp,
            trims=trims,
        )

        output_dir = tmp_path / "output"
        _write_output(output_dir, info)

        # Check summary file was created
        summary_file = output_dir / "machine_settings.tfs"
        assert summary_file.exists()

        # Check it's a valid TFS file
        df = tfs.read(summary_file)
        assert len(df) == 2  # Two trims
