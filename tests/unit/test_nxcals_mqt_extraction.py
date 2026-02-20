from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import tfs

from omc3 import mqt_extractor
from omc3.machine_data_extraction import mqt_extraction
from omc3.machine_data_extraction.mqt_extraction import generate_mqt_names
from omc3.machine_data_extraction.nxcals_knobs import NXCALSResult
from omc3.model.model_creators.lhc_model_creator import LhcBestKnowledgeCreator
from omc3.mqt_extractor import _write_mqt_file
from omc3.utils.time_tools import parse_time

SAMPLE_DIR = Path(__file__).parent.parent / "inputs" / "knob_extractor"
TEST_CASES = (
    (1, SAMPLE_DIR / "extracted_mqts_b1.str"),
    (2, SAMPLE_DIR / "extracted_mqts_b2.str"),
)


def _parse_mqt_line(line: str, tz: str = "Europe/Zurich") -> tuple[str, float, str, pd.Timestamp]:
    pattern = re.compile(
        r"([a-zA-Z0-9_.]+)\s*=\s*([0-9.eE+-]+)\s*;\s*! powerconverter:\s*([a-zA-Z0-9_.]+)\s*at\s*(.+)"
    )
    match = pattern.match(line.strip())
    if not match:
        raise ValueError(f"Unexpected line format: {line}")
    name, value_str, pc_name, timestamp_raw = match.groups()
    value = float(value_str)
    timestamp = pd.Timestamp(timestamp_raw.strip(), tz=tz).replace(microsecond=0)
    return name, value, pc_name, timestamp


def _load_results_from_file(file_path: Path, tz: str = "Europe/Zurich") -> list[NXCALSResult]:
    results: list[NXCALSResult] = []
    for raw_line in file_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        name, value, pc_name, timestamp = _parse_mqt_line(line, tz=tz)
        results.append(
            NXCALSResult(
                name=name,
                value=value,
                datetime=timestamp,
                pc_name=pc_name,
            )
        )
    return results


@pytest.mark.cern_network
@pytest.mark.parametrize("beam, sample_file", TEST_CASES)
def test_get_mqts_matches_sample(beam: int, sample_file: Path):
    sample_results = _load_results_from_file(sample_file)
    expected_names = {result.name for result in sample_results}
    assert expected_names == mqt_extraction.generate_mqt_names(beam=beam)


@pytest.mark.cern_network
@pytest.mark.parametrize("beam, sample_file", TEST_CASES)
def test_main_reproduces_reference_output(tmp_path, beam: int, sample_file: Path):
    sample_results = _load_results_from_file(sample_file)

    latest_sample_time = max(result.datetime for result in sample_results)
    query_time = latest_sample_time

    output_dir = tmp_path / f"beam{beam}"
    output_dir.mkdir()
    output_path = output_dir / LhcBestKnowledgeCreator.EXTRACTED_MQTS_FILENAME

    mqt_extractor.main(time=query_time.isoformat(), beam=beam, output=output_path)

    expected_results = [
        NXCALSResult(
            name=r.name,
            value=r.value,
            pc_name=r.pc_name,
            datetime=r.datetime,
        )
        for r in sample_results
    ]

    actual_results = _load_results_from_file(output_path, tz="UTC")

    expected_sorted = sorted(expected_results, key=lambda x: x.name)
    actual_sorted = sorted(actual_results, key=lambda x: x.name)

    for exp, act in zip(expected_sorted, actual_sorted):
        assert exp.name == act.name
        assert exp.pc_name == act.pc_name
        assert exp.value == pytest.approx(act.value, rel=1e-4)
        assert abs((exp.datetime - act.datetime).total_seconds()) <= 1


@pytest.mark.cern_network
def test_get_mqts_invalid_beam():
    with pytest.raises(ValueError):
        mqt_extraction.generate_mqt_names(beam=3)


@pytest.mark.cern_network
@pytest.mark.parametrize("beam, sample_file", TEST_CASES)
def test_main_returns_tfs_dataframe(tmp_path, beam: int, sample_file: Path):
    """Test that main returns a TfsDataFrame with correct structure."""
    sample_results = _load_results_from_file(sample_file)
    latest_sample_time = max(result.datetime for result in sample_results)
    query_time = latest_sample_time

    output_path = tmp_path / f"test_output_b{beam}.madx"

    # Call main and verify return type
    result_df = mqt_extractor.main(time=query_time.isoformat(), beam=beam, output=output_path)

    # Check it's a TfsDataFrame
    assert isinstance(result_df, tfs.TfsDataFrame), "main should return a TfsDataFrame"

    # Check headers
    assert "EXTRACTION_TIME" in result_df.headers, "Missing EXTRACTION_TIME header"
    assert "BEAM" in result_df.headers, "Missing BEAM header"
    assert result_df.headers["BEAM"] == beam, f"Expected beam {beam} in headers"

    # Check columns
    expected_columns = ["MADX", "VALUE", "TIMESTAMP", "PC_NAME", "TIME"]
    for col in expected_columns:
        assert col in result_df.columns, f"Missing column {col}"

    # Check we have the right number of rows
    assert len(result_df) == 16, f"Expected 16 MQT entries, got {len(result_df)}"


@pytest.mark.cern_network
@pytest.mark.parametrize("beam", [1, 2], ids=["beam1", "beam2"])
def test_main_with_timedelta(tmp_path, beam: int):
    """Test that timedelta parameter works correctly."""
    output_path = tmp_path / f"test_timedelta_b{beam}.madx"

    # Call with timedelta going back 1 day
    result_df = mqt_extractor.main(
        time="2025-11-07T07:00:00+00:00",
        timedelta="_1d",  # 1 day ago
        beam=beam,
        output=output_path,
    )

    # Verify the extraction time is approximately 1 day ago
    extraction_time = result_df.headers["EXTRACTION_TIME"]
    expected_time = datetime(2025, 11, 6, 7, 0, 0, tzinfo=timezone.utc)
    time_diff = abs((extraction_time - expected_time).total_seconds())

    # Should be close to 1 day ago, allow 5 minute tolerance
    assert time_diff < 300, f"Extraction time should be ~1 day ago, but diff is {time_diff} seconds"


@pytest.mark.cern_network
@pytest.mark.parametrize("beam", [1, 2], ids=["beam1", "beam2"])
def test_main_with_delta_days(tmp_path, beam: int):
    """Test that data_retrieval_days parameter is properly passed through."""
    output_path = tmp_path / f"test_delta_days_b{beam}.madx"

    # Use a time 2 hours before 7am on 2025-11-07 with data_retrieval_days=2/12 (~4 hours) to ensure we get data
    past_time = datetime(2025, 11, 7, 5, 0, 0, tzinfo=timezone.utc)

    # This should work because we're looking back 4 hours from 5am (covers 7am)
    result_df = mqt_extractor.main(
        time=past_time.isoformat(), beam=beam, output=output_path, data_retrieval_days=2 / 12
    )

    # Should succeed and return valid data
    assert len(result_df) == 16, "Expected 16 MQT entries with data_retrieval_days=2/12 (~4 hours)"
    assert result_df.headers["BEAM"] == beam


def test_parse_time_now():
    """Test that _parse_time correctly handles 'now'."""
    result = parse_time("now")
    now = datetime.now(timezone.utc)

    # Should be very close to now (within 1 second)
    diff = abs((now - result).total_seconds())
    assert diff < 1, f"parse_time('now') should return current time, diff was {diff}s"


def test_parse_time_with_timedelta():
    """Test that _parse_time correctly applies timedelta."""
    now_str = datetime.now(timezone.utc).isoformat()

    # Test positive timedelta
    result_plus = parse_time(now_str, "1h")
    result_base = parse_time(now_str)
    diff_plus = (result_plus - result_base).total_seconds()
    assert abs(diff_plus - 3600) < 1, f"1h timedelta should add 3600s, got {diff_plus}s"

    # Test negative timedelta
    result_minus = parse_time(now_str, "_2h")
    diff_minus = (result_minus - result_base).total_seconds()
    assert abs(diff_minus + 7200) < 1, f"_2h timedelta should subtract 7200s, got {diff_minus}s"

# Unit tests (no CERN network required) ########################################


class TestMQTGeneration:
    """Unit tests for MQT generation logic."""

    def test_generate_mqt_names_beam1(self):
        """Test MQT name generation for beam 1."""

        names = generate_mqt_names(beam=1)

        # Should have 16 names (8 arcs * 2 types)
        assert len(names) == 16

        # All should end with b1
        assert all(name.endswith("b1") for name in names)

        # Should have both f and d types
        assert any("kqtf" in name for name in names)
        assert any("kqtd" in name for name in names)

        # Check specific expected names
        assert "kqtf.a12b1" in names
        assert "kqtd.a81b1" in names

    def test_generate_mqt_names_beam2(self):
        """Test MQT name generation for beam 2."""

        names = generate_mqt_names(beam=2)

        assert len(names) == 16
        assert all(name.endswith("b2") for name in names)

        # Check specific expected names
        assert "kqtf.a12b2" in names
        assert "kqtd.a81b2" in names

    def test_generate_mqt_names_invalid_beam(self):
        """Test that invalid beam raises ValueError."""
        with pytest.raises(ValueError, match="Beam must be 1 or 2"):
            generate_mqt_names(beam=0)

        with pytest.raises(ValueError, match="Beam must be 1 or 2"):
            generate_mqt_names(beam=3)

    def test_generate_mqt_names_beams_disjoint(self):
        """Test that beam 1 and beam 2 MQT names are different."""
        names_b1 = generate_mqt_names(beam=1)
        names_b2 = generate_mqt_names(beam=2)

        # They should be disjoint
        assert names_b1.isdisjoint(names_b2)


class TestMQTExtractorFileHandling:
    """Unit tests for MQT file I/O operations."""

    def test_mqt_file_writing(self, tmp_path):
        """Test that MQT values are correctly written to file."""
        output_file = tmp_path / "mqts.madx"
        now = datetime.now(timezone.utc)

        mqt_vals = [
            NXCALSResult("kqtf.a12b1", 0.5, now, "PC1"),
            NXCALSResult("kqtd.a12b1", 0.6, now, "PC2"),
        ]

        _write_mqt_file(output_file, mqt_vals, now, beam=1)

        # Check file was created
        assert output_file.exists()

        # Check content
        content = output_file.read_text()
        assert "kqtf.a12b1" in content
        assert "kqtd.a12b1" in content
        assert re.search(r"\b0\.5\b|\b5\.0+e-?0?1\b", content, re.IGNORECASE)
        assert re.search(r"\b0\.6\b|\b6\.0+e-?0?1\b", content, re.IGNORECASE)
        assert "beam 1" in content.lower()

    def test_mqt_file_format(self, tmp_path):
        """Test that MQT file follows expected MAD-X format."""
        output_file = tmp_path / "mqts.madx"
        now = datetime.now(timezone.utc)

        mqt_vals = [
            NXCALSResult("kqtf.a12b1", 0.123456789, now, "PC1"),
        ]

        _write_mqt_file(output_file, mqt_vals, now, beam=1)

        content = output_file.read_text()
        lines = content.strip().split("\n")

        # First two lines should be comments
        assert lines[0].startswith("!!")
        assert lines[1].startswith("!!")

        # Should have assignment statements
        assert any("=" in line for line in lines[2:])
        assert any(";" in line for line in lines[2:])


class TestMQTExtractorMain:
    """Unit tests for mqt_extractor.main without CERN access."""

    def test_main_no_network(self, tmp_path):
        """Test main returns TfsDataFrame and writes output using mocks."""
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        results = [
            NXCALSResult("kqtf.a12b1", 0.5, now, "PC1"),
            NXCALSResult("kqtd.a12b1", 0.6, now, "PC2"),
        ]

        output_path = tmp_path / "mqts.madx"

        with pytest.MonkeyPatch.context() as mpatch:
            mpatch.setattr(
                "omc3.mqt_extractor.spark_session_builder.get_or_create",
                lambda conf=None: MagicMock(),
            )
            mpatch.setattr("omc3.mqt_extractor.get_mqt_vals", lambda *_args, **_kw: results)

            df = mqt_extractor.main(time=now.isoformat(), beam=1, output=output_path)

        assert isinstance(df, tfs.TfsDataFrame)
        assert output_path.exists()
        assert df.headers["BEAM"] == 1


class TestNXCALSResultFormatting:
    """Unit tests for NXCALSResult formatting methods."""

    def test_to_madx_format_precision(self):
        """Test that to_madx maintains appropriate precision."""
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        result = NXCALSResult(
            "kqtf.a12b1",
            0.123456789012345,
            now,
            "RPMBB.UA12.RQT12.A12B1",
        )

        madx_str = result.to_madx()

        # Should contain scientific notation with reasonable precision
        assert "e" in madx_str or "." in madx_str
        # Knob name should be left-aligned and padded
        assert "kqtf.a12b1" in madx_str

    def test_to_madx_format_with_different_timezones(self):
        """Test that to_madx works with different timezones."""
        now_utc = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        now_zurich = now_utc.tz_convert("Europe/Zurich")

        result_utc = NXCALSResult("kqtf.a12b1", 0.5, now_utc, "PC1")
        result_zurich = NXCALSResult("kqtf.a12b1", 0.5, now_zurich, "PC1")

        madx_utc = result_utc.to_madx()
        madx_zurich = result_zurich.to_madx()

        # Both should be valid strings
        assert len(madx_utc) > 0
        assert len(madx_zurich) > 0
        # But with different timestamps
        assert madx_utc != madx_zurich


class TestTFSDataFrameGeneration:
    """Unit tests for TFS DataFrame generation from MQT results."""

    def test_tfs_dataframe_structure(self):
        """Test the structure of generated TFS DataFrame."""
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        results = [
            NXCALSResult("kqtf.a12b1", 0.5, now, "PC1"),
            NXCALSResult("kqtd.a12b1", 0.6, now, "PC2"),
            NXCALSResult("kqtf.a23b1", 0.7, now, "PC3"),
        ]

        df = NXCALSResult.to_tfs(results, now.to_pydatetime(), beam=1)

        # Check structure
        assert len(df) == 3

        # Check required columns - should we use the constants jgray 2026?
        required_cols = ["MADX", "VALUE", "TIMESTAMP", "PC_NAME", "TIME"]
        for col in required_cols:
            assert col in df.columns, f"Missing column {col}"

        # Check values
        assert list(df["MADX"]) == ["kqtf.a12b1", "kqtd.a12b1", "kqtf.a23b1"]
        assert list(df["VALUE"]) == [0.5, 0.6, 0.7]

    def test_tfs_headers_correct(self):
        """Test that TFS headers are set correctly."""
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        results = [NXCALSResult("kqtf.a12b1", 0.5, now, "PC1")]

        df = NXCALSResult.to_tfs(results, now.to_pydatetime(), beam=2)

        # Check headers
        assert "EXTRACTION_TIME" in df.headers
        assert "BEAM" in df.headers
        assert df.headers["BEAM"] == 2
        assert df.headers["EXTRACTION_TIME"] == now.to_pydatetime()


class TestMQTExtractionEdgeCases:
    """Unit tests for edge cases in MQT extraction."""

    def test_empty_mqt_list_to_tfs(self):
        """Test TFS generation with empty MQT list."""
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        results = []

        df = NXCALSResult.to_tfs(results, now.to_pydatetime(), beam=1)

        assert len(df) == 0
        assert "EXTRACTION_TIME" in df.headers
        assert df.headers["BEAM"] == 1

    def test_nxcals_result_with_zero_value(self):
        """Test handling of zero K-values."""
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        result = NXCALSResult("kqtf.a12b1", 0.0, now, "PC1")

        madx_str = result.to_madx()
        assert "0" in madx_str

        series = result.to_series()
        assert series["VALUE"] == 0.0

    def test_nxcals_result_with_negative_value(self):
        """Test handling of negative K-values."""
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        result = NXCALSResult("kqtf.a12b1", -0.5, now, "PC1")

        madx_str = result.to_madx()
        assert "-" in madx_str or "e-" in madx_str

        series = result.to_series()
        assert series["VALUE"] == -0.5

    def test_nxcals_result_with_very_large_value(self):
        """Test handling of very large K-values."""
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        result = NXCALSResult("kqtf.a12b1", 1234567.89, now, "PC1")

        madx_str = result.to_madx()
        assert "1234567" in madx_str or "e" in madx_str

        series = result.to_series()
        assert series["VALUE"] == 1234567.89

    def test_nxcals_result_with_very_small_value(self):
        """Test handling of very small K-values."""
        now = pd.Timestamp("2025-01-01 12:00:00", tz="UTC")
        result = NXCALSResult("kqtf.a12b1", 1.234567e-6, now, "PC1")

        madx_str = result.to_madx()
        # Should be in scientific notation
        assert "e" in madx_str.lower()

        series = result.to_series()
        assert abs(series["VALUE"] - 1.234567e-6) < 1e-12
