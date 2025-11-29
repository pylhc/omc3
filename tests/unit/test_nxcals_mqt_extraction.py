from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from omc3 import mqt_extractor
from omc3.nxcals import mqt_extraction
from omc3.nxcals.constants import EXTRACTED_MQTS_FILENAME
from omc3.nxcals.knob_extraction import NXCALSResult

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
                timestamp=timestamp,
                pc_name=pc_name,
            )
        )
    return results


@pytest.mark.cern_network
@pytest.mark.parametrize("beam, sample_file", TEST_CASES)
def test_get_mqts_matches_sample(beam: int, sample_file: Path):
    sample_results = _load_results_from_file(sample_file)
    expected_names = {result.name for result in sample_results}
    assert expected_names == mqt_extraction.get_mqts(beam=beam)


@pytest.mark.cern_network
@pytest.mark.parametrize("beam, sample_file", TEST_CASES)
def test_main_reproduces_reference_output(tmp_path, beam: int, sample_file: Path):
    sample_results = _load_results_from_file(sample_file)

    latest_sample_time = max(result.timestamp for result in sample_results)
    query_time = latest_sample_time.to_pydatetime()

    output_dir = tmp_path / f"beam{beam}"
    output_dir.mkdir()
    output_path = output_dir / EXTRACTED_MQTS_FILENAME

    mqt_extractor.main(time=query_time.isoformat(), beam=beam, output=output_path)

    expected_results = [
        NXCALSResult(
            name=r.name,
            value=r.value,
            pc_name=r.pc_name,
            timestamp=r.timestamp.tz_convert("UTC"),
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
        assert abs((exp.timestamp - act.timestamp).total_seconds()) <= 1


@pytest.mark.cern_network
def test_get_mqts_invalid_beam():
    with pytest.raises(ValueError):
        mqt_extraction.get_mqts(beam=3)


@pytest.mark.cern_network
@pytest.mark.parametrize("beam, sample_file", TEST_CASES)
def test_main_returns_tfs_dataframe(tmp_path, beam: int, sample_file: Path):
    """Test that main returns a TfsDataFrame with correct structure."""
    sample_results = _load_results_from_file(sample_file)
    latest_sample_time = max(result.timestamp for result in sample_results)
    query_time = latest_sample_time.to_pydatetime()

    output_path = tmp_path / f"test_output_b{beam}.madx"

    # Call main and verify return type
    result_df = mqt_extractor.main(time=query_time.isoformat(), beam=beam, output=output_path)

    # Check it's a TfsDataFrame
    import tfs

    assert isinstance(result_df, tfs.TfsDataFrame), "main should return a TfsDataFrame"

    # Check headers
    assert "EXTRACTION_TIME" in result_df.headers, "Missing EXTRACTION_TIME header"
    assert "BEAM" in result_df.headers, "Missing BEAM header"
    assert result_df.headers["BEAM"] == beam, f"Expected beam {beam} in headers"

    # Check columns
    expected_columns = ["madx", "value", "timestamp", "pc_name"]
    for col in expected_columns:
        assert col in result_df.columns, f"Missing column {col}"

    # Check we have the right number of rows
    assert len(result_df) == 16, f"Expected 16 MQT entries, got {len(result_df)}"


@pytest.mark.cern_network
@pytest.mark.parametrize("beam", [1, 2])
def test_main_with_timedelta(tmp_path, beam: int):
    """Test that timedelta parameter works correctly."""
    from datetime import datetime

    output_path = tmp_path / f"test_timedelta_b{beam}.madx"

    # Get current time
    now = datetime.now()

    # Call with timedelta going back 1 day
    result_df = mqt_extractor.main(
        time="now",
        timedelta="_1d",  # 1 day ago
        beam=beam,
        output=output_path,
    )

    # Verify the extraction time is approximately 1 day ago
    extraction_time = result_df.headers["EXTRACTION_TIME"]
    time_diff = abs((now - extraction_time).total_seconds())

    # Should be close to 1 day (86400 seconds), allow 5 minute tolerance
    one_day_seconds = 86400
    assert abs(time_diff - one_day_seconds) < 300, (
        f"Time difference should be ~1 day, but got {time_diff} seconds"
    )


@pytest.mark.cern_network
@pytest.mark.parametrize("beam", [1, 2])
def test_main_with_delta_days(tmp_path, beam: int):
    """Test that delta_days parameter is properly passed through."""
    from datetime import datetime, timedelta

    output_path = tmp_path / f"test_delta_days_b{beam}.madx"

    # Use a time 3 days ago with delta_days=5 to ensure we get data
    past_time = datetime.now() - timedelta(days=3)

    # This should work because we're looking back 5 days
    result_df = mqt_extractor.main(
        time=past_time.isoformat(), beam=beam, output=output_path, delta_days=5
    )

    # Should succeed and return valid data
    assert len(result_df) == 16, "Expected 16 MQT entries with delta_days=5"
    assert result_df.headers["BEAM"] == beam


def test_parse_time_now():
    """Test that _parse_time correctly handles 'now'."""
    from datetime import datetime

    result = mqt_extractor._parse_time("now")
    now = datetime.now()

    # Should be very close to now (within 1 second)
    diff = abs((now - result).total_seconds())
    assert diff < 1, f"_parse_time('now') should return current time, diff was {diff}s"


def test_parse_time_with_timedelta():
    """Test that _parse_time correctly applies timedelta."""
    from datetime import datetime

    now_str = datetime.now().isoformat()

    # Test positive timedelta
    result_plus = mqt_extractor._parse_time(now_str, "1h")
    result_base = mqt_extractor._parse_time(now_str)
    diff_plus = (result_plus - result_base).total_seconds()
    assert abs(diff_plus - 3600) < 1, f"1h timedelta should add 3600s, got {diff_plus}s"

    # Test negative timedelta
    result_minus = mqt_extractor._parse_time(now_str, "_2h")
    diff_minus = (result_minus - result_base).total_seconds()
    assert abs(diff_minus + 7200) < 1, f"_2h timedelta should subtract 7200s, got {diff_minus}s"
