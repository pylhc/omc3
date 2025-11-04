from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from omc3.nxcals import mqt_extraction
from omc3.nxcals.constants import EXTRACTED_MQTS_FILENAME
from omc3.nxcals.knob_extraction import NXCalResult
from omc3.scripts import mqt_extraction as mqt_script

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


def _load_results_from_file(file_path: Path, tz: str = "Europe/Zurich") -> list[NXCalResult]:
    results: list[NXCalResult] = []
    for raw_line in file_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        name, value, pc_name, timestamp = _parse_mqt_line(line, tz=tz)
        results.append(
            NXCalResult(
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

    mqt_script.retrieve_mqts(time=query_time, beam=beam, output_path=output_dir)

    output_path = output_dir / EXTRACTED_MQTS_FILENAME

    expected_results = [
        NXCalResult(
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
