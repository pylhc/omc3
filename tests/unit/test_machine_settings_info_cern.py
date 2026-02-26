"""
Integration tests for machine_settings_info script that require CERN network access.

Tests the data retrieval, processing, and output generation for machine settings information
with real CERN data.
"""

from __future__ import annotations

import re
from datetime import datetime

import dateutil.tz as tz
import pytest

from omc3.knob_extractor import KNOB_CATEGORIES
from omc3.scripts.machine_settings_info import get_info

# Checking everything against the logbook here:
# https://be-op-logbook.web.cern.ch/elogbook-server/#/logbook?logbookId=1081&dateFrom=2025-11-07T00%3A00%3A00&dateTo=2025-11-07T23%3A59%3A59&eventToHighlight=4442872

REQUEST_TIME = "2025-11-07T00:19:55-01:00"
REQUEST_KWARGS = {
    "time": REQUEST_TIME,
    "timedelta": "_1h",
    "data_retrieval_days": 0.25,
    "accel": "lhc",
    "knob_definitions": True,
    "log": False,
}
EXPECTED_TIME = datetime(2025, 11, 7, 00, 19, 55, tzinfo=tz.UTC)
EXPECTED_FILL_START = datetime(2025, 11, 6, 21, 37, 34, tzinfo=tz.UTC)
EXPECTED_BEAMPROCESS = "PHYSICS-6.8TeV-1.2m-2025_V1@135_[END]"
EXPECTED_BP_START = datetime(2025, 11, 6, 22, 42, 17, 348000, tzinfo=tz.UTC)
EXPECTED_OPTICS = "R2025aRP_A120cmC120cmA10mL200cm"
ALLOWED_MISSING_KNOBS = { # No idea why these are missing, I assume they are meant to be missing for some reason (jgray 2026)
    "LHCBEAM/IP8-SDISP-CORR-XING-H",
    "LHCBEAM/IP2-XING-H-MURAD",
    "LHCBEAM/IP5-SDISP-CORR-SEP",
    "LHCBEAM/IP1-SDISP-CORR-SEP",
}


def _get_machine_settings_info(tmp_path, knobs: list[str]):
    return get_info({**REQUEST_KWARGS, "knobs": knobs, "output_dir": tmp_path})


def _get_default_knob_set() -> set[str]:
    knob_set = set()
    for knob_list in KNOB_CATEGORIES.values():
        knob_set.update(knob.replace(":", "/") for knob in knob_list)
    return knob_set - ALLOWED_MISSING_KNOBS


def _assert_result_context(result):
    assert result.accelerator == "lhc"
    assert result.time == EXPECTED_TIME


def _assert_fill_and_beamprocess(result):
    fill_info = result.fill
    assert fill_info.no == 11258, f"Wrong fill number, got {fill_info.no}"
    assert fill_info.accelerator == "lhc", f"Wrong fill accelerator, got {fill_info.accelerator}"
    assert fill_info.start_time.to_pydatetime().replace(microsecond=0) == EXPECTED_FILL_START, (
        f"Wrong fill start time, got {fill_info.start_time}"
    )
    assert result.beamprocess.name == EXPECTED_BEAMPROCESS

    assert result.beamprocess.name == EXPECTED_BEAMPROCESS, (
        f"Wrong beam process name, got {result.beamprocess.name}"
    )
    assert result.beamprocess.accelerator == "LHC", (
        f"Wrong beam process accelerator, got {result.beamprocess.accelerator}"
    )
    assert result.beamprocess.context_category == "OPERATIONAL", (
        f"Wrong beam process context category, got {result.beamprocess.context_category}"
    )
    assert result.beamprocess.start_time == EXPECTED_BP_START, (
        f"Wrong beam process start time, got {result.beamprocess.start_time}"
    )
    assert (
        result.beamprocess.description
        == "Actual beamprocess for PHYSICS-6.8TeV-1.2m-2025_V1@135\nCloned from: PHYSICS-6.8TeV-1.2m-2025_V1@135_[END]"
    ), f"Wrong beam process description, got {result.beamprocess.description}"
    assert result.beamprocess.category == "DISCRETE", (
        f"Wrong beam process category, got {result.beamprocess.category}"
    )


def _assert_optics(result):
    assert result.optics.name == EXPECTED_OPTICS, f"Wrong optics name, got {result.optics.name}"
    assert result.optics.start_time == EXPECTED_BP_START, (
        f"Wrong optics start time, got {result.optics.start_time}"
    )
    assert result.optics.accelerator == "LHC", (
        f"Wrong optics accelerator, got {result.optics.accelerator}"
    )
    assert result.optics.beamprocess == result.beamprocess, (
        f"Optics beam process does not match expected, got {result.optics.beamprocess}"
    )


def _assert_trim_histories(result, knob_set: set[str]):
    assert result.trim_histories.beamprocess == EXPECTED_BEAMPROCESS, (
        f"Wrong trim history beam process, got {result.trim_histories.beamprocess}"
    )
    assert result.trim_histories.accelerator == "LHC", (
        f"Wrong trim history accelerator, got {result.trim_histories.accelerator}"
    )
    assert result.trim_histories.end_time == result.time, (
        f"Wrong trim history end time, got {result.trim_histories.end_time}"
    )

    trim_set = set(result.trim_histories.trims.keys())
    missing_trims = knob_set - trim_set
    assert not missing_trims, f"Missing trim histories for knobs: {missing_trims}"

    for knob in knob_set:
        times = result.trim_histories.trims[knob].time
        data = result.trim_histories.trims[knob].data
        assert len(times) == len(data), (
            f"Trim history for {knob} has mismatched time and data lengths"
        )
        assert all(t2 > t1 for t1, t2 in zip(times, times[1:])), (
            f"Trim history for {knob} has non-increasing timestamps"
        )
        assert data[-1] == result.trims[knob], (
            f"Final trim value for {knob} does not match expected, got {data[-1]}, expected {result.trims[knob]}"
        )

    missing_knobs = knob_set - set(result.trims.keys())
    assert not missing_knobs, f"Missing final trim values for knobs: {missing_knobs}"


def _assert_landau_knob_definitions(result, beams: tuple[int, ...] = (1, 2)):
    knob_names = {f"LHCBEAM{beam}/LANDAU_DAMPING" for beam in beams}
    missing_defs = knob_names - set(result.knob_definitions.keys())
    assert not missing_defs, f"Missing knob definitions for knobs: {missing_defs}"

    for beam in beams:
        knob_def = result.knob_definitions[f"LHCBEAM{beam}/LANDAU_DAMPING"]
        assert knob_def.name == f"LHCBEAM{beam}/LANDAU_DAMPING", (
            f"Wrong knob definition name, got {knob_def.name}"
        )
        assert knob_def.optics == result.optics.name, (
            f"Knob definition optics does not match expected, got {knob_def.optics}"
        )
        for part in knob_def.parts:
            assert part.type == "K", f"Wrong knob part type, got {part.type}"
            assert part.factor == -6, f"Wrong knob part factor, got {part.factor}"
            assert re.match(rf"RO[FD]\.A(12|23|34|45|56|67|78|81)B{beam}/K3", part.circuit), (
                f"Knob part circuit does not match expected pattern, got {part.circuit}"
            )
            assert re.match(rf"ko[fd]\.a(12|23|34|45|56|67|78|81)b{beam}", part.madx_name), (
                f"Knob part madx name does not match expected pattern, got {part.madx_name}"
            )


@pytest.mark.cern_network
def test_cern_network_machine_settings_info_default(tmp_path):
    """Check that the default knobs works"""
    result = _get_machine_settings_info(tmp_path, knobs=["default"])

    _assert_result_context(result)
    _assert_fill_and_beamprocess(result)
    _assert_optics(result)
    _assert_trim_histories(result, _get_default_knob_set())
    _assert_landau_knob_definitions(result)


@pytest.mark.cern_network
def test_cern_network_machine_settings_info_landau(tmp_path):
    """Check that the landau damping knobs works"""
    landau_knobs = {"LHCBEAM1/LANDAU_DAMPING", "LHCBEAM2/LANDAU_DAMPING"}
    result = _get_machine_settings_info(tmp_path, knobs=sorted(landau_knobs))

    _assert_result_context(result)
    _assert_fill_and_beamprocess(result)
    _assert_optics(result)
    _assert_trim_histories(result, landau_knobs)
    _assert_landau_knob_definitions(result)
