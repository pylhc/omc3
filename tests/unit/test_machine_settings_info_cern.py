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


@pytest.mark.cern_network
def test_cern_network_machine_settings_info(tmp_path):
    """Check that the default knobs works"""
    # Example structure (fill in with real values):
    result = get_info(
        {
            "time": "2025-11-07T00:19:55-01:00",  # Time in UTC-1 timezone
            "timedelta": "_1h",  # Look back 1 hour from the specified time
            "data_retrieval_days": 0.25,
            "knobs": ["default"],
            "accel": "lhc",
            "output_dir": tmp_path,
            "knob_definitions": True,
            "log": False,
        }
    )

    assert result.accelerator == "lhc"
    assert result.time == datetime(2025, 11, 7, 00, 19, 55, tzinfo=tz.UTC)

    # Checking everything against the logbook here:
    # https://be-op-logbook.web.cern.ch/elogbook-server/#/logbook?logbookId=1081&dateFrom=2025-11-07T00%3A00%3A00&dateTo=2025-11-07T23%3A59%3A59&eventToHighlight=4442872

    # fill info
    fill_info = result.fill
    assert fill_info.no == 11258, f"Wrong fill number, got {fill_info.no}"
    assert fill_info.accelerator == "lhc", f"Wrong fill accelerator, got {fill_info.accelerator}"
    # Fill according to LHC logbook around 22:37:58 CET
    expected_start = datetime(2025, 11, 6, 21, 37, 34, tzinfo=tz.UTC)
    expected_beamprocess = "PHYSICS-6.8TeV-1.2m-2025_V1@135_[END]"
    assert fill_info.start_time.to_pydatetime().replace(microsecond=0) == expected_start, (
        f"Wrong fill start time, got {fill_info.start_time}"
    )
    assert result.beamprocess.name == expected_beamprocess

    # beam process info
    expected_bp_start = datetime(2025, 11, 6, 22, 42, 17, 348000, tzinfo=tz.UTC)
    assert result.beamprocess.name == expected_beamprocess, (
        f"Wrong beam process name, got {result.beamprocess.name}"
    )
    assert result.beamprocess.accelerator == "LHC", (
        f"Wrong beam process accelerator, got {result.beamprocess.accelerator}"
    )
    assert result.beamprocess.context_category == "OPERATIONAL", (
        f"Wrong beam process context category, got {result.beamprocess.context_category}"
    )
    assert result.beamprocess.start_time == expected_bp_start, (
        f"Wrong beam process start time, got {result.beamprocess.start_time}"
    )
    assert (
        result.beamprocess.description
        == "Actual beamprocess for PHYSICS-6.8TeV-1.2m-2025_V1@135\nCloned from: PHYSICS-6.8TeV-1.2m-2025_V1@135_[END]"
    ), f"Wrong beam process description, got {result.beamprocess.description}"

    # Should I be checking that this should be the case - what is discrete?
    assert result.beamprocess.category == "DISCRETE", (
        f"Wrong beam process category, got {result.beamprocess.category}"
    )

    # Optics info
    assert result.optics.name == "R2025aRP_A120cmC120cmA10mL200cm", (
        f"Wrong optics name, got {result.optics.name}"
    )  # At 1.2m at this time (flat top).
    assert result.optics.start_time == expected_bp_start, (
        f"Wrong optics start time, got {result.optics.start_time}"
    )
    assert result.optics.accelerator == "LHC", (
        f"Wrong optics accelerator, got {result.optics.accelerator}"
    )
    assert result.optics.beamprocess == result.beamprocess, (
        f"Optics beam process does not match expected, got {result.optics.beamprocess}"
    )

    # trim histories
    assert result.trim_histories.beamprocess == expected_beamprocess, (
        f"Wrong trim history beam process, got {result.trim_histories.beamprocess}"
    )
    assert result.trim_histories.accelerator == "LHC", (
        f"Wrong trim history accelerator, got {result.trim_histories.accelerator}"
    )
    # End time is actually the request time as the process hasnt ended yet
    assert result.trim_histories.end_time == result.time, (
        f"Wrong trim history end time, got {result.trim_histories.end_time}"
    )

    # Anyone know why these knobs are missing?
    # They are in the knob categories but not in the trim histories.
    allowed_missing_knobs = {
        "LHCBEAM/IP8-SDISP-CORR-XING-H",
        "LHCBEAM/IP2-XING-H-MURAD",
        "LHCBEAM/IP5-SDISP-CORR-SEP",
        "LHCBEAM/IP1-SDISP-CORR-SEP",
    }
    knob_set = set()
    for typ, knob_list in KNOB_CATEGORIES.items():
        knob_set.update([knob.replace(":", "/") for knob in knob_list])
    knob_set -= allowed_missing_knobs  # Remove allowed missing knobs

    trim_set = set(result.trim_histories.trims.keys())
    missing_trims = knob_set - trim_set
    assert not missing_trims, f"Missing trim histories for knobs: {missing_trims}"

    for knob in knob_set:
        # Check that the timstamps increase and has the same length as the data
        times = result.trim_histories.trims[knob].time
        data = result.trim_histories.trims[knob].data
        assert len(times) == len(data), (
            f"Trim history for {knob} has mismatched time and data lengths"
        )
        assert all(t2 > t1 for t1, t2 in zip(times, times[1:])), (
            f"Trim history for {knob} has non-increasing timestamps"
        )

        # check the final trim in the history is the same as the trims
        assert data[-1] == result.trims[knob], (
            f"Final trim value for {knob} does not match expected, got {data[-1]}, expected {result.trims[knob]}"
        )

    missing_knobs = knob_set - set(result.trims.keys())
    assert not missing_knobs, f"Missing final trim values for knobs: {missing_knobs}"

    # Check knob definitions were extracted and match requested context
    missing_defs = set(result.knob_definitions.keys()) - knob_set
    assert not missing_defs, f"Missing knob definitions for knobs: {missing_defs}"

    # Lets check the landau damping knobs as they are simple and should be there
    for beam in [1, 2]:
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

            # The circuit should be like: RO[FD]\.A(12|23|34|45|56|67|78|81)B1/K3
            assert re.match(rf"RO[FD]\.A(12|23|34|45|56|67|78|81)B{beam}/K3", part.circuit), (
                f"Knob part circuit does not match expected pattern, got {part.circuit}"
            )

            # The madx name should be like: ko[fd]\.a(12|23|34|45|56|67|78|81)b1
            assert re.match(rf"ko[fd]\.a(12|23|34|45|56|67|78|81)b{beam}", part.madx_name), (
                f"Knob part madx name does not match expected pattern, got {part.madx_name}"
            )
