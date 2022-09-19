import pytest
from pathlib import Path

from datetime import datetime, timezone, timedelta

# import private functions to test
from omc3.knob_extractor import _time_from_str, _add_delta, _get_knobs_dict, KNOB_NAMES


@pytest.mark.basic
def test_timezones():
    assert _time_from_str("2022-06-26T03:00+02:00") == datetime(2022, 6, 26, 3,
                                                                0, 0, tzinfo=timezone(timedelta(seconds=7200)))


@pytest.mark.basic
def test_time_and_delta():
    t1 = _time_from_str("2022-06-26T03:00")

    assert t1 == datetime(2022, 6, 26, 3, 0, 0)

    # 2 hours earlier
    t2 = _add_delta(t1, "_2h")
    assert t2 == datetime(2022, 6, 26, 1, 0, 0)

    # 1 week earlier
    t2 = _add_delta(t1, "_1w")
    assert t2 == datetime(2022, 6, 19, 3, 0, 0)

    # 1 week and 1 hour earlier
    t2 = _add_delta(t1, "_1w1h")
    assert t2 == datetime(2022, 6, 19, 2, 0, 0)

    # 1 month later
    t2 = _add_delta(t1, "1M")
    assert t2 == datetime(2022, 7, 26, 3, 0, 0)


@pytest.mark.basic
def test_knobdict():
    knobdict = _get_knobs_dict(Path(__file__).parent / "../inputs/knobs.txt" )

    for knobname in KNOB_NAMES:
        for knobkey in KNOB_NAMES[knobname]:
            assert knobkey.replace(":", "/") in knobdict
