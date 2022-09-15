import pytest

# import private functions to test
from omc3.knob_extractor import _parse_time_from_str, _add_time_delta
# import the rest
from omc3.knob_extractor import *


@pytest.mark.basic
def test_time_and_delta():
    t1 = _parse_time_from_str("2022-06-26T03:00")

    assert t1 == datetime(2022, 6, 26, 3, 0, 0)

    # 2 hours earlier
    t2 = _add_time_delta(t1, "_2h")
    assert t2 == datetime(2022, 6, 26, 1, 0, 0)

    # 1 week earlier
    t2 = _add_time_delta(t1, "_1w")
    assert t2 == datetime(2022, 6, 19, 3, 0, 0)

    # 1 week and 1 hour earlier
    t2 = _add_time_delta(t1, "_1w1h")
    assert t2 == datetime(2022, 6, 19, 2, 0, 0)

    # 1 month later
    t2 = _add_time_delta(t1, "1M")
    assert t2 == datetime(2022, 7, 26, 3, 0, 0)

    # 20 years later
    t2 = _add_time_delta(t1, "20Y")
    assert t2 == datetime(2042, 6, 26, 3, 0, 0)


@pytest.mark.basic
def test_command_args():
    # TODO: maybe check the resulting `knobs.madx`

    # correct command
    try:
        main(["dummy", "disp", "chroma", "--time", "2022-05-04T14:00"])
    except Exception as e:
        assert False, e

    # invalid knob name
    try:
        main(["knob_extractor.py", "invalid_knob", "--time", "2022-05-04T14:00"])
    except:
        pass
    else:
        assert False, "this should throw"

    # another valid time string
    try:
        main(["knob_extractor.py", "disp", "--time", "2022-05-04 14:00"])
    except Exception as e:
        assert False, e

    # `now` is also a valid time string
    try:
        main(["knob_extractor.py", "disp", "--time", "now"])
    except Exception as e:
        assert False, e

    # invalid time string
    try:
        main(["knob_extractor.py", "disp", "--time", "hello,world"])
    except RuntimeError as e:
        pass # this should be thrown
    else:
        assert False, "this should throw"

    # test extraction of all the knobs
    try:
        main(["knob_extractor.py", "disp", "sep", "xing", "chroma", "ip_offset", "mo", "--time", "now"])
    except Exception as e:
        assert False, e


