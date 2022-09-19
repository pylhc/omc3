import re
from pathlib import Path

import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

import tfs

from omc3.knob_extractor import (
    _parse_time, _add_time_delta, main, _parse_knobs_defintions,
    KNOB_CATEGORIES, lsa2name, KnobEntry, _write_knobsfile, _extract
)

INPUTS = Path(__file__).parent.parent / "inputs" / "knob_extractor"


@pytest.mark.basic
def test_extraction():
    knobs_dict = {
        "LHCBEAM1:LANDAU_DAMPING": KnobEntry(madx="landau1", lsa="LHCBEAM1/LANDAU_DAMPING", scaling=-1),
        "LHCBEAM2:LANDAU_DAMPING": KnobEntry(madx="landau2", lsa="LHCBEAM1/LANDAU_DAMPING", scaling=-1),
        "other": KnobEntry(madx="other_knob", lsa="other/knob", scaling=1),
    }
    values = [8904238, 34.323904, 3489.23409]
    time = datetime.now()
    timestamp = time.timestamp()*1e9  # java format

    fake_ldb = {
        f"LhcStateTracker:{key}:target": {f"LhcStateTracker:{key}:target": [[timestamp, timestamp], [-1, value]]}
        for key, value in zip(knobs_dict.keys(), values)
    }

    extracted = _extract(fake_ldb, knobs_dict=knobs_dict, knob_categories=["mo", "other"], time=time)

    assert len(extracted) == len(knobs_dict)
    for idx, (key, entry) in enumerate(extracted.items()):
        assert entry.value == values[idx]  # depends on the order of "mo" in the file


@pytest.mark.basic
def test_parse_knobdict_from_file(knob_definitions):
    knob_dict = _parse_knobs_defintions(knob_definitions)
    for knobs in KNOB_CATEGORIES.values():
        for knob in knobs:
            assert knob in knob_dict.keys()
            knob_entry = knob_dict[knob]

            assert knob_entry.value is None
            assert abs(knob_entry.scaling) == 1
            assert lsa2name(knob_entry.lsa) == knob
            assert len(knob_entry.madx)
            assert knob_entry.madx in knob_entry.get_madx()
            assert knob_entry.get_madx().strip().startswith("!")

            knob_entry.value = 10
            assert str(10) in knob_entry.get_madx()


@pytest.mark.basic
def test_parse_knobdict_from_dataframe(tmp_path):
    df = pd.DataFrame(data=[["madx_name", "lsa_name", 1., "something"]], columns=["madx", "lsa", "scaling", "other"])

    path = tmp_path / "knob_defs.tfs"
    tfs.write(path, df)

    knob_dict = _parse_knobs_defintions(path)
    assert len(knob_dict) == 1
    assert "lsa_name" in knob_dict
    knob_entry = knob_dict["lsa_name"]
    assert knob_entry.lsa == "lsa_name"
    assert knob_entry.madx == "madx_name"
    assert knob_entry.scaling == 1


@pytest.mark.basic
def test_write_file(tmp_path):
    knobs_dict = {
        "LHCBEAM1:LANDAU_DAMPING": KnobEntry(madx="moknob1", lsa="moknob1.lsa", scaling=-1, value=-4783),
        "LHCBEAM2:LANDAU_DAMPING": KnobEntry(madx="moknob2", lsa="moknob2.lsa", scaling=1, value=0.0),  # one should be 0.0 to test this case
        "knob1": KnobEntry(madx="knob1.madx", lsa="knob1.lsa", scaling=-1, value=12.43383),
        "knob2": KnobEntry(madx="knob2.madx", lsa="knob2.lsa", scaling=1, value=-3.0231),
        "knob3": KnobEntry(madx="knob3.madx", lsa="knob3.lsa", scaling=-1, value=-9.7492),
    }
    path = tmp_path / "knobs.txt"
    time = datetime.now()
    _write_knobsfile(path, knobs_dict, time=time)
    read_as_dict, full_text = parse_output_file(path)
    assert str(time) in full_text
    assert " mo " in full_text
    assert " Other Knobs " in full_text
    assert len(read_as_dict) == len(knobs_dict)
    for _, entry in knobs_dict.items():
        assert read_as_dict[entry.madx] == entry.value * entry.scaling


@pytest.mark.basic
def test_time_and_delta():
    time_str = "2022-06-26T03:00"
    t1 = _parse_time(time_str)

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

    t3 = _parse_time(time_str, "20Y")
    assert t2 == t3


@pytest.mark.basic
def test_timezones():
    assert (
            _parse_time("2022-06-26T03:00+02:00")
            ==
            datetime(2022, 6, 26, 3, 0, 0, tzinfo=timezone(timedelta(seconds=7200)))
    )


# Helper -----------------------------------------------------------------------


def parse_output_file(file_path):
    txt = Path(file_path).read_text()
    d = {}
    pattern = re.compile(r"\s*(\S+)\s*:=\s*([^;\s*]+)\s*;")
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("!"):
            continue

        match = pattern.match(line)
        d[match.group(1)] = float(match.group(2))
    return d, txt


@pytest.fixture()
def knob_definitions():
    return INPUTS / "knob_definitions.txt"
