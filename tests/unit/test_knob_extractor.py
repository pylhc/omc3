import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import pytest
import tfs

from generic_parser import EntryPoint
from generic_parser.dict_parser import ArgumentError
from omc3 import knob_extractor
from omc3.knob_extractor import (KNOB_CATEGORIES, KnobEntry, _add_time_delta,
                                 _extract, _parse_knobs_defintions,
                                 _parse_time, _write_knobsfile, lsa2name, main,
                                 get_params
                                 )

from tests.conftest import cli_args

INPUTS = Path(__file__).parent.parent / "inputs" / "knob_extractor"


class TestFullRun:
    @pytest.mark.basic
    @pytest.mark.parametrize("commandline", [True, False], ids=["as function", "cli"])
    def test_full(self, tmp_path, knob_definitions, monkeypatch, commandline):
        knobs_dict = _parse_knobs_defintions(knob_definitions)
        all_variables = [knob for category in KNOB_CATEGORIES.values() for knob in category]
        for knob in all_variables:
            value = np.random.random() * 10 - 5
            threshold = np.random.random() < 0.3
            knobs_dict[knob].value = 0.0 if threshold else value

        start_time = datetime.now().timestamp()
        path = tmp_path / "knobs.txt"

        # Mock Pytimber ---
        class MyLDB:
            def __init__(self, *args, **kwargs):
                pass

            @staticmethod
            def get(key, time):
                now_time = datetime.now().timestamp()
                assert start_time <= time <= now_time
                name = ":".join(key.split(":")[1:-1])
                return {key: [[739173129, 42398328], [-1, knobs_dict[name].value]]}

        class MockTimber:
            LoggingDB = MyLDB

        monkeypatch.setattr(knob_extractor, "pytimber", MockTimber())

        # Main ---
        if commandline:
            with cli_args("--time", "now",
                          "--output", str(path),
                          "--knob_definitions", str(knob_definitions),
                          script="knob_extract.py"):
                main()

            # Asserts ---
            parsed_output, _ = parse_output_file(path)
            assert len(all_variables) == len(parsed_output)
            for knob in all_variables:
                assert parsed_output[knobs_dict[knob].madx] == knobs_dict[knob].value * knobs_dict[knob].scaling

        else:
            knobs_extracted = main(time="now", output=path, knob_definitions=knob_definitions)

            # Asserts ---
            parsed_output, _ = parse_output_file(path)
            assert len(all_variables) == len(parsed_output)
            assert len(knobs_extracted) == len(parsed_output)
            for knob in all_variables:
                assert knobs_dict[knob].value == knobs_extracted[knob].value
                assert parsed_output[knobs_extracted[knob].madx] == knobs_extracted[knob].value * knobs_extracted[knob].scaling
    
    @pytest.mark.basic
    @pytest.mark.parametrize("commandline", [True, False], ids=["as function", "cli"])
    def test_state(self, tmp_path, monkeypatch, caplog, commandline):
        # Mock Pytimber ---
        class MyLDB:
            def __init__(self, *args, **kwargs):
                pass
            
            @staticmethod
            def get(key, time):
                if key == "LhcStateTracker:State":
                    return {key: f"The State of the affairs at {time} is good!"}
                else:
                    raise ValueError("This test failed, probably because the StateKey changed. Update Test.")

        class MockTimber:
            LoggingDB = MyLDB

        monkeypatch.setattr(knob_extractor, "pytimber", MockTimber())

        time = datetime.now()
        # Main ---
        with caplog.at_level(logging.INFO):
            if commandline:
                with cli_args("--time", str(time), "--state",
                              script="knob_extract.py"):
                    main()

            else:
                path = tmp_path / "knobs.txt"
                knobs_extracted = main(
                    time=str(time), state=True,
                    output=path, knobs=["fsf"]  # these should not matter, but if state is false
                )
                assert not path.is_file()
                assert knobs_extracted is None

        assert str(time) in caplog.text
        assert "The State of the affairs" in caplog.text

    @pytest.mark.basic
    def test_knob_not_defined(self, knob_definitions, monkeypatch):
        # Mock Pytimber ---
        class MyLDB:
            def __init__(self, *args, **kwargs):
                pass

            @staticmethod
            def get(key, time):
                raise ValueError("This test failed: The code should not have run this far.")

        class MockTimber:
            LoggingDB = MyLDB

        monkeypatch.setattr(knob_extractor, "pytimber", MockTimber())

        # run ---
        knob_name = "non_existent_knob"
        with pytest.raises(KeyError) as e:
            main(knob_definitions=knob_definitions, knobs=[knob_name])
        assert knob_name in str(e)


class TestKnobExtraction:
    @pytest.mark.basic
    def test_extraction(self):
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


class TestIO:
    @pytest.mark.basic
    def test_parse_knobdict_from_file(self, knob_definitions):
        knob_dict = _parse_knobs_defintions(knob_definitions)
        for knobs in KNOB_CATEGORIES.values():
            for knob in knobs:
                assert knob in knob_dict.keys()
                knob_entry = knob_dict[knob]

                assert knob_entry.value is None
                assert abs(knob_entry.scaling) == 1
                assert lsa2name(knob_entry.lsa) == knob
                assert len(knob_entry.madx)
                assert knob_entry.madx in knob_entry.get_madx_command()
                assert knob_entry.get_madx_command().strip().startswith("!")

                knob_entry.value = 10
                assert str(10) in knob_entry.get_madx_command()

    @pytest.mark.basic
    def test_parse_knobdict_from_dataframe(self, tmp_path):
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
    def test_write_file(self, tmp_path):
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


class TestTime:
    @pytest.mark.basic
    def test_time_and_delta(self):
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
    def test_timezones(self):
        assert (
                _parse_time("2022-06-26T03:00+02:00")
                ==
                datetime(2022, 6, 26, 3, 0, 0, tzinfo=timezone(timedelta(seconds=7200)))
        )


class TestParser:
    @pytest.mark.basic
    def test_defaults(self, main_entrypoint):
        opt = main_entrypoint.parse()
        assert isinstance(opt.knobs, list)
        assert opt.time == "now"
        assert opt.timedelta is None
        assert opt.state is False
        assert opt.knob_definitions is None

    @pytest.mark.basic
    def test_cli_parsing(self, main_entrypoint):
        my_opts = dict(
            knobs=["knob1", "knob2", "knob3"],
            time="2022-06-23T12:53:01",
            timedelta="_27y",
            output="help.txt",
            knob_definitions="knob_def.txt",
        )
        my_types = dict(
            knobs=list,
            time=str,
            timedelta=str,
            output=Path,
            knob_definitions=Path,
        )

        # run main
        with cli_args(*dict2args(my_opts)):
            opt = main_entrypoint.parse()

        # check all is correct
        assert all(k in opt.keys() for k in my_opts.keys())
        for k in my_opts.keys():
            assert str(my_opts[k]) == str(opt[k])
            assert isinstance(opt[k], my_types[k])

    @pytest.mark.basic
    def test_time_fail(self, main_entrypoint):
        # we should allow this somewhen in the future
        with pytest.raises(ArgumentError) as e:
            main_entrypoint.parse(time=datetime.now())
        assert "time" in str(e)

    @pytest.mark.basic
    def test_timedelta_fail(self, main_entrypoint):
        with pytest.raises(ArgumentError) as e:
            main_entrypoint.parse(timedelta=-2)
        assert "timedelta" in str(e)

    @pytest.mark.basic
    def test_knobs_fail(self, main_entrypoint):
        with pytest.raises(ArgumentError) as e:
            main_entrypoint.parse(knobs="knob1, knob2, knob3")
        assert "knobs" in str(e)


# CERN Tests -------------------------------------------------------------------

class TestInsideCERNNetwork:
    @pytest.mark.cern_network
    def test_extractor_in_cern_network(self, tmp_path, knob_definitions, saved_knobfile_and_time):
        path_saved, time_saved = saved_knobfile_and_time
        path = tmp_path / "knobs.txt"
        main(time=time_saved, output=path, knob_definitions=knob_definitions)

        parsed_output, _ = parse_output_file(path)
        parsed_saved, _ = parse_output_file(path_saved)

        assert len(parsed_saved) == len(parsed_output)
        for key in parsed_output.keys():
            assert parsed_output[key] == parsed_saved[key]


# Helper -----------------------------------------------------------------------


def parse_output_file(file_path) -> Tuple[Dict[str, float], str]:
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


def dict2args(args_dict: Dict[str, Any]) -> List[str]:
    """ Convert a dictionary to an args-list.
    Keys are flags, values are their arguments. """
    args = []
    for k, v in args_dict.items():
        args.append(f"--{k}")
        if isinstance(v, list):
            args += [str(item) for item in v]
        else:
            args.append(str(v))
    return args


@pytest.fixture()
def knob_definitions() -> Path:
    return INPUTS / "knob_definitions.txt"


@pytest.fixture()
def saved_knobfile_and_time() -> Tuple[Path, str]:
    return INPUTS / "knobs_2022-06-25.txt", "2022-06-25T00:20:00+00:00"


@pytest.fixture()
def main_entrypoint() -> EntryPoint:
    return EntryPoint(get_params(), strict=True)