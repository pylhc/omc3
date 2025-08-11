import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import tfs
from generic_parser import EntryPoint
from generic_parser.dict_parser import ArgumentError
from pandas._testing import assert_dict_equal, assert_frame_equal

from omc3 import knob_extractor
from omc3.knob_extractor import (
    KNOB_CATEGORIES,
    STATE_VARIABLES,
    Col,
    Head,
    _add_time_delta,
    _extract_and_gather,
    _parse_knobs_defintions,
    _parse_time,
    _write_knobsfile,
    check_for_undefined_knobs,
    get_madx_command,
    get_params,
    load_knobs_definitions,
    lsa2name,
    main,
)
from tests.conftest import cli_args

INPUTS = Path(__file__).parent.parent / "inputs" / "knob_extractor"


class TestFullRun:
    @pytest.mark.basic
    @pytest.mark.parametrize("commandline", [True, False], ids=["as function", "cli"])
    def test_full(self, tmp_path, knob_definitions, monkeypatch, commandline):
        kobs_defs = _parse_knobs_defintions(knob_definitions)
        all_variables = [knob for category in KNOB_CATEGORIES.values() for knob in category]
        for knob in all_variables:
            value = np.random.random() * 10 - 5
            threshold = np.random.random() < 0.3
            kobs_defs.loc[knob, Col.value] = 0.0 if threshold else value

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
                return {key: [[739173129, 42398328], [-1, kobs_defs.loc[name, Col.value]]]}

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
                knob_entry = kobs_defs.loc[knob, :]
                assert parsed_output[knob_entry[Col.madx]] == knob_entry[Col.value] * knob_entry[Col.scaling]

        else:
            knobs_extracted = main(time="now", output=path, knob_definitions=knob_definitions)

            # Asserts ---
            parsed_output, _ = parse_output_file(path)
            assert len(all_variables) == len(parsed_output)
            assert len(knobs_extracted) == len(parsed_output)
            for knob in all_variables:
                knob_entry = kobs_defs.loc[knob, :]
                assert knob_entry[Col.value] == knobs_extracted.loc[knob, Col.value]
                assert parsed_output[knob_entry[Col.madx]] == knob_entry[Col.value] * knob_entry[Col.scaling]

    @pytest.mark.basic
    @pytest.mark.parametrize("commandline", [True, False], ids=["as function", "cli"])
    def test_state(self, tmp_path, monkeypatch, caplog, commandline):
        returns = {v: np.random.random() for v in STATE_VARIABLES}

        # Mock Pytimber ---
        class MyLDB:
            def __init__(self, *args, **kwargs):
                pass

            @staticmethod
            def get(key, time):
                intro = "LhcStateTracker:State:"
                if key.startswith(intro):
                    return {key: ([time], [returns[key[len(intro):]]])}
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
                state_extracted = main(
                    time=str(time), state=True,
                    output=path, knobs=["fsf"]  # these should not matter, but if state is false
                )
                assert not path.is_file()
                assert len(state_extracted)
                for name in state_extracted.index:
                    value = state_extracted.loc[name, Col.value]
                    lsa_name = state_extracted.loc[name, Col.lsa]
                    assert value == returns[lsa_name]
                    assert len(re.findall(fr"{name}:\s*{str(value)}", caplog.text)) == 1

    @pytest.mark.basic
    def test_knob_not_defined_run(self, knob_definitions, monkeypatch):
        # Mock Pytimber ---
        class MyLDB:
            def __init__(self, *args, **kwargs):
                pass

            @staticmethod
            def get(key, time):
                raise ArgumentError("Got past the KnobCheck!")
                # return {key: [[478973], [343.343]]}

        class MockTimber:
            LoggingDB = MyLDB

        monkeypatch.setattr(knob_extractor, "pytimber", MockTimber())
        knob_definitions_df = load_knobs_definitions(knob_definitions)

        # run ------------------------------------------------------------------
        knobs_undefined = ["non_existent_knob", "other_knob"]
        knobs_defined = knob_definitions_df.index.tolist()

        # undefined only ---
        with pytest.raises(KeyError) as e:
            main(knob_definitions=knob_definitions, knobs=knobs_undefined)

        for knob in knobs_undefined:
            assert knob in str(e)

        # defined only ---
        with pytest.raises(ArgumentError) as e:
            main(knob_definitions=knob_definitions, knobs=knobs_defined)
        assert "KnobCheck" in str(e)  # see mock Pytimber above

        # both ---
        with pytest.raises(KeyError) as e:
            main(knob_definitions=knob_definitions, knobs=knobs_undefined+knobs_defined)

        for knob in knobs_undefined:
            assert knob in str(e)


class TestKnobExtraction:
    @pytest.mark.basic
    def test_extraction(self):
        knobs_dict = pd.DataFrame({
            "LHCBEAM1:LANDAU_DAMPING": knob_def(madx="landau1", lsa="LHCBEAM1/LANDAU_DAMPING", scaling=-1),
            "LHCBEAM2:LANDAU_DAMPING": knob_def(madx="landau2", lsa="LHCBEAM1/LANDAU_DAMPING", scaling=-1),
            "other": knob_def(madx="other_knob", lsa="other/knob", scaling=1),
        }).transpose()
        values = [8904238, 34.323904, 3489.23409]
        time = datetime.now()
        timestamp = time.timestamp()*1e9  # java format

        fake_ldb = {
            f"LhcStateTracker:{key}:target": {f"LhcStateTracker:{key}:target": [[timestamp, timestamp], [-1, value]]}
            for key, value in zip(knobs_dict.index, values)
        }

        extracted = _extract_and_gather(fake_ldb, knobs_definitions=knobs_dict, knob_categories=["mo", "other"], time=time)

        assert len(extracted) == len(knobs_dict)
        for idx, (key, entry) in enumerate(extracted.iterrows()):
            assert entry[Col.value] == values[idx]  # depends on the order of "mo" in the file


class TestIO:
    @pytest.mark.basic
    def test_parse_knobdefs_from_file(self, knob_definitions):
        knob_defs = _parse_knobs_defintions(knob_definitions)
        for knobs in KNOB_CATEGORIES.values():
            for knob in knobs:
                assert knob in knob_defs.index
                knob_entry = knob_defs.loc[knob, :].copy()

                assert abs(knob_entry[Col.scaling]) == 1
                assert lsa2name(knob_entry[Col.lsa]) == knob
                assert len(knob_entry[Col.madx])

                with pytest.raises(KeyError) as e:
                    get_madx_command(knob_entry)
                assert "Value entry not found" in str(e)

                knob_entry[Col.value] = pd.NA
                madx_command = get_madx_command(knob_entry)
                assert knob_entry[Col.madx] in madx_command
                assert madx_command.strip().startswith("!")

                knob_entry[Col.value] = 10
                madx_command = get_madx_command(knob_entry)
                assert str(10) in madx_command

    @pytest.mark.basic
    def test_parse_knobdict_from_dataframe(self, tmp_path):
        df = pd.DataFrame(data=[["madx_name", "lsa_name", 1., "something"]], columns=["madx", "lsa", "scaling", "other"])

        path = tmp_path / "knob_defs.tfs"
        tfs.write(path, df)

        knob_defs = _parse_knobs_defintions(path)
        assert len(knob_defs) == 1
        assert "lsa_name" in knob_defs.index
        knob_entry = knob_defs.loc["lsa_name", :]
        assert knob_entry[Col.lsa] == "lsa_name"
        assert knob_entry[Col.madx] == "madx_name"
        assert knob_entry[Col.scaling] == 1

    @pytest.mark.basic
    def test_write_file(self, tmp_path):
        knobs_defs = pd.DataFrame({
            "LHCBEAM1:LANDAU_DAMPING": knob_def(madx="moknob1", lsa="moknob1.lsa", scaling=-1, value=-4783),
            "LHCBEAM2:LANDAU_DAMPING": knob_def(madx="moknob2", lsa="moknob2.lsa", scaling=1, value=0.0),  # one should be 0.0 to test this case
            "knob1": knob_def(madx="knob1.madx", lsa="knob1.lsa", scaling=-1, value=12.43383),
            "knob2": knob_def(madx="knob2.madx", lsa="knob2.lsa", scaling=1, value=-3.0231),
            "knob3": knob_def(madx="knob3.madx", lsa="knob3.lsa", scaling=-1, value=-9.7492),
        }).transpose()
        path = tmp_path / "knobs.txt"
        time = datetime.now()
        knobs_defs = tfs.TfsDataFrame(knobs_defs, headers={Head.time: time})
        _write_knobsfile(path, knobs_defs)
        read_as_dict, full_text = parse_output_file(path)
        assert str(time) in full_text
        assert " mo " in full_text
        assert " Other Knobs " in full_text
        assert len(read_as_dict) == len(knobs_defs)
        for _, entry in knobs_defs.iterrows():
            assert read_as_dict[entry.madx] == entry[Col.value] * entry[Col.scaling]

    @pytest.mark.basic
    def test_knob_not_defined(self, knob_definitions, monkeypatch):
        knob_definitions_df = load_knobs_definitions(knob_definitions)

        # run ------------------------------------------------------------------
        knobs_undefined = ["this_knob_does_not_exist", "Knobby_McKnobface"]
        knobs_defined = knob_definitions_df.index.tolist()
        knob_categories = list(KNOB_CATEGORIES.keys())

        # undefined only ---
        with pytest.raises(KeyError) as e:
            check_for_undefined_knobs(knob_definitions_df, knobs_undefined)

        for knob in knobs_undefined:
            assert knob in str(e)

        # defined only ---
            check_for_undefined_knobs(knob_definitions_df, knobs_defined)
            check_for_undefined_knobs(knob_definitions_df, knob_categories)
            check_for_undefined_knobs(knob_definitions_df, knobs_defined + knob_categories)

        # all ---
        with pytest.raises(KeyError) as e:
            check_for_undefined_knobs(knob_definitions_df,
                                      knob_categories + knobs_undefined + knobs_defined)

        for knob in knobs_undefined:
            assert knob in str(e)

    @pytest.mark.basic
    def test_load_knobdefinitions_with_any_number_entries(self, tmp_path):
        definition_file = tmp_path / "knob_defs_tmp.txt"
        values = [18.8, 12.0, 10, 108.8]
        definition_file.write_text(
            f"knob1_madx, knob1/lsa, {values[0]}, 19.8, 38\n"
            f"knob2_madx, knob2/lsa, {values[1]}, 483.8\n"
            f"knob3_madx, knob3/lsa, {values[2]}\n"
            f"knob4_madx, knob4/lsa, {values[3]}, 19.8, other stuff\n"
        )

        df = load_knobs_definitions(definition_file)
        assert len(df) == len(values)

        for idx, value in enumerate(values, start=1):
            name = f"knob{idx}:lsa"
            assert name in df.index
            assert df.loc[name, Col.scaling] == value
            assert df.loc[name, Col.madx] == f"knob{idx}_madx"
            assert df.loc[name, Col.lsa] == f"knob{idx}/lsa"

    @pytest.mark.basic
    def test_load_knobdefinitions_fail_no_scaling(self, tmp_path):
        definition_file = tmp_path / "knob_defs_tmp.txt"
        definition_file.write_text(
            "knob1_madx, knob1/lsa\n"
            "knob2_madx, knob2/lsa\n"
        )

        with pytest.raises(pd.errors.ParserError) as e:
            load_knobs_definitions(definition_file)
        assert "expected 3 and found 2" in str(e)

    @pytest.mark.basic
    def test_load_knobdefinitions_fail_wrong_scaling(self, tmp_path):
        definition_file = tmp_path / "knob_defs_tmp.txt"
        definition_file.write_text("knob1_madx, knob1/lsa, wrong\n")

        # with pytest.raises(pd.errors.ParserError):
        with pytest.raises(ValueError) as e:
            load_knobs_definitions(definition_file)
        assert "could not convert string to float" in str(e)


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
        opt = main_entrypoint.parse([])
        assert isinstance(opt.knobs, list)
        assert opt.time == "now"
        assert opt.timedelta is None
        assert opt.state is False
        assert opt.knob_definitions is None

    @pytest.mark.basic
    def test_cli_parsing(self, main_entrypoint):
        my_opts = {
            "knobs": ["knob1", "knob2", "knob3"],
            "time": "2022-06-23T12:53:01",
            "timedelta": "_27y",
            "output": "help.txt",
            "knob_definitions": "knob_def.txt",
        }
        my_types = {
            "knobs": list,
            "time": str,
            "timedelta": str,
            "output": Path,
            "knob_definitions": Path,
        }

        # run main
        with cli_args(*dict2args(my_opts)):
            opt = main_entrypoint.parse()

        # check all is correct
        assert all(k in opt for k in my_opts)
        for k in my_opts:
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
        for key in parsed_output:
            assert parsed_output[key] == parsed_saved[key]

    @pytest.mark.cern_network
    def test_state_in_cern_network(self, state_tfs):
        # get recorded data
        old_df = tfs.read_tfs(state_tfs, index="NAME")
        time = old_df.headers[Head.time]

        # run main
        state_df = main(time=time, state=True)

        # format a bit to make frames equal
        state_df = state_df.applymap(str)
        state_df.headers[Head.time] = str(state_df.headers[Head.time])
        state_df.index.name = "NAME"

        # check frames
        assert_dict_equal(state_df.headers, old_df.headers)
        assert_frame_equal(state_df, old_df)


# Helper -----------------------------------------------------------------------


def knob_def(**kwargs):
    return pd.Series(dict(**kwargs))


def parse_output_file(file_path) -> tuple[dict[str, float], str]:
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


def dict2args(args_dict: dict[str, Any]) -> list[str]:
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
def state_tfs() -> Path:
    return INPUTS / "state_2022-06-25.tfs"


@pytest.fixture()
def saved_knobfile_and_time() -> tuple[Path, str]:
    return INPUTS / "knobs_2022-06-25.txt", "2022-06-25T00:20:00+00:00"


@pytest.fixture()
def main_entrypoint() -> EntryPoint:
    return EntryPoint(get_params(), strict=True)
