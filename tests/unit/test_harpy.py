from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import tfs
import turn_by_turn as tbt
from generic_parser import DotDict
from pandas.testing import assert_frame_equal
from tfs.testing import assert_dict_equal

from omc3.harpy import _parallel, handler
from omc3.hole_in_one import _add_suffix_and_iter_bunches, hole_in_one_entrypoint
from tests.accuracy.test_harpy import _get_model_dataframe

if TYPE_CHECKING:
    from tfs.frame import TfsDataFrame


@pytest.mark.basic
@pytest.mark.parametrize("suffix", ("_my_suffix", None))
def test_input_suffix_and_single_bunch(suffix):
    """Tests the function :func:`omc3.hole_in_one._add_suffix_and_loop_over_bunches`
    by checking that the suffix is attached to single-bunch files."""
    input_name = "input_file.sdds"
    options = DotDict(
        suffix=suffix,
        bunch_ids=None,
    )
    tbt_data = tbt.TbtData(
        nturns=1,
        matrices=[1],
        bunch_ids=[0],
    )
    n_data = 0
    for data, file_name in _add_suffix_and_iter_bunches(tbt_data, options, input_name):
        suffix_str = suffix or ""
        assert file_name == f"{input_name}{suffix_str}"
        assert "bunchID" not in input_name
        assert data is tbt_data
        n_data += 1

    assert n_data == 1


@pytest.mark.basic
@pytest.mark.parametrize("suffix", ("_my_suffix", None))
@pytest.mark.parametrize("bunches", (None, (1, 15)))
def test_input_suffix_and_multibunch(suffix, bunches):
    """Tests the function :func:`omc3.hole_in_one._add_suffix_and_loop_over_bunches`
    by checking that the suffixes are attached to multi-bunch files and they are
    split up into single-bunch files correctly."""
    input_name = "input_file.sdds"
    options = DotDict(
        suffix=suffix,
        bunch_ids=None if bunches is None else list(bunches),
    )
    tbt_data = tbt.TbtData(
        nturns=1,
        matrices=[1, 2, 3],
        bunch_ids=[1, 10, 15],
    )
    n_data = 0
    bunch_ids = bunches or tbt_data.bunch_ids
    matrices = [tbt_data.matrices[tbt_data.bunch_ids.index(id_)] for id_ in bunch_ids]
    for (data, filename_with_suffix), bunch_id, matrix in zip(
        _add_suffix_and_iter_bunches(tbt_data, options, input_name), bunch_ids, matrices
    ):
        bunch_str = f"_bunchID{bunch_id}"
        suffix_str = suffix or ""
        assert filename_with_suffix == f"{input_name}{bunch_str}{suffix_str}"

        assert len(data.matrices) == 1
        assert data.matrices[0] == matrix
        assert data.bunch_ids[0] == bunch_id
        n_data += 1

    if bunches:
        assert n_data == len(bunches)
    else:
        assert n_data == len(tbt_data.bunch_ids)


@pytest.mark.extended
@pytest.mark.parametrize("suffix", ("_my_suffix", None))
@pytest.mark.parametrize("bunches", (None, (1, 15)))
def test_harpy_with_suffix_and_bunchid(tmp_path, suffix, bunches):
    """Runs harpy and checks that the right files are created.

    Only with bunchID as we have enough tests in the accuracy tests,
    that implicitly check that the single-bunch files are created.
    """
    all_bunches = [1, 5, 15]
    tbt_file = tmp_path / "test_file.sdds"

    # Mock some TbT data ---
    model = _get_model_dataframe()
    tbt.write(tbt_file, create_tbt_data(model=model, bunch_ids=all_bunches))

    # Run harpy ---
    hole_in_one_entrypoint(
        harpy=True,
        clean=False,
        autotunes="transverse",
        outputdir=str(tmp_path),
        files=[tbt_file],
        to_write=["lin", "spectra"],
        turn_bits=4,  # make it fast
        output_bits=4,
        unit="m",
        suffix=suffix,
        bunch_ids=None if bunches is None else list(bunches),
    )

    # Check that the right files are created ---
    exts = [".lin", ".freqs", ".amps"]
    suffix_str = suffix or ""
    for bunch in all_bunches:
        for ext in exts:
            for plane in "xy":
                file_path = Path(f"{tbt_file!s}_bunchID{bunch}{suffix_str}{ext}{plane}")
                if bunches is None or bunch in bunches:
                    assert file_path.is_file()
                    tfs.read(file_path)
                else:
                    assert not file_path.is_file()


@pytest.mark.basic
@pytest.mark.parametrize("n_jobs", (0, 2))  # keep njobs not too high for CI
@pytest.mark.parametrize("to_write", (["lin"], ["lin", "full_spectra"]))
def test_harpy_parallel_matches_serial(tmp_path, n_jobs, to_write):
    """
    The auto (n_jobs=0) and forced-pool (n_jobs=2) paths must produce output files
    identical to the serial (n_jobs=1) run: only the orchestration changes, not the maths.
    The ``full_spectra`` case (default from the GUI in the CCC) writes out additional files
    as``.amps``/``.freqs`` which we will check too.
    """
    model = _get_model_dataframe()
    bunch_ids = [1, 5, 15]

    serial_dir = tmp_path / "serial"
    serial_dir.mkdir()
    parallel_dir = tmp_path / "parallel"
    parallel_dir.mkdir()

    # Run it the old way, serial calculation per bunch
    serial_file = _run_harpy_multibunch(serial_dir, model, bunch_ids, n_jobs=1, to_write=to_write)

    # Run it the "new" way, parallelising over bunches
    parallel_file = _run_harpy_multibunch(
        parallel_dir, model, bunch_ids, n_jobs=n_jobs, to_write=to_write
    )

    # Check lin files always, and spectra files (.amps/.freqs) when full spectra were written
    extensions: list[str] = [".lin"]
    if "full_spectra" in to_write:
        extensions += [".amps", ".freqs"]

    # Now we compare results for each bunch
    for bunch in bunch_ids:
        for plane in "xy":  # there's output per plane
            for (
                ext
            ) in extensions:  # and we check all relevant output files (potentially incl. spectra)
                serial: TfsDataFrame = tfs.read(f"{serial_file}_bunchID{bunch}{ext}{plane}")
                parallel: TfsDataFrame = tfs.read(f"{parallel_file}_bunchID{bunch}{ext}{plane}")
                assert_frame_equal(serial, parallel)
                # Exclude TIME header, which is the (wall-clock) timestamp of each run
                serial_headers = {k: v for k, v in serial.headers.items() if k != "TIME"}
                parallel_headers = {k: v for k, v in parallel.headers.items() if k != "TIME"}
                assert_dict_equal(serial_headers, parallel_headers)


@pytest.mark.basic
def test_harpy_ram_clamp_forces_serial(tmp_path, monkeypatch):
    """
    A tiny available-RAM reading must clamp the automatic pool down to a single
    worker, and the run must still complete and produce the expected lin files.
    """
    monkeypatch.setattr(_parallel, "available_ram_bytes", lambda: int(1e6))  # 1 MB RAM
    model = _get_model_dataframe()
    bunch_ids = [1, 5, 15]
    tbt_file = _run_harpy_multibunch(tmp_path, model, bunch_ids, n_jobs=0)
    for bunch in bunch_ids:
        for plane in "xy":
            assert Path(f"{tbt_file}_bunchID{bunch}.lin{plane}").is_file()


@pytest.mark.basic
def test_analyse_no_tasks_returns_empty():
    """
    With no bunches to process, the orchestrator should return early with
    an empty list, without computing a strategy or starting a pool.
    """
    assert handler.analyse_bunches_parallel([], DotDict(n_jobs=0)) == []


@pytest.mark.basic
def test_harpy_negative_n_jobs_is_rejected(tmp_path):
    """
    A negative --n_jobs is rejected up front by the harpy entrypoint validation,
    before any file is read (so the input path need not exist).
    """
    with pytest.raises(AttributeError, match="n_jobs must be >= 0"):
        hole_in_one_entrypoint(
            harpy=True,
            autotunes="transverse",
            outputdir=str(tmp_path),
            files=["does_not_need_to_exist.sdds"],
            n_jobs=-1,
        )


# Helper ---


def _run_harpy_multibunch(
    dirpath: Path,
    model: pd.DataFrame,
    bunch_ids,
    n_jobs: int,
    to_write: Sequence[str] = ("lin",),
) -> Path:
    """Write a multi-bunch tbt file and run harpy on it with the given ``n_jobs``."""
    tbt_file = dirpath / "test_file.sdds"
    tbt.write(tbt_file, create_tbt_data(model=model, bunch_ids=bunch_ids, n_turns=512))
    hole_in_one_entrypoint(
        harpy=True,
        clean=True,
        autotunes="transverse",
        outputdir=str(dirpath),
        files=[tbt_file],
        to_write=list(to_write),
        turn_bits=10,
        output_bits=8,
        unit="m",
        n_jobs=n_jobs,
    )
    return tbt_file


def create_tbt_data(
    model: pd.DataFrame, bunch_ids: Sequence[int] = (0,), n_turns: int = 10
) -> tbt.TbtData:
    """Create simple turn-by-turn data based on the given model.

    Args:
        model (pd.DataFrame): Model to base the turn-by-turn data on
        bunch_ids (Sequence[int], optional): Which bunces to create. The data is the same for all bunches. Defaults to (0, ).
        n_turns (int, optional): How many turns to create. Defaults to 10.

    Returns:
        tbt.TbtData: Created TbtData
    """
    # fmt: off
    ints = np.arange(n_turns) - n_turns / 2
    data_x = model.loc[:, "AMPX"].to_numpy()[:, None] * np.cos(2 * np.pi * (model.loc[:, "MUX"].to_numpy()[:, None] + model.loc[:, "TUNEX"].to_numpy()[:, None] * ints[None, :]))
    data_y = model.loc[:, "AMPY"].to_numpy()[:, None] * np.cos(2 * np.pi * (model.loc[:, "MUY"].to_numpy()[:, None] + model.loc[:, "TUNEY"].to_numpy()[:, None] * ints[None, :]))
    matrix = tbt.TransverseData(X=pd.DataFrame(data=data_x, index=model.index), Y=pd.DataFrame(data=data_y, index=model.index))
    # fmt: on
    return tbt.TbtData(
        matrices=[matrix] * len(bunch_ids), bunch_ids=list(bunch_ids), nturns=n_turns
    )
