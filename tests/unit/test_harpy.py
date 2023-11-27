from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import tfs
import turn_by_turn as tbt
from generic_parser import DotDict

from omc3.hole_in_one import _add_suffix_and_loop_over_bunches, hole_in_one_entrypoint
from tests.accuracy.test_harpy import _get_model_dataframe


@pytest.mark.basic
@pytest.mark.parametrize("suffix", ("_my_suffix", None))
def test_input_suffix_and_single_bunch(suffix):
    """ Tests the function :func:`omc3.hole_in_one._add_suffix_and_loop_over_bunches` 
    by checking that the suffix is attached to single-bunch files."""
    file_name = "input_file.sdds"
    options = DotDict(
        files=file_name,
        suffix=suffix,
        bunches=None,
    )
    tbt_data = tbt.TbtData(
        nturns=1,
        matrices=[1],
        bunch_ids=[0],
    )
    n_data = 0
    for data, opt in _add_suffix_and_loop_over_bunches(tbt_data, options):
        suffix_str = suffix or ""
        assert opt.files.endswith(f"{file_name}{suffix_str}")
        assert "bunchID" not in opt.files
        assert data is tbt_data
        n_data += 1
    
    assert n_data == 1


@pytest.mark.basic
@pytest.mark.parametrize("suffix", ("_my_suffix", None))
@pytest.mark.parametrize("bunches", (None, (1, 15)))
def test_input_suffix_and_multibunch(suffix, bunches):
    """ Tests the function :func:`omc3.hole_in_one._add_suffix_and_loop_over_bunches` 
    by checking that the suffixes are attached to multi-bunch files and they are
    split up into single-bunch files correctly."""
    file_name = "input_file.sdds"
    options = DotDict(
        files=file_name,
        suffix=suffix,
        bunches=None if bunches is None else list(bunches),
    )
    tbt_data = tbt.TbtData(
        nturns=1,
        matrices=[1, 2, 3],
        bunch_ids=[1, 10, 15],
    )
    n_data = 0
    bunch_ids = bunches or tbt_data.bunch_ids
    matrices =  [tbt_data.matrices[tbt_data.bunch_ids.index(id_)] for id_ in bunch_ids]
    for (data, opt), bunch_id, matrix in zip(_add_suffix_and_loop_over_bunches(tbt_data, options), bunch_ids, matrices):
        bunch_str = f"_bunchID{bunch_id}"
        suffix_str = suffix or ""
        assert opt is not options
        assert opt.files.endswith(f"{file_name}{bunch_str}{suffix_str}")

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
    """ Runs harpy and checks that the right files are created. 
    
    Only with bunchID as we have enough tests in the accuracy tests, 
    that implicitly check that the single-bunch files are created.
    """
    all_bunches = [1, 5, 15]
    tbt_file = tmp_path / "test_file.sdds"

    # Mock some TbT data ---
    model = _get_model_dataframe()
    tbt.write(tbt_file, create_tbt_data(model=model, bunch_ids=all_bunches))

    # Run harpy ---
    hole_in_one_entrypoint(harpy=True,
                           clean=False,
                           autotunes="transverse",
                           outputdir=str(tmp_path),
                           files=[tbt_file],
                           to_write=["lin", "spectra"],
                           turn_bits=4,  # make it fast
                           output_bits=4,
                           unit="m",
                           suffix=suffix,
                           bunches=None if bunches is None else list(bunches),
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


# Helper ---

def create_tbt_data(model: pd.DataFrame, bunch_ids: Sequence[int] = (0, ), n_turns: int = 10) -> tbt.TbtData:
    """Create simple turn-by-turn data based on the given model.

    Args:
        model (pd.DataFrame): Model to base the turn-by-turn data on
        bunch_ids (Sequence[int], optional): Which bunces to create. The data is the same for all bunches. Defaults to (0, ).
        n_turns (int, optional): How many turns to create. Defaults to 10.

    Returns:
        tbt.TbtData: Created TbtData
    """
    ints = np.arange(n_turns) - n_turns / 2
    data_x = model.loc[:, "AMPX"].to_numpy()[:, None] * np.cos(2 * np.pi * (model.loc[:, "MUX"].to_numpy()[:, None] + model.loc[:, "TUNEX"].to_numpy()[:, None] * ints[None, :]))
    data_y = model.loc[:, "AMPY"].to_numpy()[:, None] * np.cos(2 * np.pi * (model.loc[:, "MUY"].to_numpy()[:, None] + model.loc[:, "TUNEY"].to_numpy()[:, None] * ints[None, :]))
    matrix = tbt.TransverseData(X=pd.DataFrame(data=data_x, index=model.index), Y=pd.DataFrame(data=data_y, index=model.index))
    return tbt.TbtData(matrices=[matrix] * len(bunch_ids), bunch_ids=list(bunch_ids), nturns=n_turns)  
