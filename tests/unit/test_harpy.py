import pytest
import turn_by_turn as tbt
from generic_parser import DotDict

from omc3.hole_in_one import _add_suffix_and_loop_over_bunches


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
    