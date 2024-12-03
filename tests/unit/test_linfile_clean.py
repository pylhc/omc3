import os
import shutil
from pathlib import Path
from pandas.testing import assert_frame_equal

from omc3.definitions.constants import PLANES
from omc3.harpy.constants import COL_TUNE, COL_NATTUNE, COL_NAME
from omc3.scripts.linfile_clean import main, clean_columns, restore_files


import tfs
import pytest

INPUT_DIR = Path(__file__).parent.parent / "inputs"


@pytest.mark.basic
def test_filter_tune(tmp_path):
    """ Test filtering works on outlier created by modify linfiles function. """
    columns = [COL_TUNE]
    plane_columns = [f"{col}{p}" for col in columns for p in PLANES]
    linfiles = _copy_and_modify_linfiles(tmp_path, columns=columns, index=[2], by=0.1)
    unfiltered = {p: tfs.read(f) for p, f in linfiles.items()}

    # if limit not given, would filter two elements in X
    clean_columns(files=linfiles.values(), columns=plane_columns, limit=0.01)

    filtered = {p: tfs.read(f) for p, f in linfiles.items()}

    for plane in PLANES:
        assert len(filtered[plane]) == 4
        assert unfiltered[plane][COL_NAME][2] not in filtered[plane][COL_NAME].to_list()


def test_filter_tune_limit(tmp_path):
    """ Test filtering works on outlier created by modify linfiles function. """
    columns = [COL_TUNE]
    plane_columns = [f"{col}{p}" for col in columns for p in PLANES]
    linfiles = _copy_and_modify_linfiles(tmp_path, columns=columns, index=[2], by=0.1)
    unfiltered = {p: tfs.read(f) for p, f in linfiles.items()}

    # choose limit greater than the changes made
    clean_columns(files=linfiles.values(), columns=plane_columns, limit=0.2)

    filtered = {p: tfs.read(f) for p, f in linfiles.items()}

    for plane in PLANES:
        assert_frame_equal(unfiltered[plane], filtered[plane])


@pytest.mark.basic
def test_keep_bpms(tmp_path):
    """ Test that keeping BPMs works. """
    columns = [COL_TUNE]
    plane_columns = [f"{col}{p}" for col in columns for p in PLANES]
    
    # To be filtered BPMS are (due to the values in the example linfiles)
    filtered_bpms = {
        "X": ["BPM.10L4.B1", "BPM.10L2.B1"],
        "Y": ["BPM.10L1.B1", "BPM.10L2.B1"],
    }

    # Test that all BPMs are filtered without the keep-flag --------------------
    linfiles = _copy_and_modify_linfiles(tmp_path, columns=columns)
    unfiltered = {p: tfs.read(f) for p, f in linfiles.items()}

    # if limit not given, filters two elements in X
    clean_columns(files=linfiles.values(), columns=plane_columns)
    filtered = {p: tfs.read(f) for p, f in linfiles.items()}
    

    for plane in PLANES:
        assert len(filtered[plane]) == len(unfiltered[plane]) - 2
        for bpm in filtered_bpms[plane]:
            assert bpm not in filtered[plane][COL_NAME].to_list()
            assert bpm in unfiltered[plane][COL_NAME].to_list()

    # Now with keeping one of them ---------------------------------------------
    linfiles = _copy_and_modify_linfiles(tmp_path, columns=columns)
    unfiltered = {p: tfs.read(f) for p, f in linfiles.items()}

    # if limit not given, filters two elements in X
    clean_columns(files=linfiles.values(), columns=plane_columns, keep=[filtered_bpms["X"][1]])
    filtered = {p: tfs.read(f) for p, f in linfiles.items()}
    for plane in PLANES:
        assert len(filtered[plane]) == len(unfiltered[plane]) - 1
        for bpm in filtered_bpms[plane]:
            assert bpm in unfiltered[plane][COL_NAME].to_list()

        assert filtered_bpms[plane][0] not in filtered[plane][COL_NAME].to_list()
        assert filtered_bpms[plane][1] in filtered[plane][COL_NAME].to_list()


@pytest.mark.basic
def test_filter_tune_nattune(tmp_path):
    """Tests that filtering works for two columns."""
    columns = [COL_TUNE, COL_NATTUNE]
    plane_columns = [f"{col}{p}" for col in columns for p in PLANES]
    linfiles = _copy_and_modify_linfiles(tmp_path)
    # unfiltered = {p: tfs.read(f) for p, f in linfiles.items()}

    clean_columns(files=linfiles.values(), columns=plane_columns)

    filtered = {p: tfs.read(f) for p, f in linfiles.items()}

    for plane in PLANES:
        assert len(filtered[plane]) == 2  # empirically determined


@pytest.mark.basic
def test_filter_between_limits(tmp_path):
    """ Test filtering works on outlier created by modify linfiles function. """
    columns = [COL_TUNE]
    plane_columns = [f"{col}{p}" for col in columns for p in PLANES]
    
    # Test that no BPMs are filtered by the auto-clean (sanity check) ----------
    linfiles = _copy_and_modify_linfiles(tmp_path, columns=columns, index=[2, 3], by=0.1)
    unfiltered = {p: tfs.read(f) for p, f in linfiles.items()}

    clean_columns(files=linfiles.values(), columns=plane_columns)

    filtered = {p: tfs.read(f) for p, f in linfiles.items()}

    for plane in PLANES:
        assert_frame_equal(unfiltered[plane], filtered[plane])
    
    # Test that the two BPMs are filtered by the limits-clean ------------------
    linfiles = _copy_and_modify_linfiles(tmp_path, columns=columns, index=[2, 3], by=0.1)
    unfiltered = {p: tfs.read(f) for p, f in linfiles.items()}
    
    # choosing values so that both planes are filtered
    # X tunes are 0.26 + 0.1, Y tunes are 0.32 + 0.1
    clean_columns(files=linfiles.values(), columns=plane_columns, limit=(0.20, 0.35))

    filtered = {p: tfs.read(f) for p, f in linfiles.items()}

    for plane in PLANES:
        assert len(filtered[plane]) == 3
        assert unfiltered[plane][COL_NAME][2] not in filtered[plane][COL_NAME].to_list()
        assert unfiltered[plane][COL_NAME][3] not in filtered[plane][COL_NAME].to_list()
    
    
    # Test that keep flag is also respected in the limits-clean ----------------
    linfiles = _copy_and_modify_linfiles(tmp_path, columns=columns, index=[2, 3], by=0.1)
    unfiltered = {p: tfs.read(f) for p, f in linfiles.items()}
    
    # choosing values so that both planes are filtered
    # X tunes are 0.26 + 0.1, Y tunes are 0.32 + 0.1
    clean_columns(files=linfiles.values(), columns=plane_columns, limit=(0.20, 0.35), keep=[unfiltered["X"][COL_NAME][2]])

    filtered = {p: tfs.read(f) for p, f in linfiles.items()}

    for plane in PLANES:
        assert len(filtered[plane]) == 4
        assert unfiltered[plane][COL_NAME][2] in filtered[plane][COL_NAME].to_list()
        assert unfiltered[plane][COL_NAME][3] not in filtered[plane][COL_NAME].to_list()


@pytest.mark.basic
def test_backup_and_restore(tmp_path):
    """Test that the backup and restore functionality works."""
    columns = [COL_TUNE]
    plane_columns = [f"{col}{p}" for col in columns for p in PLANES]
    linfiles = _copy_and_modify_linfiles(tmp_path)
    unfiltered = {p: tfs.read(f) for p, f in linfiles.items()}

    _assert_nlinfiles(tmp_path, 1)
    clean_columns(files=linfiles.values(), columns=plane_columns, backup=True)
    _assert_nlinfiles(tmp_path, 2)
    clean_columns(files=linfiles.values(), columns=plane_columns, backup=True)
    _assert_nlinfiles(tmp_path, 3)

    filtered = {p: tfs.read(f) for p, f in linfiles.items()}

    restore_files(files=linfiles.values())
    _assert_nlinfiles(tmp_path, 2)
    restore_files(files=linfiles.values())
    _assert_nlinfiles(tmp_path, 1)

    restored = {p: tfs.read(f) for p, f in linfiles.items()}

    for plane in PLANES:
        assert_frame_equal(unfiltered[plane], restored[plane])
        with pytest.raises(AssertionError):
            assert_frame_equal(unfiltered[plane], filtered[plane])

    with pytest.raises(IOError):
        restore_files(files=linfiles.values())


@pytest.mark.basic
def test_main(tmp_path):
    """Test basically all functionality as above, but going through `main`."""
    os.chdir(tmp_path)
    columns = [COL_TUNE]
    plane_columns = [f"{col}{p}" for col in columns for p in PLANES]
    linfiles = _copy_and_modify_linfiles(tmp_path, columns=columns, index=[2], by=0.1)
    unfiltered = {p: tfs.read(f) for p, f in linfiles.items()}

    # if limit not given, would filter two elements in X
    main(files=list(linfiles.values()), columns=plane_columns, limit=[0.01], backup=True)
    _assert_nlinfiles(tmp_path, 2)

    filtered = {p: tfs.read(f) for p, f in linfiles.items()}

    main(files=list(linfiles.values()), restore=True)
    _assert_nlinfiles(tmp_path, 1)

    restored = {p: tfs.read(f) for p, f in linfiles.items()}

    # Test that the data makes sense
    for plane in PLANES:
        assert len(filtered[plane]) == 4
        assert unfiltered[plane][COL_NAME][2] not in filtered[plane][COL_NAME]

        assert_frame_equal(unfiltered[plane], restored[plane])
        with pytest.raises(AssertionError):
            assert_frame_equal(unfiltered[plane], filtered[plane])

    # Tests that inis were written and they are usable for re-running
    inis = list(Path('.').glob("*.ini"))
    assert len(inis) == 2
    main(entry_cfg=sorted(inis)[0])
    assert len(list(Path('.').glob("*.ini"))) == 3
    _assert_nlinfiles(tmp_path, 2)


# Helper -----------------------------------------------------------------------

def _assert_nlinfiles(path, nfiles):
    for plane in PLANES:
        assert len(list(path.glob(f"*.lin{plane.lower()}*"))) == nfiles


def _copy_and_modify_linfiles(out_path: Path, columns=None, index=None, by=0.0):
    paths = {}
    for plane in PLANES:
        lin_file_src = _get_inputs_linfile(plane)
        lin_file_dst = out_path / lin_file_src.name
        paths[plane] = lin_file_dst
        if index is not None and columns is not None:
            plane_columns = [f"{col}{plane}" for col in columns]
            df = tfs.read(lin_file_src)
            df.loc[index, plane_columns] = df.loc[index, plane_columns] + by
            tfs.write_tfs(lin_file_dst, df)
        else:
            shutil.copy(lin_file_src, lin_file_dst)
    return paths


def _get_inputs_linfile(plane: str):
    return INPUT_DIR / f"spec_test.sdds.lin{plane.lower()}"
