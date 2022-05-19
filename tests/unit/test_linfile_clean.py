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
        assert unfiltered[plane][COL_NAME][2] not in filtered[plane][COL_NAME]


def test_filter_tune_limit(tmp_path):
    """ Test filtering works on outlier creatd by modify linfiles function. """
    columns = [COL_TUNE]
    plane_columns = [f"{col}{p}" for col in columns for p in PLANES]
    linfiles = _copy_and_modify_linfiles(tmp_path, columns=columns, index=[2], by=0.1)
    unfiltered = {p: tfs.read(f) for p, f in linfiles.items()}

    # if limit not given, would filter two elements in X
    clean_columns(files=linfiles.values(), columns=plane_columns, limit=0.2)

    filtered = {p: tfs.read(f) for p, f in linfiles.items()}

    for plane in PLANES:
        assert_frame_equal(unfiltered[plane], filtered[plane])


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
def test_backup_and_restore(tmp_path):
    """Test that the backup and restore functionality works."""
    columns = [COL_TUNE]
    plane_columns = [f"{col}{p}" for col in columns for p in PLANES]
    linfiles = _copy_and_modify_linfiles(tmp_path)
    unfiltered = {p: tfs.read(f) for p, f in linfiles.items()}

    _assert_nlinfilesfiles(tmp_path, 1)
    clean_columns(files=linfiles.values(), columns=plane_columns, backup=True)
    _assert_nlinfilesfiles(tmp_path, 2)
    clean_columns(files=linfiles.values(), columns=plane_columns, backup=True)
    _assert_nlinfilesfiles(tmp_path, 3)

    filtered = {p: tfs.read(f) for p, f in linfiles.items()}

    restore_files(files=linfiles.values())
    _assert_nlinfilesfiles(tmp_path, 2)
    restore_files(files=linfiles.values())
    _assert_nlinfilesfiles(tmp_path, 1)

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
    main(files=list(linfiles.values()), columns=plane_columns, limit=0.01, backup=True)
    _assert_nlinfilesfiles(tmp_path, 2)

    filtered = {p: tfs.read(f) for p, f in linfiles.items()}

    main(files=list(linfiles.values()), restore=True)
    _assert_nlinfilesfiles(tmp_path, 1)

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
    _assert_nlinfilesfiles(tmp_path, 2)


# Helper -----------------------------------------------------------------------

def _assert_nlinfilesfiles(path, nfiles):
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
