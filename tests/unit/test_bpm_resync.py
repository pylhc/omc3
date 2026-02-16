import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tfs
import turn_by_turn as tbt
from generic_parser.dict_parser import ArgumentError

from omc3.scripts import resync_bpms as resync

INPUTS_DIR = Path(__file__).parent.parent / "inputs" / "bpm_resync"
OPTICS_DIR = Path(__file__).parent.parent / "inputs" / "bpm_resync"


def test_bad_arg_optics_type():
    with pytest.raises(ArgumentError) as e:
        resync.main(input=Path("yeah ok"),
                    optics_dir=42,
                    output_file="yay",
                    ring="HER")  # type: ignore
    assert "optics_dir' is not of type Path" in str(e.value)


def test_bad_arg_output_file_type():
    with pytest.raises(ArgumentError) as e:
        resync.main(input=Path("yeah ok"),
                    optics_dir=Path("yay"),
                    output_file=42,
                    ring="HER")  # type: ignore
    assert "output_file' is not of type Path" in str(e.value)


def test_bad_arg_ring():
    with pytest.raises(ArgumentError) as e:
        resync.main(input="yeah ok",
                    optics_dir=Path("yay"),
                    output_file=Path("wat"),
                    ring="MOON_COLLIDER")  # type: ignore
    assert "ring' needs to be one of" in str(e.value)

def test_bad_arg_tbt_datatype():
    with pytest.raises(ArgumentError) as e:
        resync.main(input="yeah ok",
                    optics_dir=Path("yay"),
                    output_file=Path("wat"),
                    ring="HER",
                    tbt_datatype="quantum_sdds")  # type: ignore
    assert "tbt_datatype' needs to be one of" in str(e.value)


def test_resync(tmp_path):
    # Synchronize the BPMs and check against the control
    resync.main(input=INPUTS_DIR / "unsynced.sdds",
                optics_dir=OPTICS_DIR,
                output_file=tmp_path / "output.sdds",
                ring="HER")  # type: ignore

    assert Path(tmp_path / "output.sdds").exists()

    synced_tbt = tbt.read(INPUTS_DIR / "synced.sdds")
    output_tbt = tbt.read(tmp_path / "output.sdds")

    assert np.all(synced_tbt.matrices[0].X == output_tbt.matrices[0].X)
    assert np.all(synced_tbt.matrices[0].Y == output_tbt.matrices[0].Y)


def test_overwrite_ok(tmp_path):
    # Write an output file to create a conflict
    (tmp_path / "output.sdds").write_text("This file already exists.")

    # Synchronize the BPMs and check against the control
    resync.main(input=INPUTS_DIR / "unsynced.sdds",
                optics_dir=OPTICS_DIR,
                output_file=tmp_path / "output.sdds",
                ring="HER",
                overwrite=True)  # type: ignore

    assert Path(tmp_path / "output.sdds").exists()

    synced_tbt = tbt.read(INPUTS_DIR / "synced.sdds")
    output_tbt = tbt.read(tmp_path / "output.sdds")

    assert np.all(synced_tbt.matrices[0].X == output_tbt.matrices[0].X)
    assert np.all(synced_tbt.matrices[0].Y == output_tbt.matrices[0].Y)


def test_overwrite_raise(tmp_path):
    # Write an output file to create a conflict
    (tmp_path / "output.sdds").write_text("This file already exists.")

    # Synchronize the BPMs and check against the control
    with pytest.raises(FileExistsError) as e:
        resync.main(input=INPUTS_DIR / "unsynced.sdds",
                    optics_dir=OPTICS_DIR,
                    output_file=tmp_path / "output.sdds",
                    ring="HER",
                    overwrite=False)  # type: ignore

    assert "output.sdds already exists, aborting." in str(e.value)