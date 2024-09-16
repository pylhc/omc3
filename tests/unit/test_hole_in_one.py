""" 
Tests for the hole in one analysis.
Tests hole_in_one as a unit, to see if it runs as expected.

A goal to implement these tests was to make sure that the 
following bugs are fixed, i.e. for omc3 <= 0.15.3 some of the test configurations will fail for 
one of the following reasons:

    - In the optics analysis ERRAMPX/Y was required, but this column
    was only set in the cleaning step, i.e. with clean=True
    https://github.com/pylhc/omc3/issues/451

    - Paths were not accepted as file-input to hole-in-one
    https://github.com/pylhc/omc3/issues/452

    - Pandas to numpy dtype conversions for the lin files went wrong and numpy had 'obj' arrays.
    https://github.com/pylhc/omc3/issues/453
    
    - RDT and CRDT dimensions mismatch when off-momentum files were analysed.
    https://github.com/pylhc/omc3/issues/456

"""
from __future__ import annotations
from pathlib import Path

import pytest

from omc3.hole_in_one import hole_in_one_entrypoint, DEFAULT_CONFIG_FILENAME, LINFILES_SUBFOLDER
from omc3.harpy.constants import FILE_LIN_EXT, FILE_AMPS_EXT, FILE_FREQS_EXT
from omc3.optics_measurements.constants import (
    EXT, 
    BETA_NAME, AMP_BETA_NAME, F1001_NAME, F1010_NAME, PHASE_NAME, TOTAL_PHASE_NAME, KICK_NAME, ORBIT_NAME,
    DISPERSION_NAME, NORM_DISP_NAME,
)

INPUT_DIR = Path(__file__).parent.parent / 'inputs'

MODEL_DIR = INPUT_DIR / "models" / "2022_inj_b1_acd"
SDDS_DIR = INPUT_DIR / "lhcb1_tbt_inj_on_off_mom"

# These are 2024 injection optics files from Beam1, should work with the 2022 model as well.
SDDS_FILES = {
    "SINGLE": ["Beam1@BunchTurn@2024_03_08@17_41_48_045_250turns.sdds"],
    "0Hz": ["Beam1@BunchTurn@2024_03_08@17_41_48_045_250turns.sdds", "Beam1@BunchTurn@2024_03_08@17_42_58_701_250turns.sdds", "Beam1@BunchTurn@2024_03_08@17_44_07_494_250turns.sdds"],
    "+50Hz": ["Beam1@BunchTurn@2024_03_08@17_56_20_055_250turns.sdds", "Beam1@BunchTurn@2024_03_08@17_57_41_540_250turns.sdds", "Beam1@BunchTurn@2024_03_08@17_58_53_905_250turns.sdds"],
    "-50Hz": ["Beam1@BunchTurn@2024_03_08@18_24_02_100_250turns.sdds", "Beam1@BunchTurn@2024_03_08@18_25_23_729_250turns.sdds", "Beam1@BunchTurn@2024_03_08@18_26_41_811_250turns.sdds"],
}


@pytest.mark.extended
@pytest.mark.parametrize("which_files", ("SINGLE", "0Hz", "all"))
@pytest.mark.parametrize("clean", (True, False), ids=lambda val : f"clean={val}")
def test_hole_in_two(tmp_path, clean, which_files):
    """
    Test that is closely related to how actual analysis are done.
    """
    # Run harpy on the SDDS file
    analysis_output = tmp_path / "analysis_output"
    sdds_files = _get_sdds_files(which_files)
    hole_in_one_entrypoint(
        harpy=True,
        clean=clean,
        output_bits=8,
        turn_bits=10,
        autotunes="transverse",
        outputdir=analysis_output,
        files=sdds_files,
        model=MODEL_DIR / "twiss_elements.dat",
        to_write=["lin", "spectra"],
        unit="mm"
    )

    for sdds_file in sdds_files:
        _check_all_harpy_files(analysis_output, sdds_file)

    # Run optics on the analysis output
    optics_output = tmp_path / "optics_output"
    hole_in_one_entrypoint(
        beam=1,
        optics=True,
        accel="lhc",
        year="2022",
        model_dir=MODEL_DIR,
        files=[analysis_output / sdds_file.name for sdds_file in sdds_files],
        nonlinear=['rdt', 'crdt'],
        outputdir=optics_output,
    )

    _check_linear_optics_files(optics_output, off_momentum=(which_files == "all"))
    _check_nonlinear_optics_files(optics_output, "rdt")
    _check_nonlinear_optics_files(optics_output, "crdt")


@pytest.mark.extended
@pytest.mark.parametrize("which_files", ("SINGLE", "0Hz", "all"))
@pytest.mark.parametrize("clean", (True, False), ids=lambda val : f"clean={val}")
def test_hole_in_one(tmp_path, clean, which_files):
    """
    This test runs harpy, optics and optics analysis in one.
    """
    output = tmp_path / "output"
    files = _get_sdds_files(which_files)
    hole_in_one_entrypoint(
        harpy=True,
        optics=True,
        clean=clean,
        output_bits=8,
        turn_bits=10,
        autotunes="transverse",
        outputdir=output,
        files=files,
        model=MODEL_DIR / "twiss_elements.dat",
        to_write=["lin", "spectra", "bpm_summary"],
        window="hann",
        compensation="model",
        coupling_method=2,
        nonlinear=['rdt', 'crdt'],
        unit="mm",
        beam=1,
        accel="lhc",
        year="2022",
        model_dir=MODEL_DIR,
    )

    for sdds_file in files:
        _check_all_harpy_files(output / LINFILES_SUBFOLDER, sdds_file)

    _check_linear_optics_files(output, off_momentum=(which_files == "all"))
    _check_nonlinear_optics_files(output, "rdt")
    _check_nonlinear_optics_files(output, "crdt")


# Helper -----------------------------------------------------------------------

def _check_all_harpy_files(outputdir: Path, sdds_file: Path):
    assert outputdir.is_dir()
    for ext in (FILE_LIN_EXT, FILE_AMPS_EXT, FILE_FREQS_EXT):
        for plane in ("x", "y"):
            assert (outputdir / f"{sdds_file.name}{ext.format(plane=plane)}").is_file()


def _check_linear_optics_files(outputdir: Path, off_momentum: bool = False):
    assert outputdir.is_dir()

    assert len(list(outputdir.glob(DEFAULT_CONFIG_FILENAME.format(time="*")))) == 1

    for filename in (F1001_NAME, F1010_NAME):
        assert (outputdir / f"{filename}{EXT}").is_file()

    for filename in (BETA_NAME, AMP_BETA_NAME, PHASE_NAME, TOTAL_PHASE_NAME, KICK_NAME, ORBIT_NAME):
        for plane in ("x", "y"):
            assert (outputdir / f"{filename}{plane}{EXT}").is_file()

    if off_momentum:
        for filename in   (DISPERSION_NAME,):
            for plane in ("x", "y"):
                assert (outputdir / f"{filename}{plane}{EXT}").is_file()
        
        assert (outputdir / f"{NORM_DISP_NAME}{'x'}{EXT}").is_file()



def _check_nonlinear_optics_files(outputdir: Path, type_: str):
    assert outputdir.is_dir()
    nonlin_dir = outputdir / type_
    assert nonlin_dir.is_dir()

    magnets = ["octupole", "sextupole", "quadrupole"]
    for magnet in magnets:
        for orientation in ("normal", "skew"):
            full_manget_name = f"{orientation}_{magnet}"

            if type_ == "crdt":
                if full_manget_name in ("skew_octupole", "normal_quadrupole"):
                    continue

            magnet_dir = nonlin_dir / full_manget_name
            assert magnet_dir.is_dir()
            assert len(list(magnet_dir.glob(f"*{EXT}"))) > 0


def _get_sdds_files(which: str) -> list[Path]:
    if which in SDDS_FILES.keys():
        return [SDDS_DIR / sdds_file for sdds_file in SDDS_FILES[which]]

    if "all":    
        return [SDDS_DIR / sdds_file for sdds_file in SDDS_FILES["0Hz"] + SDDS_FILES["+50Hz"] + SDDS_FILES["-50Hz"]] 
    
    raise ValueError(f"which should be one of {SDDS_FILES.keys()} or 'all'")
