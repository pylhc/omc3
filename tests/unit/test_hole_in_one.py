""" 
Tests for the hole in one analysis.
Tests hole_in_one as a unit, to see if it runs as expected.
"""
from pathlib import Path

import pytest

from omc3.hole_in_one import hole_in_one_entrypoint, DEFAULT_CONFIG_FILENAME
from omc3.harpy.constants import LINFILES_SUBFOLDER, FILE_LIN_EXT, FILE_AMPS_EXT, FILE_FREQS_EXT
from omc3.optics_measurements.constants import (
    EXT, BETA_NAME, AMP_BETA_NAME, F1001_NAME, F1010_NAME, PHASE_NAME, TOTAL_PHASE_NAME, KICK_NAME
)


INPUT_DIR = Path(__file__).parent.parent / 'inputs'

SDDS_FILE = INPUT_DIR / "lhc_200_turns.sdds"
MODEL_DIR = INPUT_DIR / "models" / "2022_inj_b1_acd"


@pytest.mark.exteded
@pytest.mark.parametrize("clean", (True, False))
def test_hole_in_one(tmp_path, clean):
    """
    Test that is closely related to how actual analysis are done.
    This test checks that everything runs through without issues.

    A goal to implement this test was to make sure that the 
    following bugs are fixed, i.e. for omc3 <= 0.15.3 this test will fail for the following reasons:
      - In the optics analysis ERRAMPX/Y was required, but this column
        was only set in the cleaning step, i.e. with clean=True
    """
    # Run harpy on the SDDS file
    analysis_output = tmp_path / "analysis_output"
    hole_in_one_entrypoint(
        harpy=True,
        clean=clean,
        output_bits=8,
        turn_bits=10,
        autotunes="transverse",
        outputdir=analysis_output,
        files=[SDDS_FILE],
        model=MODEL_DIR / "twiss_elements.dat",
        to_write=["lin", "spectra"],
        unit="mm"
    )

    _check_all_harpy_files(analysis_output)

    # Run optics on the analysis output
    optics_output = tmp_path / "optics_output"
    hole_in_one_entrypoint(
        beam=1,
        optics=True,
        accel="lhc",
        year="2022",
        model_dir=MODEL_DIR,
        files=[analysis_output/SDDS_FILE.name],
        outputdir=optics_output,
    )

    _check_linear_optics_files(optics_output)


@pytest.mark.extended
def test_hole_in_one_in_one(tmp_path):
    """
    This test runs harpy, optics and optics analysis in one.
    This test checks that everything runs through without issues.

    A goal to implement this test was to make sure that the 
    following bugs are fixed, i.e. for omc3 <= 0.15.3 this test will fail for the following reasons:
        - Pandas to numpy dtype conversions for the lin files went wrong and numpy had 'obj' arrays.
    """
    output = tmp_path / "output"
    hole_in_one_entrypoint(
        harpy=True,
        optics=True,
        clean=False,
        output_bits=8,
        turn_bits=10,
        autotunes="transverse",
        outputdir=output,
        files=[SDDS_FILE],
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

    _check_all_harpy_files(output / LINFILES_SUBFOLDER)
    _check_linear_optics_files(output)

    _check_nonlinear_optics_files(output, "rdt")
    _check_nonlinear_optics_files(output, "crdt")


# Helper -----------------------------------------------------------------------

def _check_all_harpy_files(outputdir: Path):
    assert outputdir.is_dir()
    for ext in (FILE_LIN_EXT, FILE_AMPS_EXT, FILE_FREQS_EXT):
        for plane in ("x", "y"):
            assert (outputdir / f"{SDDS_FILE.name}{ext.format(plane=plane)}").is_file()


def _check_linear_optics_files(outputdir: Path):
    assert outputdir.is_dir()

    assert len(list(outputdir.glob(DEFAULT_CONFIG_FILENAME.format(time="*")))) == 1

    for filename in (F1001_NAME, F1010_NAME):
        assert (outputdir / f"{filename}{EXT}").is_file()

    for filename in (BETA_NAME, AMP_BETA_NAME, PHASE_NAME, TOTAL_PHASE_NAME, KICK_NAME):
        for plane in ("x", "y"):
            assert (outputdir / f"{filename}{plane}{EXT}").is_file()


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

