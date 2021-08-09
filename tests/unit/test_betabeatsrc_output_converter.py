from pathlib import Path

import numpy as np
import pytest
import tfs

from omc3.optics_measurements.constants import DELTA, ERR, MDL
from omc3.definitions.constants import PLANES
from omc3.scripts.betabeatsrc_output_converter import converter_entrypoint

from tests.conftest import cli_args

INPUT_DIR = Path(__file__).parent.parent / "inputs"
BBRSC_OUTPUTS = INPUT_DIR / "bbsrc_output_converter"


@pytest.mark.basic
@pytest.mark.parametrize("suffix", ["_free", "_free2"])
def test_betabeatsrc_output_converter(tmp_path, suffix):
    converter_entrypoint(inputdir=str(BBRSC_OUTPUTS), outputdir=str(tmp_path), suffix=suffix)

    _assert_correct_files_are_present(tmp_path)


@pytest.mark.basic
@pytest.mark.parametrize("suffix", ["_free", "_free2"])
def test_betabeatsrc_output_converter_commandline(tmp_path, suffix):
    with cli_args("--inputdir", str(BBRSC_OUTPUTS.absolute()),
                  "--outputdir", str(tmp_path),
                  "--suffix", suffix):
        converter_entrypoint()

    _assert_correct_files_are_present(tmp_path)


# ----- Helpers ----- #


def _assert_correct_files_are_present(outputdir: Path) -> None:
    """Simply checks the expected converted files are present in the outputdir"""
    for plane in PLANES:
        assert (outputdir / f"beta_amplitude_{plane.lower()}.tfs").is_file()
        assert (outputdir / f"beta_phase_{plane.lower()}.tfs").is_file()
        assert (outputdir / f"orbit_{plane.lower()}.tfs").is_file()
        assert (outputdir / f"phase_{plane.lower()}.tfs").is_file()
        assert (outputdir / f"total_phase_{plane.lower()}.tfs").is_file()

    for rdt in ["1001", "1010"]:
        assert (outputdir / f"coupling_f{rdt}.tfs").is_file()
