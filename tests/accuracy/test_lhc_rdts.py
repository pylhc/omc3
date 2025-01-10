from pathlib import Path

import numpy as np
import pytest
import tfs

from omc3.definitions.constants import PI2
from tests.inputs.lhc_rdts.omc3_helpers import (
    get_file_suffix,
    get_rdt_type,
    get_rdts,
    get_rdts_from_harpy,
    run_harpy,
)
from tests.inputs.lhc_rdts.rdt_constants import (
    DATA_DIR,
    MODEL_ANALYTICAL_PREFIX,
    MODEL_NG_PREFIX,
    FREQ_OUT_DIR,
)

INPUTS = Path(__file__).parent.parent / "inputs"


@pytest.fixture(scope="module")
def initialise_test_paths(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Initialize temporary paths for test analysis.

    This fixture creates temporary directories for each beam to store analysis results.
    The directories are created using the pytest temporary path factory.

    Args:
        tmp_path_factory (pytest.TempPathFactory): Factory for creating temporary paths.

    Returns:
        dict: A dictionary mapping beam number to the temporary path.
    """
    paths = {}
    for beam in (1, 2):
        path = tmp_path_factory.mktemp(f"analysis_beam{beam}", numbered=False)
        paths[beam] = path
    return paths


@pytest.fixture(scope="module", autouse=True)
def run_selective_harpy():
    """Run the harpy analysis for selected RDTs.

    This fixture runs the harpy analysis for beam 1 and beam 2 to test the
    opposite_direction flag in combination with optics=True.
    """
    run_harpy(beam=1)
    run_harpy(beam=2)
    yield # Run the tests

    # Clean up the analysis files
    for analysis_path in FREQ_OUT_DIR.iterdir():
        analysis_path.unlink()
    FREQ_OUT_DIR.rmdir()


AMPLITUDE_TOLERANCES = {
    1: {  # Beam 1: 0.6% and 4% for sextupoles and octupoles respectively
        "sextupole": 6e-3,
        "octupole": 4e-2,
    },
    2: {  # Beam 2: 0.6% and 5% for sextupoles and octupoles respectively
        "sextupole": 6e-3,
        "octupole": 5e-2,
    },
}
PHASE_TOLERANCES = {
    1: {  # Beam 1: 0.04 rad and 0.03 rad for orders 2 and 3 respectively
        "sextupole": 4e-2,
        "octupole": 3e-2,
    },
    2: {  # Beam 2: 0.07 rad and 0.05 rad for orders 2 and 3 respectively
        "sextupole": 7e-2,
        "octupole": 5e-2,
    },
}


@pytest.mark.parametrize("rdt", get_rdts())
@pytest.mark.parametrize("beam", [1, 2], ids=lambda x: f"Beam{x}")
def test_lhc_rdts(beam: int, rdt: str, initialise_test_paths):
    """Test the RDTs calculated by OMC3 against the analytical model and MAD-NG.

    The test forces harpy to run for the selected RDTs to ensure the use of the
    opposite_direction flag is consistent with the optics=True, beam=2 situation.

    The configuration for the test is set in the file tests/inputs/lhc_rdts/rdt_constants.py:
    - NTURNS = 1000
    - KICK_AMP = 1e-3
    - SEXTUPOLE_STRENGTH = 3e-5
    - OCTUPOLE_STRENGTH = 3e-3

    If this configuration is changed, the test needs to be updated accordingly by running
    the create_data.py script in the same directory.

    This configuration was specifically chosen to ensure that OMC3 is given a fair chance
    to produce correct results. The main issues that can arise is detuning. The octupole
    and sextupole strengths are chosen to be small enough to avoid large detuning effects,
    but also large enough to produce a measurable effect. Furthermore, the larger each
    strength, the more likely the regime will become nonlinear, which can lead to larger
    errors in the RDT calculations.
    """
    # Retrieve the current RDT configuration
    _, order = get_rdt_type(rdt)

    # Get the temporary analysis path
    analysis_path = initialise_test_paths[beam]

    # Now perform the optics calculations from the harpy output
    # As this is run per RDT, the check_previous is set to True to avoid rerunning the analysis
    rdt_dfs = get_rdts_from_harpy(beam, output_dir=analysis_path, check_previous=True)
    omc_df = rdt_dfs[rdt]  # get the result of a specific RDT

    # Retrieve the MAD-NG results for the set of RDTs
    file_suffix = get_file_suffix(beam)
    ng_df = tfs.read(DATA_DIR / f"{MODEL_NG_PREFIX}_{file_suffix}.tfs", index="NAME")
    ng_rdt = rdt.split("_")[0].upper()

    # Reconstruct the complex numbers from the real and imaginary parts
    ng_complex = ng_df[ng_rdt]
    omc_complex = omc_df["REAL"] + 1j * omc_df["IMAG"]

    # Calculate the amplitudes
    ng_amplitude = np.abs(ng_complex)
    omc_amplitude = np.abs(omc_complex)
    assert np.allclose(omc_amplitude, omc_df["AMP"], rtol=1e-10)

    # Compare the amplitudes
    ng_diff = ng_amplitude - omc_amplitude
    ng_diff[ng_amplitude.abs() > 1] = (
        ng_diff[ng_amplitude.abs() > 1] / ng_amplitude[ng_amplitude.abs() > 1]
    )
    assert ng_diff.abs().max() < AMPLITUDE_TOLERANCES[beam][order]

    # Compare OMC3 phase to real and imaginary parts
    omc_phase = np.angle(omc_complex) / PI2
    diff = omc_phase - omc_df["PHASE"]
    diff = (diff + 0.5) % 1 - 0.5
    assert diff.abs().max() < 1e-11

    # Compare the phases
    ng_phase_diff = np.angle(ng_complex / omc_complex)
    assert ng_phase_diff.max() < PHASE_TOLERANCES[beam][order]

    # Now check the analytical model for sextupoles
    if order == "sextupole":
        analytical_df = tfs.read(
            DATA_DIR / f"{MODEL_ANALYTICAL_PREFIX}_{file_suffix}.tfs", index="NAME"
        )
        analytical_complex = analytical_df[ng_rdt]

        # Calculate the amplitudes
        analytical_amplitude = np.abs(analytical_complex)
        analytical_diff = analytical_amplitude - omc_amplitude

        # Compare the amplitudes, relative when above 1, absolute when below
        gt1 = analytical_amplitude.abs() > 1
        analytical_diff[gt1] = analytical_diff[gt1] / analytical_amplitude[gt1]
        assert analytical_diff.abs().max() < 1.1e-2

        # Compare the phases (this for some reason fails for f1300_x, max diff of about 0.2)
        analytical_phase_diff = np.angle(analytical_complex / omc_complex)
        assert analytical_phase_diff.max() < 6e-2
