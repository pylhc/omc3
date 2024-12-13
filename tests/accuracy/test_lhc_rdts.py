from pathlib import Path

import pytest
import tfs
import numpy as np

from tests.inputs.lhc_rdts.omc3_helpers import (
    get_file_suffix,
    get_rdts,
    get_rdts_from_harpy,
    run_harpy,
)
from tests.inputs.lhc_rdts.rdt_constants import (
    DATA_DIR,
    MODEL_ANALYTICAL_PREFIX,
    MODEL_NG_PREFIX,
)

INPUTS = Path(__file__).parent.parent / "inputs"


def generate_rdts():
    """Generate a list of RDT configurations for testing.

    This function creates a list of tuples containing RDTs, beams, and orders 
    for the test cases. It iterates over beams 1 and 2, and orders 2 and 3, 
    retrieving the RDTs for each combination.

    Returns:
        list: A list of tuples, each containing an RDT, beam, and order.
    """
    rdt_configurations = []
    for beam in (1, 2):
        for order in [2, 3]:
            for rdt in get_rdts(beam, order):
                rdt_configurations.append((rdt, beam, order))
    return rdt_configurations


@pytest.fixture(scope="module")
def initialise_test_paths(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Initialize temporary paths for test analysis.

    This fixture creates temporary directories for each combination of beam and order 
    to store analysis results. The directories are created using the pytest temporary 
    path factory.

    Args:
        tmp_path_factory (pytest.TempPathFactory): Factory for creating temporary paths.

    Returns:
        dict: A dictionary mapping (beam, order) tuples to their respective temporary paths.
    """
    paths = {}
    for beam in (1, 2):
        for order in [2, 3]:
            path = tmp_path_factory.mktemp(
                f"analysis_beam{beam}_order{order}", numbered=False
            )
            paths[(beam, order)] = path
    return paths


@pytest.fixture(scope="module", autouse=True)
def run_selective_harpy():
    """Run the harpy analysis for selected RDTs.

    This fixture runs the harpy analysis for beam 1, order 3 and beam 2, order 2 
    to test the opposite_direction flag in combination with optics=True.
    """
    run_harpy(beam=1, order=3)
    run_harpy(beam=2, order=2)


AMPLITUDE_TOLERANCES = {
    1: {2: 8e-4, 3: 4e-2}, # Beam 1: 0.08% and 4% for orders 2 and 3
    2: {2: 2e-3, 3: 6e-2}, # Beam 2: 0.2% and 6% for orders 2 and 3
}
PHASE_TOLERANCES = {
    1: {2: 2e-3, 3: 8e-3}, # Beam 1: 0.002 rad and 0.008 rad for orders 2 and 3
    2: {2: 2e-3, 3: 4e-2}, # Beam 2: 0.002 rad and 0.04 rad for orders 2 and 3
}

@pytest.mark.parametrize(
    "rdt_config", generate_rdts(), ids=lambda x: f"Beam{x[1]}-{x[0]}"
)
def test_lhc_rdts(rdt_config: tuple[str, int, int], initialise_test_paths):
    """Test the RDTs calculated by OMC3 against the analytical model and MAD-NG.

    This test verifies the amplitude and phase of the RDTs calculated by OMC3 against 
    the analytical model and MAD-NG. See 

    This selection minimises the number of required input files while covering different 
    types of RDTs at multiple orders and beams. The test does not combine skew and normal 
    RDTs in the same test due to:
        - The analytical model not producing correct results for these cases.
        - The OMC3 result for skew sextupolar RDTs, particularly F2010_Y, becoming 
          dependent on a low kick amplitude to produce correct results.

    The test forces harpy to run for the selected RDTs to ensure the use of the 
    opposite_direction flag is consistent with the optics=True, beam=2 situation.

    The configuration for the test is set in the file tests/inputs/lhc_rdts/rdt_constants.py:
    - NTURNS = 1000
    - KICK_AMP = 1e-4
    - SEXTUPOLE_STRENGTH = 1e-3
    - OCTUPOLE_STRENGTH = 5e-3

    If this configuration is changed, the test needs to be updated accordingly by running 
    the create_data.py script in the same directory.
    
    This configuration was specifically chosen to ensure that OMC3 is given a fair chance
    to produce correct results. The main issues that can arise is detuning. The octupole 
    and sextupole strengths are chosen to be small enough to avoid large detuning effects. 
    But to ensure that the RDTs are not too noisy, the octupoles have 10x the kick amplitude.
    """
    # Retrieve the current RDT configuration
    rdt, beam, order = rdt_config

    # Get the temporary analysis path
    analysis_path = initialise_test_paths[(beam, order)]

    # Now perform the optics calculations from the harpy output
    # As this is run per RDT, the check_previous is set to True to avoid rerunning the analysis
    rdt_dfs = get_rdts_from_harpy(
        beam, order, output_dir=analysis_path, check_previous=True
    )
    omc_df = rdt_dfs[rdt] # get the result of a specific RDT

    # Retrieve the MAD-NG results for the set of RDTs
    file_suffix = get_file_suffix(beam, order)
    ng_df = tfs.read(DATA_DIR / f"{MODEL_NG_PREFIX}_{file_suffix}.tfs", index="NAME")
    ng_rdt = rdt.split("_")[0].upper()

    # Reconstruct the complex numbers from the real and imaginary parts
    ng_complex = ng_df[f"{ng_rdt}REAL"] + 1j * ng_df[f"{ng_rdt}IMAG"]
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

    # Compare the phases
    ng_phase_diff = np.angle(ng_complex / omc_complex)
    assert ng_phase_diff.max() < PHASE_TOLERANCES[beam][order]

    # Now check the analytical model if order < 3
    if order < 3:
        analytical_df = tfs.read(
            DATA_DIR / f"{MODEL_ANALYTICAL_PREFIX}_{file_suffix}.tfs", index="NAME"
        )
        analytical_complex = (
            analytical_df[f"{ng_rdt}REAL"] + 1j * analytical_df[f"{ng_rdt}IMAG"]
        )
        
        # Calculate the amplitudes
        analytical_amplitude = np.abs(analytical_complex)
        analytical_diff = analytical_amplitude - omc_amplitude

        # Compare the amplitudes, relative when above 1, absolute when below
        gt1 = analytical_amplitude.abs() > 1
        analytical_diff[gt1] = analytical_diff[gt1] / analytical_amplitude[gt1]
        assert analytical_diff.abs().max() < 2.0e-2

        # Compare the phases (this for some reason fails for f1300_x, max diff of about 0.2)
        analytical_phase_diff = np.angle(analytical_complex / omc_complex)
        assert analytical_phase_diff.max() < 5e-2
