from pathlib import Path

import numpy as np
import pytest
import tfs

from omc3.definitions.constants import PI2
from tests.utils.compression import compress_model, decompress_model
from tests.utils.lhc_rdts.constants import (
    DATA_DIR,
    FREQ_OUT_DIR,
    MODEL_ANALYTICAL_PREFIX,
    MODEL_NG_PREFIX,
)
from tests.utils.lhc_rdts.functions import (
    get_file_suffix,
    get_model_dir,
    get_rdt_names,
    get_rdt_type,
    get_rdts_from_optics_analysis,
    run_harpy,
)

INPUTS = Path(__file__).parent.parent / "inputs"


@pytest.fixture(scope="module")
def rdts_from_optics_analysis(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the optics analysis for the selected RDTs.

    This fixture runs the optics analysis for the selected RDTs for both beams.

    Args:
        initialise_test_paths (dict): A dictionary mapping beam number to the temporary path.

    Returns:
        dict: A dictionary mapping beam number to the RDTs calculated by OMC3.
    """
    dfs = {}
    for beam in (1, 2):
        analysis_path = tmp_path_factory.mktemp(f"analysis_beam{beam}", numbered=False)
        dfs[beam] = get_rdts_from_optics_analysis(beam, output_dir=analysis_path)
    return dfs


@pytest.fixture(scope="module", autouse=True)
def run_selective_harpy():
    """Run the harpy analysis for selected RDTs.

    This fixture runs the harpy analysis for beam 1 and beam 2 to test the
    opposite_direction flag in combination with optics=True.
    """
    run_harpy(beam=1)
    run_harpy(beam=2)
    decompress_model(get_model_dir(beam=1))
    decompress_model(get_model_dir(beam=2))
    yield # Run the tests

    compress_model(get_model_dir(beam=1))
    compress_model(get_model_dir(beam=2))
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


@pytest.mark.parametrize("rdt", get_rdt_names())
@pytest.mark.parametrize("beam", [1, 2], ids=lambda x: f"Beam{x}")
def test_lhc_rdts(beam: int, rdt: str, rdts_from_optics_analysis):
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

    # Now retrieve the optics calculations from the OMC3 analysis
    rdt_dfs = rdts_from_optics_analysis[beam]
    omc_df = rdt_dfs[rdt]  # get the result of a specific RDT

    # Retrieve the MAD-NG results for the set of RDTs
    file_suffix = get_file_suffix(beam)
    ng_df = tfs.read(DATA_DIR / f"{MODEL_NG_PREFIX}_{file_suffix}.tfs", index="NAME")
    ng_rdt = rdt.split("_")[0].upper()

    # Reconstruct the complex numbers from the real and imaginary parts
    ng_complex = ng_df[ng_rdt]
    omc_complex = omc_df["REAL"] + 1j * omc_df["IMAG"]

    # Calculate the OMC3 and MAD-NG amplitudes from the complex numbers
    ng_amplitude = np.abs(ng_complex)
    omc_amplitude = np.abs(omc_complex)

    # Compare omc3 vs MAD-NG amplitudes, relative when above 1, absolute when below
    ng_diff = ng_amplitude - omc_amplitude
    ng_diff[ng_amplitude.abs() > 1] = (
        ng_diff[ng_amplitude.abs() > 1] / ng_amplitude[ng_amplitude.abs() > 1]
    )
    assert ng_diff.abs().max() < AMPLITUDE_TOLERANCES[beam][order]

    # Compare omc3 vs MAD-NG phases
    ng_phase_diff = np.angle(ng_complex / omc_complex)
    assert ng_phase_diff.max() < PHASE_TOLERANCES[beam][order]

    # Check amplitude and phase vs real and imaginary calculations in omc3 is correct
    omc_phase = np.angle(omc_complex) / PI2
    diff = omc_phase - omc_df["PHASE"]
    diff = (diff + 0.5) % 1 - 0.5
    assert diff.abs().max() < 1e-11
    assert np.allclose(omc_amplitude, omc_df["AMP"], rtol=1e-10)

    # Now check the analytical model for sextupoles
    # Analytical seems to disagree with MAD-NG and OMC3 for octupoles
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

        # Compare the phases
        analytical_phase_diff = np.angle(analytical_complex / omc_complex)
        assert analytical_phase_diff.max() < 6e-2
