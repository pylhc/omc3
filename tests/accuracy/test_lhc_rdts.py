from pathlib import Path

import pytest
import tfs
from numpy import abs, angle

from tests.inputs.lhc_rdts.omc3_helpers import (
    get_file_ext,
    get_rdts,
    get_rdts_from_harpy,
    run_harpy,
)
from tests.inputs.lhc_rdts.rdt_constants import (
    DATA_DIR,
    # MODEL_ANALYTICAL_PREFIX,
    MODEL_NG_PREFIX,
)

INPUTS = Path(__file__).parent.parent / "inputs"


def generate_rdts():
    rdt_configurations = []
    for beam in (1, 2):
        for order in [2, 3]:
            for rdt in get_rdts(order):
                rdt_configurations.append((rdt, beam, order))
    return rdt_configurations


@pytest.fixture(scope="module")
def pre_test_calculations(tmp_path_factory: pytest.TempPathFactory) -> dict:
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
    # To test the harpy opposite_direction when in combination with the optics=True
    run_harpy(beam=1, order=2)
    run_harpy(beam=2, order=3)


@pytest.mark.parametrize(
    "rdt_config", generate_rdts(), ids=lambda x: f"Beam{x[1]}-{x[0]}"
)
def test_lhc_rdts(rdt_config: tuple[str, int, int], pre_test_calculations):
    rdt, beam, order = rdt_config
    analysis_path = pre_test_calculations[(beam, order)]
    rdt_dfs = get_rdts_from_harpy(
        beam, order, output_dir=analysis_path, check_previous=True
    )
    omc_df = rdt_dfs[rdt]

    file_stg_ext = get_file_ext(beam, order)
    ng_df = tfs.read(DATA_DIR / f"{MODEL_NG_PREFIX}_{file_stg_ext}.tfs", index="NAME")
    ng_rdt = rdt.split("_")[0].upper()

    # Reconstruct the complex numbers
    ng_complex = ng_df[f"{ng_rdt}REAL"] + 1j * ng_df[f"{ng_rdt}IMAG"]
    omc_complex = omc_df["REAL"] + 1j * omc_df["IMAG"]

    # Calculate the amplitudes
    ng_amplitude = abs(ng_complex)
    omc_amplitude = abs(omc_complex)

    # Compare the amplitudes
    ng_diff = ng_amplitude - omc_amplitude
    ng_diff[ng_amplitude.abs() > 1] = (
        ng_diff[ng_amplitude.abs() > 1] / ng_amplitude[ng_amplitude.abs() > 1]
    )
    assert ng_diff.abs().max() < 5e-2  # 5% difference

    # Compare the phases
    ng_phase_diff = angle(ng_complex / omc_complex)
    assert ng_phase_diff.max() < 5e-2  # 0.05 rad difference

    # Now check the analytical model
    # analytical_df = tfs.read(
    #     DATA_DIR / f"{MODEL_ANALYTICAL_PREFIX}_{file_stg_ext}.tfs", index="NAME"
    # )
    # analytical_complex = (
    #     analytical_df[f"{ng_rdt}REAL"] + 1j * analytical_df[f"{ng_rdt}IMAG"]
    # )
    
    # # Calculate the amplitudes
    # analytical_amplitude = abs(analytical_complex)
    # analytical_diff = analytical_amplitude - omc_amplitude

    # # Compare the amplitudes, relative when above 1, absolute when below
    # gt1 = analytical_amplitude.abs() > 1
    # analytical_diff[gt1] = analytical_diff[gt1] / analytical_amplitude[gt1]
    # assert analytical_diff.abs().max() < 5e-2

    # # Compare the phases
    # analytical_phase_diff = angle(analytical_complex / omc_complex)
    # assert analytical_phase_diff.max() < 5e-2
