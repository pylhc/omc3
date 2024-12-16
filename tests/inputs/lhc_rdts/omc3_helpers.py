from pathlib import Path

import tfs

from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.optics_measurements.constants import RDT_FOLDER
from tests.inputs.lhc_rdts.rdt_constants import (
    ANALYSIS_DIR,
    DATA_DIR,
    FREQ_OUT_DIR,
    NORMAL_RDTS3,
    NORMAL_RDTS4,
    SKEW_RDTS3,
    SKEW_RDTS4,
    TEST_DIR,
)


def filter_IPs(df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """Filter the DataFrame to include only BPMs."""
    return df.filter(regex=r"^BPM\.[1-9][0-9].", axis="index")


def get_file_suffix(beam: int) -> str:
    """Return the file suffix for the test files based on beam and order."""
    assert beam in [1, 2], "Beam must be 1 or 2"
    return f"b{beam}"


def get_rdts() -> list[str]:
    """Return the all the RDTs."""
    return NORMAL_RDTS3 + SKEW_RDTS3 + NORMAL_RDTS4 + SKEW_RDTS4


def get_tbt_name(beam: int, sdds: bool = True) -> str:
    """Return the name of the TBT file for the given test parameters."""
    return f"tbt_data_{get_file_suffix(beam)}.{'sdds' if sdds else 'tfs.bz2'}"


def get_model_dir(beam: int) -> Path:
    """Return the model directory for the given test parameters."""
    return TEST_DIR / f"model_{get_file_suffix(beam)}"


def get_max_rdt_order(rdts: list[str]) -> int:
    """Return the maximum order of the RDTs."""
    return max(sum(int(num) for num in rdt.split("_")[0][1:]) for rdt in rdts)


def get_output_dir(tbt_name: str, output_dir: Path = None) -> Path:
    """Return the output directory for the given TBT, and create it if it does not exist."""
    if output_dir is None:
        output_dir = ANALYSIS_DIR / f"{tbt_name.split('.')[0]}"
        output_dir.mkdir(exist_ok=True)
    return output_dir


def get_rdt_type(rdt: str) -> tuple[str, str]:
    rdt_as_list = [int(num) for num in rdt.split("_")[0][1:]]
    is_skew = (rdt_as_list[2] + rdt_as_list[3]) % 2 == 1
    order = sum(rdt_as_list)
    return "skew" if is_skew else "normal", "octupole" if order == 4 else "sextupole"


def get_rdt_paths(rdts: list[str], output_dir: Path) -> dict[str, Path]:
    """Return a dictionary of RDTs and their corresponding file paths."""
    # Is there a method for this in the OMC3 source code? (jgray 2024)
    rdt_paths = {}
    for rdt in rdts:
        is_skew, order_name = get_rdt_type(rdt)
        rdt_paths[rdt] = output_dir / f"{RDT_FOLDER}/{is_skew}_{order_name}/{rdt}.tfs"
    return rdt_paths


def get_tunes(output_dir: Path) -> list[float]:
    optics_file = output_dir / "beta_amplitude_x.tfs"
    df = tfs.read(optics_file)
    tunes = [df.headers["Q1"], df.headers["Q2"]]
    return tunes


def get_rdts_from_harpy(
    beam: int,
    output_dir: Path = None,
    check_previous=False,
) -> dict[str, tfs.TfsDataFrame]:
    """
    Run the optics analysis for the given test parameters and return the RDTs.

    If output_dir is None, the output directory will be created in the rdt_constants.ANALYSIS_DIR.
    If check_previous is True, the analysis will only be done if the output files do not exist.
    """
    rdts = get_rdts()
    only_coupling = all(rdt.lower() in ["f1001", "f1010"] for rdt in rdts)
    rdt_order = get_max_rdt_order(rdts)
    tbt_name = get_tbt_name(beam)
    output_dir = get_output_dir(tbt_name, output_dir)

    rdt_paths = get_rdt_paths(rdts, output_dir)

    # Run the analysis if the output files do not exist or check_previous is False
    if not check_previous or any(not path.exists() for path in rdt_paths.values()):
        hole_in_one_entrypoint(
            files=[FREQ_OUT_DIR / tbt_name],
            outputdir=output_dir,
            optics=True,
            accel="lhc",
            beam=beam,
            year="2024",
            energy=6.8,
            model_dir=get_model_dir(beam),
            only_coupling=only_coupling,
            compensation="none",
            nonlinear=["rdt"],
            rdt_magnet_order=rdt_order,
        )
        tunes = get_tunes(output_dir)
        print(f"Tunes for beam {beam}: {tunes}")
        if abs(tunes[0] - 0.28) > 0.0001 or abs(tunes[1] - 0.31) > 0.0001:
            raise ValueError(
                "Tunes are far from the expected values, rdts will be wrong/outside the tolerance"
            )
    dfs = {
        rdt: filter_IPs(tfs.read(path, index="NAME")) for rdt, path in rdt_paths.items()
    }
    return dfs


def run_harpy(beam: int) -> None:
    """Run Harpy for the given test parameters."""

    tbt_file = DATA_DIR / get_tbt_name(beam, sdds=True)
    hole_in_one_entrypoint(
        harpy=True,
        files=[tbt_file],
        outputdir=FREQ_OUT_DIR,
        to_write=["lin", "spectra"],
        opposite_direction=beam == 2,
        tunes=[0.28, 0.31, 0.0],
        natdeltas=[0.0, -0.0, 0.0],
        turn_bits=18,
    )
