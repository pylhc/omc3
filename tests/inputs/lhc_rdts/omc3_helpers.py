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

def get_file_ext(beam: int, order: int) -> str:
    """Return the file extension for the test files based on beam and order."""
    assert beam in [1, 2], "Beam must be 1 or 2"
    assert order in [2, 3], "Order must be 2 or 3"
    order_name = "oct" if order == 3 else "sext"
    return f"b{beam}_{order_name}"

def get_rdts(order: int) -> list[str]:
    """Return the RDTs for the given order."""
    if order == 2:
        return SKEW_RDTS3 + NORMAL_RDTS3
    elif order == 3:
        return SKEW_RDTS4 + NORMAL_RDTS4
    else:
        raise ValueError("Order must be 2 or 3")

def get_tbt_name(beam: int, order: int, sdds: bool = True) -> str:
    """Return the name of the TBT file for the given test parameters."""
    return f"tbt_data_{get_file_ext(beam, order)}.{'sdds' if sdds else 'tfs'}"

def get_model_dir(beam: int, order: int) -> Path:
    """Return the model directory for the given test parameters."""
    return TEST_DIR / f"model_{get_file_ext(beam, order)}"

def get_max_rdt_order(rdts: list[str]) -> int:
    """Return the maximum order of the RDTs."""
    return max(sum(int(num) for num in rdt.split("_")[0][1:]) for rdt in rdts)

def get_output_dir(tbt_name: str, output_dir: Path = None) -> Path:
    """Return the output directory for the given TBT, and create it if it does not exist."""
    if output_dir is None:
        output_dir = ANALYSIS_DIR / f"{tbt_name.split('.')[0]}"
        output_dir.mkdir(exist_ok=True)
    return output_dir

def is_rdt_skew(rdt: str) -> bool:
    """Check if the RDT is a skew RDT."""
    rdt_as_list = [int(num) for num in rdt.split("_")[0][1:]]
    is_skew = (rdt_as_list[2] + rdt_as_list[3]) % 2 == 1
    return "skew" if is_skew else "normal"

def get_rdt_paths(rdts: list[str], output_dir: Path, order_name: str) -> dict[str, Path]:
    """Return a dictionary of RDTs and their corresponding file paths."""
    return {
        rdt: output_dir / f"{RDT_FOLDER}/{is_rdt_skew(rdt)}_{order_name}/{rdt}.tfs"
        for rdt in rdts
    }

def get_rdts_from_harpy(
    beam: int,
    order: int,
    output_dir: Path = None,
    check_previous=False,
) -> dict[str, tfs.TfsDataFrame]:
    """
    Run the optics analysis for the given test parameters and return the RDTs.

    If output_dir is None, the output directory will be created in the rdt_constants.ANALYSIS_DIR.
    If check_previous is True, the analysis will only be done if the output files do not exist.
    """
    rdts = get_rdts(order)
    only_coupling = all(rdt.lower() in ["f1001", "f1010"] for rdt in rdts)
    order_name = "octupole" if order == 3 else "sextupole"
    rdt_order = get_max_rdt_order(rdts)
    tbt_name = get_tbt_name(beam, order)
    output_dir = get_output_dir(tbt_name, output_dir)

    rdt_paths = get_rdt_paths(rdts, output_dir, order_name)
    
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
            model_dir=get_model_dir(beam, order),
            only_coupling=only_coupling,
            compensation="none",
            nonlinear=["rdt"],
            rdt_magnet_order=rdt_order,
        )
    dfs = {
        rdt: filter_IPs(tfs.read(path, index="NAME"))
        for rdt, path in rdt_paths.items()
    }
    return dfs

def run_harpy(beam: int, order: int) -> None:
    """Run Harpy for the given test parameters."""
    hole_in_one_entrypoint(
        harpy=True,
        files=[DATA_DIR / get_tbt_name(beam, order)],
        outputdir=FREQ_OUT_DIR,
        to_write=["lin", "spectra"],
        opposite_direction=beam == 2,
        tunes=[0.28, 0.31, 0.0],
        natdeltas=[0.0, -0.0, 0.0],
        turn_bits=18,
    )