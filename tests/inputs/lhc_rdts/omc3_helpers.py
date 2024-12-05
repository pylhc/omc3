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
    return df.filter(regex=r"^BPM\.[1-9][0-9].", axis="index")

def get_file_ext(beam: int, order: int, is_skew: bool) -> str:
    """Using the test parameters, return the file extension for the test files."""
    assert beam in [1, 2], "Beam must be 1 or 2"
    assert order in [2, 3], "Order must be 2 or 3"
    assert isinstance(is_skew, bool), "is_skew must be a boolean"

    order_name = "oct" if order == 3 else "sext"
    return f"b{beam}_{order_name}_{'s' if is_skew else 'n'}"

def get_rdts(order: int, is_skew: bool) -> list[str]:
    """Return the RDTs for the given order and skew."""
    rdt_map = {
        (2, True): SKEW_RDTS3,
        (3, True): SKEW_RDTS4,
        (2, False): NORMAL_RDTS3,
        (3, False): NORMAL_RDTS4,
    }
    return rdt_map.get((order, is_skew))

def get_tbt_name(beam: int, order: int, is_skew: bool, sdds: bool = True) -> str:
    """Return the name of the TBT file for the given test parameters."""
    return f"tbt_data_{get_file_ext(beam, order, is_skew)}.{'sdds' if sdds else 'tfs'}"

def get_model_dir(beam: int, order: int, is_skew: bool) -> Path:
    """Return the model directory for the given test parameters."""
    return TEST_DIR / f"model_{get_file_ext(beam, order, is_skew)}"

def get_max_rdt_order(rdts: list[str]) -> int:
    """Return the maximum order of the RDTs."""
    return max(sum(int(num) for num in rdt.split("_")[0][1:]) for rdt in rdts)

def get_output_dir(tbt_name: str, output_dir: Path = None) -> Path:
    """Return the output directory for the given TBT, and create it if it does not exist."""
    if output_dir is None:
        output_dir = ANALYSIS_DIR / f"{tbt_name.split('.')[0]}"
        output_dir.mkdir(exist_ok=True)
    return output_dir

def do_analysis_needed(rdts: list[str], output_dir: Path, rdt_type: str, order_name: str) -> bool:
    """Check if the analysis needs to be done for the given RDTs."""
    return any(
        not (output_dir / f"{RDT_FOLDER}/{rdt_type}_{order_name}/{rdt}.tfs").exists()
        for rdt in rdts
    )

def get_rdts_from_harpy(
    beam: int,
    order: int,
    is_skew: bool,
    output_dir: Path = None,
    check_previous=False,
) -> dict[str, tfs.TfsDataFrame]:
    """
    Run the optics analysis for the given test parameters and return the RDTs.

    If output_dir is None, the output directory will be created in the rdt_constants.ANALYSIS_DIR.
    If check_previous is True, the analysis will only be done if the output files do not exist.    
    """
    
    rdts = get_rdts(order, is_skew)
    only_coupling = all(rdt.lower() in ["f1001", "f1010"] for rdt in rdts)
    
    rdt_type = "skew" if is_skew else "normal"
    order_name = "octupole" if order == 3 else "sextupole"
    rdt_order = get_max_rdt_order(rdts)
    
    tbt_name = get_tbt_name(beam, order, is_skew)
    output_dir = get_output_dir(tbt_name, output_dir)

    if not check_previous or do_analysis_needed(rdts, output_dir, rdt_type, order_name):
        hole_in_one_entrypoint(
            files=[FREQ_OUT_DIR / tbt_name],
            outputdir=output_dir,
            optics=True,
            accel="lhc",
            beam=beam,
            year="2024",
            energy=6.8,
            model_dir=get_model_dir(beam, order, is_skew),
            only_coupling=only_coupling,
            compensation="none",
            nonlinear=["rdt"],
            rdt_magnet_order=rdt_order,
        )

    dfs = {
        rdt: filter_IPs(tfs.read(output_dir / f"{RDT_FOLDER}/{rdt_type}_{order_name}/{rdt}.tfs", index="NAME"))
        for rdt in rdts
    }
    return dfs

def run_harpy(beam: int, order: int, is_skew: bool) -> None:
    """Run Harpy for the given test parameters."""
    hole_in_one_entrypoint(
        harpy=True,
        files=[DATA_DIR / get_tbt_name(beam, order, is_skew)],
        outputdir=FREQ_OUT_DIR,
        to_write=["lin", "spectra"],
        opposite_direction=beam == 2,
        tunes=[0.28, 0.31, 0.0],
        natdeltas=[0.0, -0.0, 0.0],
        turn_bits=18,
    )