import shutil
from pathlib import Path

import tfs
import turn_by_turn as tbt

from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.model.constants import TWISS_ELEMENTS_DAT
from omc3.optics_measurements import phase

N_TURNS = 200       
THIS_DIR = Path(__file__).parent
SDDS_FILE = THIS_DIR / "MULTITURN_ACQ__18-11-24_16-04-57.sdds"  # /user/spsop/MultiTurn/
MODEL_DIR = THIS_DIR / "model_Q20_noacd" 
TEST_FILE = THIS_DIR / f"sps_{N_TURNS}turns.sdds"


def reduce_tbt_turns():
    """ Reduce the amount of turns in the tbt file to N_TURNS. """
    tbt_data = tbt.read_tbt(SDDS_FILE,  datatype="sps") 
    tbt_data.matrices[0].X = tbt_data.matrices[0].X.loc[:, :N_TURNS]
    tbt_data.matrices[0].Y = tbt_data.matrices[0].Y.loc[:, :N_TURNS]
    tbt_data.nturns = N_TURNS
    tbt.write_tbt(TEST_FILE, tbt_data, datatype="sps")


def reduce_twiss_elements():
    """ Remove DRIFTS from twiss_elements (renamed to twiss_elements_large beforehand). """
    file_in = MODEL_DIR / "twiss_elements_large.dat"
    file_out = MODEL_DIR / TWISS_ELEMENTS_DAT

    df = tfs.read(file_in, index="NAME")
    df = df.loc[~df.index.str.match("DRIFT"), :]
    tfs.write(file_out, df, save_index="NAME")


def run_analysis():
    """ Run the omc3 analysis on the sps data. 
    To test if it works and also to create the spectrum files for the spectrum plot tests.
    Similar to what is done in `tests/unit/test_hole_in_one/test_hole_in_one_sps`.
    """
    output = THIS_DIR / "tmp"
    hole_in_one_entrypoint(
        harpy=True,
        optics=True,  # do not need to run optics, but good to test if it works
        clean=True,
        tbt_datatype="sps",
        compensation=phase.CompensationMode.NONE,
        output_bits=8,
        turn_bits=12,  # lower and rdt calculation fails for coupling
        resonances=4,
        # nattunes = [0.13, 0.18, 0.0],
        autotunes="transverse",
        outputdir=output,
        files=[TEST_FILE],
        model=MODEL_DIR / TWISS_ELEMENTS_DAT,
        to_write=["lin", "spectra",],
        window="hann",
        coupling_method=2,
        nonlinear=['rdt',],
        rdt_magnet_order=4,
        unit="mm",
        accel="generic",
        model_dir=MODEL_DIR,
        three_bpm_method=True,  # n-bpm method needs error-def file
    )
    shutil.move(output / "lin_files", THIS_DIR)
    shutil.rmtree(output)


if __name__ == "__main__":
    # reduce_tbt_turns()
    # reduce_twiss_elements()
    # run_analysis()
    pass