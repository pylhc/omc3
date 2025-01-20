"""
Run as module to avoid import errors.

```
python -m tests.inputs.kmod.create_reference_data
```  
"""
from pathlib import Path

import tfs

from omc3.model.constants import TWISS_ELEMENTS_DAT
from omc3.model_creator import create_instance_and_model
from tests.unit.test_kmod_averaging import update_reference_files as update_averages
from tests.unit.test_kmod_import import get_model_path
from tests.unit.test_kmod_import import update_reference_files as update_imports
from tests.unit.test_kmod_lumi_imbalance import (
    update_reference_files as update_lumi_imbalances,
)

THIS_DIR = Path(__file__).parent


def create_model(beam: int):
    # Create ---
    create_instance_and_model(
        beam=beam,
        outputdir=THIS_DIR / f"b{beam}_model",
        accel="lhc",
        type="nominal",
        nat_tunes=(0.28, 0.31),
        drv_tunes=(0.27, 0.322),
        driven_excitation="acd",
        dpp=0.0,
        energy=6800.0,
        modifiers="R2024aRP_A22cmC22cmA10mL200cm.madx",
        fetch="afs",
        year=2024,
    )

    # Compress ---
    df = tfs.read(THIS_DIR / f"b{beam}_model" / TWISS_ELEMENTS_DAT, index="NAME")
    df = df.loc[df.index.str.match("^(M|B|IP\d$)"), ["S", "BETX", "BETY"]]
    tfs.write(get_model_path(beam), df, save_index="NAME")


if __name__ == "__main__":
    # Create Twiss Files ---
    # for beam in (1, 2):
    #     create_model(beam)

    # Update references ---
    update_averages()
    update_imports()
    update_lumi_imbalances()
