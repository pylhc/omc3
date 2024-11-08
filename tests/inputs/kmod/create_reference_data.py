"""
Run as module to avoid import errors.

```
python -m tests.inputs.kmod.create_reference_data
```  

To create the models:

```
python -m omc3.model_creator --outputdir ./tests/inputs/kmod/b1_model --accel lhc --type nominal --nat_tunes 0.28 0.31 --drv_tunes 0.27 0.322 --driven_excitation acd --dpp 0.0 --energy 6800.0 --modifiers R2024aRP_A22cmC22cmA10mL200cm.madx  --fetch afs --beam 1 --year 2024
python -m omc3.model_creator --outputdir ./tests/inputs/kmod/b2_model --accel lhc --type nominal --nat_tunes 0.28 0.31 --drv_tunes 0.27 0.322 --driven_excitation acd --dpp 0.0 --energy 6800.0 --modifiers R2024aRP_A22cmC22cmA10mL200cm.madx  --fetch afs --beam 2 --year 2024
```
"""
import tfs
from pathlib import Path
from tests.unit.test_kmod_averaging import update_reference_files as update_averages
from tests.unit.test_kmod_import import update_reference_files as update_imports, _get_model_path 
from tests.unit.test_kmod_lumi_imbalance import update_reference_files as update_lumi_imbalances

THIS_DIR = Path(__file__).parent

# Compress models
for beam in (1, 2):
    df = tfs.read(THIS_DIR / f"b{beam}_model" / "twiss_elements.dat", index="NAME")
    df = df.loc[df.index.str.match("^(M|B|IP\d$)"), ["S", "BETX", "BETY"]]
    tfs.write(_get_model_path(beam), df, save_index="NAME")

# Update references
update_averages()
update_imports()
update_lumi_imbalances()
