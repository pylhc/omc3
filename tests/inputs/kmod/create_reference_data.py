"""
Run as module to avoid import errors.

```
python -m tests.inputs.kmod.create_reference_data
```  
"""
from tests.unit.test_kmod_averaging import update_reference_files as update_averages
from tests.unit.test_kmod_import import update_reference_files as update_imports
from tests.unit.test_kmod_lumi_imbalance import update_reference_files as update_lumi_imbalances

update_averages()
update_imports()
update_lumi_imbalances()