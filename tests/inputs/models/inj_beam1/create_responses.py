""" Script to create the varmap and response files in this model-folder. """
import shutil
import os
from omc3.response_creator import create_response_entrypoint
from omc3 import model
from tests.accuracy.test_global_correction import get_skew_params, get_normal_params
from tests.accuracy.test_response_creator import _adapt_optics_params
from pathlib import Path

DELTA_K = 2e-5
beam = 1

model_dir = Path(__file__).parent

macros_path = model_dir / "macros"
shutil.copytree(Path(model.__file__).parent / "madx_macros", macros_path)

for creator in ('twiss', 'madx'):
    for is_skew in (True, False):
        _, optics_params, variables, fullresponse, _ = get_skew_params(beam) if is_skew else get_normal_params(beam)
        optics_params = _adapt_optics_params(optics_params, creator, is_skew)

        # response creation
        new_response = create_response_entrypoint(
            ats=True,
            beam=beam,
            model_dir=model_dir,
            year="2018",
            accel="lhc",
            energy=0.45,
            creator=creator,
            delta_k=DELTA_K,
            variable_categories=variables,
            outfile_path=Path(f"fullresponse_{'_'.join(sorted(variables))}.h5")
        )

os.system(f'rm -r "{macros_path}"')