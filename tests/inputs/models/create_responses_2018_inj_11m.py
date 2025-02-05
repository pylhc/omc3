""" Script to create the varmap and response files in this model-folder. 

Run as module to avoid import errors.

```
python -m tests.inputs.models.create_responses_2018_inj_11m
```  
"""
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from omc3 import model
from omc3.response_creator import create_response_entrypoint
from tests.accuracy.test_global_correction import (
    CorrectionParameters,
    get_arc_by_arc_params,
    get_normal_params,
    get_skew_params,
)

DELTA_K = 2e-5
MODEL_DIR = Path(__file__).parent


def create_normal_and_skew_responses():
    for beam in (1, 2):
        for is_skew in (True, False):
            correction_params = get_skew_params(beam) if is_skew else get_normal_params(beam)
            for creator in ('twiss', 'madx'):  # run twiss first, override response by madx, leave varmap-file
                _create_response(beam, correction_params, creator)


def create_mqt_response():
    for beam in (1, 2):
        correction_params = get_arc_by_arc_params(beam)
        _create_response(beam, correction_params, 'madx')


def _create_response(beam: int, correction_params: CorrectionParameters, creator: str):
    model_dir = MODEL_DIR / f"2018_inj_b{beam}_11m"
    with cleanup(model_dir):
        create_response_entrypoint(
            ats=True,
            beam=beam,
            model_dir=model_dir,
            year="2018",
            accel="lhc",
            energy=0.45,
            creator=creator,
            delta_k=DELTA_K,
            variable_categories=correction_params.variables,
            outfile_path=model_dir / correction_params.fullresponse,
        )


@contextmanager
def cleanup(model_dir: Path):
    macros_dir = model_dir / "macros"
    shutil.copytree(Path(model.__file__).parent / "madx_macros", macros_dir)
    try:
        yield
    finally:
        os.system(f'rm -r "{macros_dir}"')
        os.system(f'rm -r "{model_dir / "job.iterate.0.madx"}"')
        os.system(f'rm -r "{model_dir / "response_madx_full.log.zip"}"')
        os.system(f'rm -r {model_dir / "response_creator_*.ini"}')


if __name__ == "__main__":
    # create_normal_and_skew_responses()
    create_mqt_response()