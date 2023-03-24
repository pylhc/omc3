import pytest

import tfs
from omc3.correction.constants import NAME
from omc3.correction.model_appenders import add_coupling_to_model
from omc3.scripts.correction_test import correction_test_entrypoint
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement

from tests.accuracy.test_global_correction import get_skew_params, get_normal_params, _create_fake_measurement


# @pytest.mark.basic
# @pytest.mark.parametrize('orientation', ('skew', 'normal'))
# def test_lhc_corrections(tmp_path, model_inj_beams, orientation):
def test_lhc_corrections(tmp_path, model_inj_beam1, orientation="normal"):
    model_inj_beams = model_inj_beam1
    beam = model_inj_beams.beam
    correction_params = get_skew_params(beam) if orientation == 'skew' else get_normal_params(beam)
    _create_fake_measurement(tmp_path, model_inj_beams.model_dir, correction_params.twiss)

    correction_test_entrypoint(
        meas_dir=tmp_path,
        output_dir=tmp_path / "Corrections",
        corrections=[correction_params.correction_filename],
        # accelerator class params:
        **model_inj_beams
    )




def _create_fake_measurement(tmp_path, model_path, twiss_path):
    model_df = tfs.read(model_path / "twiss.dat", index=NAME)
    model_df = add_coupling_to_model(model_df)

    twiss_df = tfs.read(twiss_path, index=NAME)
    twiss_df = add_coupling_to_model(twiss_df)

    # create fake measurement data
    fake_measurement(
        model=model_df,
        twiss=twiss_df,
        randomize=[],
        outputdir=tmp_path,
    )

