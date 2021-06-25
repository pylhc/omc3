import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from omc3.correction.constants import (BETA, DISP, NORM_DISP, F1001, F1010, TUNE, PHASE, VALUE)
from omc3.correction.handler import get_measurement_data, _rms
from omc3.global_correction import OPTICS_PARAMS_CHOICES
from omc3.optics_measurements.constants import (DISPERSION_NAME, BETA_NAME, PHASE_NAME, NORM_DISP_NAME)
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from tests.accuracy.test_global_correction import get_skew_params, get_normal_params

FILENAME_MAP = {
    # Names to be output on input of certain parameters.
    f'{BETA}X': f"{BETA_NAME}x",
    f'{BETA}Y': f"{BETA_NAME}y",
    f'{DISP}X': f"{DISPERSION_NAME}x",
    f'{DISP}Y': f"{DISPERSION_NAME}y",
    f'{PHASE}X': f"{PHASE_NAME}x",
    f'{PHASE}Y': f"{PHASE_NAME}y",
    f'{F1010}I': F1010.lower(),
    f'{F1010}R': F1010.lower(),
    f'{F1001}I': F1001.lower(),
    f'{F1001}R': F1001.lower(),
    f'{NORM_DISP}X': f"{NORM_DISP_NAME}x",
}


@pytest.mark.basic
@pytest.mark.parametrize('orientation', ('skew', 'normal'))
def test_read_measurement_data(tmp_path, model_inj_beams, orientation):
    """ Tests if all necessary measurement data is read.
    Hint: the `model_inj_beam1` fixture is defined in `conftest.py`."""
    is_skew = orientation == 'skew'
    beam = model_inj_beams.beam
    twiss, optics_params, variables, fullresponse, _ = get_skew_params(beam) if is_skew else get_normal_params(beam)
    meas_fake = fake_measurement(
        model=model_inj_beams.model_dir / "twiss.dat",
        twiss=twiss,
        randomize=[],
        relative_errors=[0.1],
        outputdir=tmp_path,
    )
    _, meas_dict = get_measurement_data(
        OPTICS_PARAMS_CHOICES,
        meas_dir=tmp_path,
        beta_file_name='beta_phase_',
    )
    assert len(meas_dict) == len(FILENAME_MAP) + 1  # + Q
    for key, df in meas_dict.items():
        # don't check for exact equality, because of read-write
        if key == TUNE:
            res_df = list(meas_fake.values())[0]
            for i in (1, 2):
                assert res_df.headers[f"{TUNE}{i}"] % 1 - df.loc[f"{TUNE}{i}", VALUE] < 1e-7
        else:
            res_df = meas_fake[FILENAME_MAP[key]]
            assert len(df.columns)
            assert_frame_equal(df, res_df.loc[:, df.columns])


@pytest.mark.basic
def test_rms():
    """ Tests the rms-function."""
    for _ in range(5):
        vec = np.random.rand(100)
        assert np.sqrt(np.mean(np.square(vec))) == _rms(vec)
