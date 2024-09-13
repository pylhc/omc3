import pytest
from pandas.testing import assert_frame_equal

from omc3.correction.constants import VALUE
from omc3.correction.handler import get_measurement_data
from omc3.global_correction import OPTICS_PARAMS_CHOICES, global_correction_entrypoint as global_correction
from omc3.optics_measurements.constants import (BETA, DISPERSION, NORM_DISPERSION, F1001, F1010,
                                                DISPERSION_NAME, BETA_NAME, PHASE_NAME,
                                                NORM_DISP_NAME, F1010_NAME, TUNE, PHASE,
                                                F1001_NAME)
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from tests.accuracy.test_global_correction import get_skew_params, get_normal_params

FILENAME_MAP = {
    # Names to be output on input of certain parameters.
    f'{BETA}X': f"{BETA_NAME}x",
    f'{BETA}Y': f"{BETA_NAME}y",
    f'{DISPERSION}X': f"{DISPERSION_NAME}x",
    f'{DISPERSION}Y': f"{DISPERSION_NAME}y",
    f'{PHASE}X': f"{PHASE_NAME}x",
    f'{PHASE}Y': f"{PHASE_NAME}y",
    f'{F1010}I': F1010_NAME,
    f'{F1010}R': F1010_NAME,
    f'{F1001}I': F1001_NAME,
    f'{F1001}R': F1001_NAME,
    f'{NORM_DISPERSION}X': f"{NORM_DISP_NAME}x",
}


@pytest.mark.basic
@pytest.mark.parametrize('orientation', ('skew', 'normal'))
def test_read_measurement_data(tmp_path, model_inj_beams, orientation):
    """ Tests if all necessary measurement data is read.
    Hint: the `model_inj_beam1` fixture is defined in `conftest.py`."""
    is_skew = orientation == 'skew'
    beam = model_inj_beams.beam
    correction_params = get_skew_params(beam) if is_skew else get_normal_params(beam)
    meas_fake = fake_measurement(
        model=model_inj_beams.model_dir / "twiss.dat",
        twiss=correction_params.twiss,
        randomize=[],
        relative_errors=[0.1],
        outputdir=tmp_path,
    )
    _, meas_dict = get_measurement_data(
        OPTICS_PARAMS_CHOICES,
        meas_dir=tmp_path,
        beta_filename='beta_phase_',
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
@pytest.mark.parametrize('method', ('pinv', 'omp'))
def test_lhc_global_correct_methods(tmp_path, model_inj_beams, method):
    """Creates a fake measurement from a modfied model-twiss with
    quadrupole errors and runs global correction on this measurement.
    Hint: the `model_inj_beam1` fixture is defined in `conftest.py`."""
    beam = model_inj_beams.beam
    correction_params = get_normal_params(beam)

    # create and load fake measurement
    n_correctors = 5

    meas_fake = fake_measurement(
        model=model_inj_beams.model_dir / "twiss.dat",
        twiss=correction_params.twiss,
        randomize=[],
        relative_errors=[0.1],
        outputdir=tmp_path,
    )

    # Perform global correction
    global_correction(
        **model_inj_beams,
        # correction params
        meas_dir=tmp_path,
        variable_categories=correction_params.variables,
        fullresponse_path=model_inj_beams.model_dir / correction_params.fullresponse,
        optics_params=correction_params.optics_params,
        output_dir=tmp_path,
        weights=correction_params.weights,
        svd_cut=0.01,
        method=method,
        n_correctors=n_correctors,
    )
