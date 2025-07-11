import numpy as np
import pytest
import tfs
from pandas.testing import assert_frame_equal

from omc3.correction import response_madx, response_twiss
from omc3.correction.constants import VALUE
from omc3.correction.handler import (
    _update_response,
    create_corrected_model,
    get_measurement_data,
)
from omc3.global_correction import CORRECTION_DEFAULTS, OPTICS_PARAMS_CHOICES
from omc3.global_correction import global_correction_entrypoint as global_correction
from omc3.model import manager
from omc3.model.constants import TWISS_DAT
from omc3.optics_measurements.constants import (
    BETA,
    BETA_NAME,
    DELTA,
    DISPERSION,
    DISPERSION_NAME,
    EXT,
    F1001,
    F1001_NAME,
    F1010,
    F1010_NAME,
    NORM_DISP_NAME,
    NORM_DISPERSION,
    PHASE,
    PHASE_NAME,
    TUNE,
)
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from tests.accuracy.test_global_correction import get_normal_params, get_skew_params

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
        model=model_inj_beams.model_dir / TWISS_DAT,
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

    fake_measurement(
        model=model_inj_beams.model_dir / TWISS_DAT,
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

    correction = tfs.read(tmp_path / f"{CORRECTION_DEFAULTS['output_filename']}.tfs", index="NAME")
    assert not np.any(correction["DELTA"].isna())
    assert not correction["DELTA"].isin([np.inf, -np.inf]).any()

    if method == "omp":
        assert len(correction.index) == n_correctors

    if method == "pinv":
        assert len(correction.index) > n_correctors  # unless by accident

@pytest.mark.basic
def test_update_response(tmp_path, model_inj_beams):
    """ Tests if the response is updated. """
    # create the accelerator instance
    knob = "kqd.a78"#f'kq10.l1b{beam}'
    accel_inst = manager.get_accelerator(model_inj_beams)
    delta = tfs.TfsDataFrame(
        3.34e-05,
        index=[knob],
        columns=[DELTA]
        )
    optics_params = ['PHASEX', 'PHASEY']

    # Get a response dict from the model
    ref_resp_dict = response_madx.create_fullresponse(accel_inst, [knob])

    corr_model_path = tmp_path / f"twiss_cor{EXT}"
    ref_model = create_corrected_model(corr_model_path, [], accel_inst, update_dpp=False)

    # As orbit_dpp is not in the response, it should not be updated. First for response_madx
    new_resp_dict = _update_response(
        accel_inst=accel_inst,
        corrected_elements=ref_model,
        optics_params=optics_params,
        corr_files=[],
        variable_categories=[knob],
        update_dpp=False,
        update_response="madx"
    )

    # Check everything is the same, as model has not changed.
    for key in ref_resp_dict.keys():
        assert_frame_equal(ref_resp_dict[key], new_resp_dict[key])

    corr_file = tmp_path / "corr.madx"
    with open(corr_file, "w") as f:
        f.write(f"{knob} = {knob}{delta.loc[knob, DELTA]:+.16e};")

    # Now for response_madx with the model changed
    new_resp_dict = _update_response(
        accel_inst=accel_inst,
        corrected_elements=ref_model, # This is not used in response_madx
        optics_params=optics_params,
        corr_files=[corr_file],
        variable_categories=[knob],
        update_dpp=False,
        update_response="madx"
    )
    for key in new_resp_dict.keys():
        # If the original response is 0, there is no reason why the new one should not be.
        if not ref_resp_dict[key][knob].sum() == 0:
            # The values are not the same, as the model has changed.
            assert not ref_resp_dict[key].equals(new_resp_dict[key])

            # Check the general structure of the response
            assert new_resp_dict[key].columns.equals(ref_resp_dict[key].columns)
            assert new_resp_dict[key].index.equals(ref_resp_dict[key].index)

    # Now for response_twiss
    ref_resp_dict = response_twiss.create_response(accel_inst, [knob], optics_params)

    new_resp_dict = _update_response(
        accel_inst=accel_inst,
        corrected_elements=accel_inst.elements,
        optics_params=optics_params,
        #corr_files are not used in response_twiss but here to check it isn't used (as accel_inst.modifiers is always changed)
        corr_files=[],
        variable_categories=[knob],
        update_dpp=False,
        update_response="twiss"
    )

    # Check everything is the same, as model is the reference model.
    for key in ref_resp_dict.keys():
        assert_frame_equal(ref_resp_dict[key], new_resp_dict[key])

    corr_model = create_corrected_model(corr_model_path, [corr_file], accel_inst, update_dpp=False)
    assert not ref_model.equals(corr_model) # For debugging - if this is the problem, or response_twiss

    new_resp_dict = _update_response(
        accel_inst=accel_inst,
        corrected_elements=corr_model,
        optics_params=optics_params,
        corr_files=[],
        variable_categories=[knob],
        update_dpp=False,
        update_response=True
    )
    for key in new_resp_dict.keys(): # The new dict will only have the keys for the selected optics_params
        # The values are not the same, as the model has changed.
        assert not ref_resp_dict[key].equals(new_resp_dict[key])

        # Check the general structure of the response
        assert knob in new_resp_dict[key].columns
        assert new_resp_dict[key].index.equals(ref_resp_dict[key].index)
