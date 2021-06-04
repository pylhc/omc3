from omc3.response_creator import create_response_entrypoint as create_response
from pandas.testing import assert_frame_equal
from omc3.correction.constants import (BETA, DISP, NORM_DISP, F1001, F1010, TUNE, PHASE, VALUE, ERROR,
                                       ERR, WEIGHT, DELTA)

RMS_TOL_DICT_CORRECTION = {
    f"{PHASE}X": 3.0,
    f"{PHASE}Y": 3.0,
    f"{BETA}X": 25.0,
    f"{BETA}Y": 25.0,
    f"{DISP}X": 3.0,
    f"{DISP}Y": 1,
    f"{NORM_DISP}X": 2.0,
    f"{NORM_DISP}Y": 1.0,
    f"{TUNE}": 3.0,
    f"{F1001}R": 1.0,
    f"{F1001}I": 1.0,
    f"{F1010}R": 1.0,
    f"{F1010}I": 1.0,
}


def _assert_response_madx(
        accel_settings,
        correction_dir,
        variable_categories,
        optics_params,
        comparison_fullresponse_path,
        delta_k=0.00002,
):
    fullresponse_path = correction_dir / "Fullresponse_pandas_omc3"
    create_response_entrypoint(
        **accel_settings,
        creator="madx",
        delta_k=delta_k,
        variable_categories=variable_categories,
        outfile_path=fullresponse_path,
    )

    with open(fullresponse_path, "rb") as fullresponse_file:
        fullresponse_data = pickle.load(fullresponse_file)

    with open(comparison_fullresponse_path, "rb") as comparison_fullresponse_file:
        comparison_fullresponse_data = pickle.load(comparison_fullresponse_file)

    # is_equal = True
    for key in fullresponse_data.keys():
        if key in optics_params:
            assert np.allclose(
                fullresponse_data[key][comparison_fullresponse_data[key].columns].to_numpy(),
                comparison_fullresponse_data[key].to_numpy(),
                rtol=1e-04,
                atol=1e-06,
            ), f"Fulresponse does not match for a key {key}"


def _assert_response_twiss(
        accel_settings,
        correction_dir,
        variable_categories,
        comparison_fullresponse_path,
        RMS_tol_dict,
        delta_k=0.00002,
):
    fullresponse_path = correction_dir / "Fullresponse_pandas_omc3"
    create_response_entrypoint(
        **accel_settings,
        creator="twiss",
        delta_k=delta_k,
        variable_categories=variable_categories,
        outfile_path=fullresponse_path,
    )

    with open(fullresponse_path, "rb") as fullresponse_file:
        fullresponse_data = pickle.load(fullresponse_file)

    with open(comparison_fullresponse_path, "rb") as comparison_fullresponse_file:
        comparison_fullresponse_data = pickle.load(comparison_fullresponse_file)

    # is_equal = True
    for key in fullresponse_data.keys():
        index = comparison_fullresponse_data[key].index
        columns = comparison_fullresponse_data[key].columns
        delta = (
                fullresponse_data[key].loc[index, columns].to_numpy()
                - comparison_fullresponse_data[key].to_numpy()
        )
        assert np.sqrt(np.mean(delta ** 2)) < RMS_tol_dict[key], (
            f"RMS difference between twiss and madx response is not within "
            f"tolerance {RMS_tol_dict[key]} for key {key}"
        )
