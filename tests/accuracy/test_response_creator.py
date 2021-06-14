import numpy as np
import pytest

from omc3.correction.constants import (DISP, PHASE, PHASE_ADV)
from omc3.correction.handler import _rms
from omc3.correction.response_io import read_fullresponse
from omc3.global_correction import OPTICS_PARAMS_CHOICES
from omc3.response_creator import create_response_entrypoint as create_response
from tests.accuracy.test_global_correction import get_skew_params, get_normal_params

DELTA_K = 2e-5
MADX_RTOL = 1e-3
MADX_ATOL = 1e-8 / DELTA_K  # tfs-precision / DELTA
TWISS_RMS_TOL = 0.07


@pytest.mark.basic
@pytest.mark.parametrize('orientation', ('skew', 'normal'))
@pytest.mark.parametrize('creator', ('madx', 'twiss'))
def test_reponse_accuracy(tmp_path, model_inj_beam1, orientation, creator):
    """ Tests the accuracy of a newly generated response against the saved
    response matrix. In that way also twiss and madx responses are compared to
    each other.
    Hint: the `model_inj_beam1` fixture is defined in `conftest.py`."""
    # parameter setup
    is_skew = orientation == 'skew'
    _, optics_params, variables, fullresponse, _ = get_skew_params() if is_skew else get_normal_params()
    optics_params = _adapt_optics_params(optics_params, creator, is_skew)

    # response creation
    new_response = create_response(
        **model_inj_beam1.settings,
        creator=creator,
        delta_k=DELTA_K,
        variable_categories=variables,
    )

    # compare to original response matrix
    original_response = read_fullresponse(model_inj_beam1.path / fullresponse)
    for key in optics_params:
        original = original_response[key]
        new = new_response[key].loc[original.index, original.columns]

        # ######## Relative RMS check ###############
        # print(key)
        # print(_rms(original - new)/_rms(original))
        # ###########################################

        if creator == "madx":
            check = np.allclose(original, new, rtol=MADX_RTOL, atol=MADX_ATOL)
        else:
            check = (_rms(original - new)/_rms(original)).mean() < TWISS_RMS_TOL

        assert check, f"Fullresponse via {creator} does not match for {key}"


# Helper -----------------------------------------------------------------------


def _adapt_optics_params(optics_params, creator, is_skew):
    """Changes the optics parameters stemming from the defaults
     so that they fit the creation functions."""
    if creator == 'madx':
        # madx creates responses for all anyway, so test all parameters
        optics_params = OPTICS_PARAMS_CHOICES
    elif is_skew:
        # twiss calculates (N)D[X,Y] but (N)DX is all zero and NDY not in madx.
        optics_params = list(optics_params) + [f"{DISP}Y"]

    # replace PHASE with MU, as the responses operate on MU
    return [f"{PHASE_ADV}{k[-1]}" if k[:-1] == PHASE else k for k in optics_params]
