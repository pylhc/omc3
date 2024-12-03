"""
Response Creator
----------------

Provides a response generation wrapper.
The response matrices can be either created by :mod:`omc3.correction.response_madx`
or analytically via :mod:`omc3.correction.response_twiss`.

Input arguments are split into response creation arguments and accelerator arguments.
The former are listed below, the latter depend on the accelerator you want
to use. Check :ref:`modules/model:Model` to see which ones are needed.


**Arguments:**

*--Optional--*


- **outfile_path** *(str)*:

    Name of fullresponse file.


- **creator** *(str)*:

    Create either with madx or analytically from twiss file.

    choices: ``('madx', 'twiss')``

    default: ``madx``


- **debug**:

    Print debug information.

    action: ``store_true``


- **delta_k** *(float)*:

    Delta K1 to be applied to quads for sensitivity matrix (madx-only).

    default: ``2e-05``


- **optics_params** *(str)*:

    List of parameters to correct upon (e.g. BBX BBY; twiss-only).


- **variable_categories**:

    List of the variables classes to use.

    default: ``['MQM', 'MQT', 'MQTL', 'MQY']``


"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from generic_parser.entrypoint_parser import DotDict, EntryPointParameters, entrypoint

from omc3.correction import response_madx, response_twiss
from omc3.correction.response_io import write_fullresponse
from omc3.global_correction import CORRECTION_DEFAULTS, OPTICS_PARAMS_CHOICES
from omc3.model import manager
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config

LOG = logging_tools.get_logger(__name__)


def response_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="creator",
        type=str,
        choices=("madx", "twiss"),
        default="madx",
        help="Create either with madx or analytically from twiss file.",
    )
    params.add_parameter(
        name="variable_categories",
        nargs="+",
        default=CORRECTION_DEFAULTS["variable_categories"],
        help="List of the variables classes to use.",
    )
    params.add_parameter(
        name="outfile_path",
        type=PathOrStr,
        help="Name of fullresponse file.",
    )
    params.add_parameter(
        name="delta_k",
        type=float,
        default=0.00002,
        help="Delta K1 to be applied to quads for sensitivity matrix (madx-only).",
    )
    params.add_parameter(
        name="optics_params",
        type=str,
        nargs="+",
        choices=OPTICS_PARAMS_CHOICES,
        help="List of parameters to correct upon (e.g. BBX BBY; twiss-only).",
    )
    params.add_parameter(
        help="Print debug information.",
        name="debug",
        action="store_true",
    )
    return params


@entrypoint(response_params())
def create_response_entrypoint(opt: DotDict, other_opt) -> dict[str, pd.DataFrame]:
    """Entry point for creating pandas-based response matrices.

    The response matrices can be either created by response_madx or TwissResponse.
    """
    LOG.info("Creating response.")
    if opt.outfile_path is not None:
        save_config(Path(opt.outfile_path).parent, opt=opt, unknown_opt=other_opt, script=__file__)

    accel_inst = manager.get_accelerator(other_opt)

    if opt.creator.lower() == "madx":
        fullresponse = response_madx.create_fullresponse(
            accel_inst, opt.variable_categories, delta_k=opt.delta_k
        )

    elif opt.creator.lower() == "twiss":
        fullresponse = response_twiss.create_response(accel_inst, opt.variable_categories, opt.optics_params)

    if opt.outfile_path is not None:
        write_fullresponse(opt.outfile_path, fullresponse)
    return fullresponse


# Script Mode ------------------------------------------------------------------


if __name__ == "__main__":
    create_response_entrypoint()
