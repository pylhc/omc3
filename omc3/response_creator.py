"""
Provides a response generation wrapper.
The response matrices can be either created by response_madx or analytically via TwissResponse.

:author: Joschua Dillys
"""
import os
import pickle
from model import manager
from utils import logging_tools
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint
from correction import response_madx, response_twiss
from global_correct import CORRECTION_DEFAULTS

LOG = logging_tools.get_logger(__name__)


def response_params():
    params = EntryPointParameters()
    params.add_parameter(name="creator", type=str, choices=("madx", "twiss"),
                         default="madx", help="Create either with madx or analytically from twiss file.")
    params.add_parameter(name="variable_categories", nargs="+",
                         default=CORRECTION_DEFAULTS["variable_categories"],
                         help="List of the variables classes to use.")
    params.add_parameter(name="outfile_path", required=True, type=str,
                         help="Name of fullresponse file.")
    params.add_parameter(name="delta_k", type=float, default=0.00002,
                         help="Delta K1 to be applied to quads for sensitivity matrix (madx-only).")
    params.add_parameter(name="optics_params", type=str, nargs="+",
                         help="List of parameters to correct upon (e.g. BBX BBY; twiss-only).", )  # TODO add choices
    params.add_parameter(help="Print debug information.", name="debug", action="store_true",)
    return params

@entrypoint(response_params(), strict=False)
def create_response_entrypoint(opt, other_opt):
    """ Entry point for creating pandas-based response matrices.

    The response matrices can be either created by response_madx or TwissResponse.

    Keyword Args:
        Required
        outfile_path (str): Name of fullresponse file.
                            **Flags**: ['-o', '--outfile']
        Optional
        creator (str): Create either with madx or analytically from twiss file.
                       **Flags**: --creator
                       **Choices**: ('madx', 'twiss')
                       **Default**: ``madx``
        debug: Print debug information.
               **Flags**: --debug
               **Action**: ``store_true``
        delta_k (float): Delta K1L to be applied to quads for sensitivity matrix (madx-only).
                         **Flags**: ['-k', '--deltak']
                         **Default**: ``2e-05``
        optics_params (str): List of parameters to correct upon (e.g. BBX BBY; twiss-only).
                             **Flags**: --optics_params
        variable_categories: List of the variables classes to use.
                             **Flags**: --variables
                             **Default**: ``['MQM', 'MQT', 'MQTL', 'MQY']``
    """



if __name__ == "__main__":
    create_response_entrypoint()
