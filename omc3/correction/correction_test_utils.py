"""
Correction Test Utils
---------------------

Utility functions used by `correction_test.py` as well as `plotting/plot_correction_test.py`.
"""
from typing import Dict, Union

import pandas as pd

from generic_parser.entrypoint_parser import EntryPointParameters
from omc3.plotting.plot_optics_measurements import get_optics_style_params, get_plottfs_style_params
from omc3.utils import logging_tools
from tfs import TfsDataFrame

Measurements = Dict[str, Union[TfsDataFrame, pd.DataFrame]]


LOG = logging_tools.get_logger(__name__)


def get_plotting_style_parameters():
    """ Parameters related to the style of the plots. """
    params = EntryPointParameters()
    params.update(get_optics_style_params())
    params.update(get_plottfs_style_params())
    params["plot_styles"]["default"] = params["plot_styles"]["default"] + ["correction_test"]
    return params




