"""
Tune
----
This module contains tune calculations functionality of ``optics_measurements``.
It provides functions to compute betatron tunes and structures to store them.
"""
import numpy as np
import pandas as pd

from omc3.definitions.constants import PLANES, PLANE_TO_NUM
from omc3.definitions.structures import TuneDict
from omc3.utils import stats


def calculate(measure_input, input_files):
    tune_d = TuneDict()
    accelerator = measure_input.accelerator
    for plane in PLANES:
        tune_d[plane]["QM"] = accelerator.model.headers[f"Q{PLANE_TO_NUM[plane]}"]
        tune_list = [df.headers[f"Q{PLANE_TO_NUM[plane]}"] for df in input_files.dpp_frames(plane, 0)]
        tune_rms_list = [df.headers[f"Q{PLANE_TO_NUM[plane]}RMS"] for df in input_files.dpp_frames(plane, 0)]
        measured_tune = stats.weighted_mean(np.array(tune_list), errors=np.array(tune_rms_list))
        tune_d[plane]["Q"], tune_d[plane]["QF"] = measured_tune, measured_tune
        tune_d[plane]["QFM"] = accelerator.nat_tunes[PLANE_TO_NUM[plane] - 1]
        if accelerator.excitation:
            tune_d[plane]["QM"] = accelerator.drv_tunes[PLANE_TO_NUM[plane] - 1]
            tune_d[plane]["QF"] = tune_d[plane]["Q"] - tune_d[plane]["QM"] + tune_d[plane]["QFM"]
            if measure_input.compensation == "equation":
                tune_d[plane]["ac2bpm"] = tune_d.phase_ac2bpm(
                    input_files.joined_frame(plane, [f"MU{plane}"], dpp_value=0, how='inner'),
                    plane, measure_input.accelerator)
    return tune_d
