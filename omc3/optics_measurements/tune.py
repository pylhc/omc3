"""
.. module: tune

Created on 17/07/18

:author: Lukas Malina

It computes betatron tunes and provides structures to store them.
"""
import numpy as np
from utils import stats
PLANES = ('X', 'Y')
CHAR = {"X": "1", "Y": "2"}


def calculate_tunes(measure_input, input_files):
    tune_d = TuneDict()
    accelerator = measure_input.accelerator
    for plane in PLANES:
        tune_d[plane]["QM"] = accelerator.get_model_tfs().headers["Q" + CHAR[plane]]
        tune_list = [df.headers["Q" + CHAR[plane]] for df in input_files.zero_dpp_frames(plane)]
        tune_rms_list = [df.headers["Q" + CHAR[plane] + "RMS"] for df in input_files.zero_dpp_frames(plane)]
        measured_tune = stats.weighted_mean(np.array(tune_list), errors=np.array(tune_rms_list))
        tune_d[plane]["Q"], tune_d[plane]["QF"] = measured_tune, measured_tune
        tune_d[plane]["QFM"] = accelerator.nat_tune_x if plane is "X" else accelerator.nat_tune_y
        if accelerator.excitation:
            tune_d[plane]["QM"] = accelerator.drv_tune_x if plane is "X" else accelerator.drv_tune_y
            tune_d[plane]["QF"] = tune_d[plane]["Q"] - tune_d[plane]["QM"] + tune_d[plane]["QFM"]
    return tune_d


class TuneDict(dict):
    """
    Data structure to hold tunes
    """
    def __init__(self):
        super(TuneDict, self).__init__(zip(PLANES, ({"Q": 0.0, "QF": 0.0, "QM": 0.0, "QFM": 0.0},
                                                    {"Q": 0.0, "QF": 0.0, "QM": 0.0, "QFM": 0.0})))

    def get_lambda(self, plane):
        """
        Computes lambda compensation factor

        Args:
            plane: X or Y

        Returns:
             lambda compensation factor (driven vs free motion)
        """
        return (np.sin(np.pi * (self[plane]["Q"] - self[plane]["QF"])) /
                np.sin(np.pi * (self[plane]["Q"] + self[plane]["QF"])))
