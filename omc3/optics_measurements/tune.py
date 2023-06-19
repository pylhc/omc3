"""
Tune
----
This module contains tune calculations functionality of ``optics_measurements``.
It provides functions to compute betatron tunes and structures to store them.
"""
import numpy as np
import pandas as pd

from omc3.definitions.constants import PLANES, PLANE_TO_NUM
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


class TuneDict(dict):
    """
    Data structure to hold tunes.
    """
    # TODO: detail each key
    def __init__(self):
        super(TuneDict, self).__init__(zip(PLANES, ({"Q": 0.0, "QF": 0.0, "QM": 0.0, "QFM": 0.0, "ac2bpm": None},
                                                    {"Q": 0.0, "QF": 0.0, "QM": 0.0, "QFM": 0.0, "ac2bpm": None})))

    def get_lambda(self, plane):
        """
        Computes lambda compensation factor.

        Args:
            plane: marking the horizontal or vertical plane, **X** or **Y**.

        Returns:
             lambda compensation factor (driven vs free motion).
        """
        return (np.sin(np.pi * (self[plane]["Q"] - self[plane]["QF"])) /
                np.sin(np.pi * (self[plane]["Q"] + self[plane]["QF"])))

    def phase_ac2bpm(self, df_idx_by_bpms: pd.DataFrame, plane: str, accelerator):
        """
        Returns the necessary values for the exciter compensation.
        See **DOI: 10.1103/PhysRevSTAB.11.084002**

        Args:
            df_idx_by_bpms (pandas.DataFrame): commonbpms (see GetLLM._get_commonbpms)
            plane (str): marking the horizontal or vertical plane, **X** or **Y**.
            accelerator: an `Accelerator` object.

        Returns:
            A `Tuple` consisting of four elements a, b, c, d.
                - a (string): name of the nearest BPM.
                - b (float): compensated phase advance between the exciter and the nearest BPM.
                - c (int): k of the nearest BPM.
                - d (string): name of the exciter element.
        """
        model = accelerator.elements
        r = self.get_lambda(plane)
        [k, bpmac1], exciter = accelerator.get_exciter_bpm(plane, df_idx_by_bpms.index)
        psi = model.loc[bpmac1, f"MU{plane}"] - model.loc[exciter, f"MU{plane}"]
        psi = np.arctan((1 + r) / (1 - r) * np.tan(
            2 * np.pi * psi + np.pi * self[plane]["QF"])) % np.pi - np.pi * self[plane]["Q"]
        psi = psi / (2 * np.pi)
        return bpmac1, psi, k, exciter
