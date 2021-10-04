"""
Structures
----------

Custom objects used throughout ``omc3`` for specific needs.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from omc3.definitions.constants import PLANES


class TuneDict(dict):
    """
    Data structure to hold tunes.
    """

    def __init__(self):
        super().__init__(
            zip(
                PLANES,
                (
                    {"Q": 0.0, "QF": 0.0, "QM": 0.0, "QFM": 0.0, "ac2bpm": None},
                    {"Q": 0.0, "QF": 0.0, "QM": 0.0, "QFM": 0.0, "ac2bpm": None},
                ),
            )
        )

    def get_lambda(self, plane: str) -> float:
        """
        Computes lambda compensation factor.

        Args:
            plane (str): marking the horizontal or vertical plane, **X** or **Y**.

        Returns:
             lambda compensation factor (driven vs free motion).
        """
        return np.sin(np.pi * (self[plane]["Q"] - self[plane]["QF"])) / np.sin(
            np.pi * (self[plane]["Q"] + self[plane]["QF"])
        )

    def phase_ac2bpm(
        self, df_idx_by_bpms: pd.DataFrame, plane: str, accelerator: "Accelerator"
    ) -> Tuple[str, float, int, str]:
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
        psi = (
            np.arctan((1 + r) / (1 - r) * np.tan(2 * np.pi * psi + np.pi * self[plane]["QF"])) % np.pi
            - np.pi * self[plane]["Q"]
        )
        psi = psi / (2 * np.pi)
        return bpmac1, psi, k, exciter
