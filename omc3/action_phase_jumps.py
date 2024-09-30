from __future__ import annotations

import re
import numpy as np
import pandas as pd
from omc3.utils import logging_tools
from omc3.optics_measurements.constants import BETA, PHASE_ADV
import turn_by_turn as tbt


LOG = logging_tools.get_logger(__name__)


AVERAGED = "AVERAGED"
DIFFMEAN = "DIFFMEAN"

class ActionPhaseData:
    
    def __init__(self):
        pass

    def use_bpm(self, name: str | pd.Index) -> bool | pd.Series:
        """ Function to check if the given elements should be used for action and phase calculations. """
        raise NotImplementedError


class LHCActionPhaseData(ActionPhaseData):
    
    def __init__(self):
        pass


    def use_bpm(self, name: str | pd.Index) -> bool | pd.Series:
        """ In the LHC only even-numbered BPMs are used."""
        pattern = r"B.+\.(\d+)[LR]\d\."

        if isinstance(name, str):
            match = re.match(pattern, name)
            if match is None:
                LOG.warning(f"Could not parse BPM name '{name}'")
                return False

            return int(match.group(1)) % 2 == 0
        
        return name.str.extract(pattern).astype(int) % 2 == 0




def get_action_phase_from_two_bpms(df_data: pd.DataFrame, bpm1: str, bpm2: str, plane: str) -> tuple[float, float]:
    """ Returns the action and phase from orbit, beta and phase of two BPMs, assuming no magnetic fields in between them.
    
    Implementation of Eq. 13 and Eq. 14 in [CardonaLinearNonlinearMagnetic2009]_ 
    with the missing square-roots over the betas as explained in [CardonaErratumLinearNonlinear2010]_ .

    Args:
        df_data (pd.DataFrame): Dataframe with BPMs and their beta, phase and orbit.
        bpm1 (str): First BPM name.
        bpm2 (str): Second BPM name.
        plane (str): 'X' or 'Y'.

    Returns:
        A tuple of floats with the action and phase.

    """
    z = df_data.loc[[bpm1, bpm2], f"{DIFFMEAN}{plane}"] / np.sqrt(df_data.loc[[bpm1, bpm2], f"{BETA}{plane}"])
    psi = df_data.loc[[bpm1, bpm2], f"{PHASE_ADV}{plane}"] * 2 * np.pi

    z1, z2 = z.loc[bpm1], z.loc[bpm2]  # actually z / sqrt(beta)
    psi1, psi2 = psi.loc[bpm1], psi.loc[bpm2]

    # Equation (13)
    action = (0.5*(z1**2 + z2**2) - z1 * z2 * np.cos(psi2 - psi1)) / (np.sin(psi2 - psi1)**2)

    # Equation (14)
    sin_diff = (z1 * np.sin(psi2) - z2 * np.sin(psi1))
    cos_diff = (z1 * np.cos(psi2) - z2 * np.cos(psi1)) 
    phase = np.arctan(sin_diff/cos_diff)

    # as implemented by J. Cardona - should be equivalent?
    # denom = np.sin(psi1) * np.cos(psi2) - np.cos(psi1) * np.sin(psi2)
    # sin_delta = (z1 * np.sin(psi2) - z2 * np.sin(psi1)) / denom
    # cos_delta = (z1 * np.cos(psi2) - z2 * np.cos(psi1)) / denom
    # phase = np.arctan2(sin_delta, cos_delta)
    return action, phase
    

def remove_average_orbit(tbt_data: tbt.TbtData) -> tbt.TbtData:
    """ Removes the average orbit per element, i.e. orbit offset, from the tbt-data.
    This is returned as a new TbtData object.

    Args:
        tbt_data (tbt.TbtData): TbtData to remove the average orbit from.

    Returns:
        tbt.TbtData: New TbtData instance with the average orbit removed.
     """
    new_matrices = [ 
        tbt.TransverseData(
            X=m.X - m.X.mean(axis=1), 
            Y=m.Y - m.Y.mean(axis=1)
        ) for m in tbt_data.matrices
    ] 
    return tbt.TbtData(
        matrices=new_matrices, 
        date=tbt_data.date, 
        bunch_ids=tbt_data.bunch_ids, 
        nturns=tbt_data.nturns
    )


def get_maximum_orbit_tbt(tbt_data: tbt.TbtData, bpm: str, tolerance: float = 2.0):
    pass



def action_phase_jumps():
    pass
    
    
    # build average and average max tbt data
    #