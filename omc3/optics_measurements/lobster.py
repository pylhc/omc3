
from omc3.definitions.constants import PI2, PI
from omc3.optics_measurements.beta_from_phase import _tilt_slice_matrix
import numpy as np
from numpy import tan
import pandas as pd
import tfs

from omc3.optics_measurements.constants import EXT
from pathlib import Path

def _some_combs():
    return [
        (1,2,3),  # ijkl
        (2,3,4),  # i-jkl
        (1,3,4),  # ij-kl
        (1,2,4),  # ijk-l
        (3,4,5),  # i--jkl
        (1,4,5),  # ij--kl
    ]

def _combs_in_range(combrange: int):
    return [(j,k,l) 
            for l in range(3, combrange)
            for k in range(2, l)
            for j in range(1, k)
           ]



def calculate(meas_input, tunes, phase_dict, header_dict, plane):
    print("starting LObster")
    phase = phase_dict['free']

    tune, mdltune = (tunes[plane]["Q"], tunes[plane]["QM"] % 1) \
        if meas_input.compensation == "none" else (tunes[plane]["QF"], tunes[plane]["QFM"] % 1)

    tilted_meas = _tilt_slice_matrix(phase["MEAS"].to_numpy(), 0, 10, tune) * PI2
    tilted_model = _tilt_slice_matrix(phase["MODEL"].to_numpy(), 0, 10, mdltune) * PI2
    tilted_errmeas = _tilt_slice_matrix(phase["ERRMEAS"].to_numpy(), 0, 10, mdltune) * PI2

    # phase beating
    tilted_beating = tilted_meas - tilted_model
    cot_model = 1.0 / tan(tilted_model)

    for (j,k,l) in _combs_in_range(8):
        phi = []
        errphi = []
        names = []
        modelph = []
        # go line by line
        for line_index in range(tilted_model.shape[1]):
            model_line = tilted_model[:,line_index]
            beating_line = tilted_beating[:,line_index]
            err_line = tilted_errmeas[:,line_index]

            index_j = j
            index_k = k
            index_l = l

#            current_index = 1
#            while current_index < 9 and abs(model_line[current_index]) < (j*PI/4 - 0.1):
#                current_index += 1
#            index_j = current_index
#            while current_index < 9 and abs(model_line[current_index]) < (k*PI/4 - 0.1):
#                current_index += 1
#            index_k = current_index
#            while current_index < 9 and abs(model_line[current_index]) < (l*PI/4 - 0.1):
#                current_index += 1
#            index_l = current_index
#
#            if index_j >= index_k or index_k >= index_l:
#                continue

            rescaling = 1.0/tan(model_line[index_l] - model_line[index_j]) \
                +1.0/tan(model_line[index_k] - model_line[index_j]) \
                +1.0/tan(model_line[index_k]) \
                -1.0/tan(model_line[index_l])

            names.append(phase["MODEL"].index[line_index])

            phi_ijkl = 1.0/tan(model_line[index_l] - model_line[index_j]) * (beating_line[index_l] - beating_line[index_j]) \
                +1.0/tan(model_line[index_k] - model_line[index_j]) * (beating_line[index_j] - beating_line[index_k]) \
                +1.0/tan(model_line[index_k]) * beating_line[index_k] \
                -1.0/tan(model_line[index_l]) * beating_line[index_l]

            err_ijkl = np.sqrt(
                1.0/tan(model_line[index_l] - model_line[index_j])**2 * (err_line[index_l]**2 + err_line[index_j]**2)
                +1.0/tan(model_line[index_k] - model_line[index_j])**2 * (err_line[index_j]**2 + err_line[index_k]**2)
                +1.0/tan(model_line[index_k])**2 * err_line[index_k]**2
                +1.0/tan(model_line[index_l])**2 * err_line[index_l]**2
            )

            phi.append(
                phi_ijkl / rescaling
            )

            errphi.append(err_ijkl / rescaling)

            modelph.append(
                f"0-{int(360/PI2*model_line[index_j])}"
                f"-{int(360/PI2*model_line[index_k])}"
                f"-{int(360/PI2*model_line[index_l])}"
            )

        local_df = pd.DataFrame(index=names)
        local_df["NAME"] = names
        local_df["S"] = meas_input.accelerator.model.loc[names, "S"]
        local_df["LOCALOBS"] = phi
        local_df["ERR"] = errphi
        local_df["MODEL_PHASES"] = modelph

        tfs.write(Path(meas_input.outputdir) / f"lobster_{tuple_to_string((0,j,k,l))}{EXT}", local_df, header_dict)


def tuple_to_string(t):
    (i,j,k,l) = t
    pattern = ['-'] * (l+1)
    pattern[i] = 'I'
    pattern[j] = 'J'
    pattern[k] = 'K'
    pattern[l] = 'L'
    return "".join(pattern)
