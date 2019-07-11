"""
Kick
----------------

:module: optics_measurements.kick
:author: Lukas Malina

Computes kick actions.
"""
from os.path import join
import pandas as pd
import numpy as np
import tfs
from optics_measurements.constants import ERR, RES, EXT, KICK_NAME, PLANE_TO_NUM


def calculate(measure_input, input_files, scale, header_dict, plane):
    """

    Args:
        measure_input: Optics_input object
        input_files: Stores the input files tfs
        scale: measured beta functions
        header_dict: OrderedDict containing information about the analysis
        plane: "X" or "Y"

    Returns:
        DataFrame containing actions and their errors
    """
    try:
        tunes_actions = _getkick(measure_input, input_files, plane)
    except IndexError:  # occurs if either no x or no y files exist
        return pd.DataFrame
    col_names = ["DPP", "DPPAMP", f"Q{plane}", f"{ERR}Q{plane}", f"NATQ{plane}", f"{ERR}NATQ{plane}",
                 f"sqrt2J{plane}", f"{ERR}sqrt2J{plane}", f"2J{plane}", f"{ERR}2J{plane}"]
    kick_frame = pd.DataFrame(data=tunes_actions, columns=col_names)
    kick_frame = _rescale_actions(kick_frame, scale, plane)
    header = _get_header(header_dict, scale)
    tfs.write(join(measure_input.outputdir, f"{KICK_NAME}{plane.lower()}{EXT}"), kick_frame, header)
    return kick_frame.loc[:, [f"sqrt2J{plane}", f"{ERR}sqrt2J{plane}"]].values


def _get_header(header_dict, beta_d):
    header = header_dict.copy()
    header["RescalingFactor"] = beta_d
    return header


def _rescale_actions(df, scaling_factor, plane):
    for col in (f"sqrt2J{plane}", f"{ERR}sqrt2J{plane}", f"2J{plane}", f"{ERR}2J{plane}"):
        df[f"{col}{RES}"] = df.loc[:, col].values * scaling_factor
    return df


def _getkick(measure_input, files, plane):
    out = np.zeros([len(files[plane]), 10])
    for i, df in enumerate(files[plane]):
        # what if the following is not there - except KeyError?
        out[i, 0] = df["DPP"]
        out[i, 1] = df["DPPAMP"]
        out[i, 2] = df[f"Q{PLANE_TO_NUM[plane]}"]
        out[i, 3] = df[f"Q{PLANE_TO_NUM[plane]}RMS"]
        if f"NATQ{PLANE_TO_NUM[plane]}" in df.headers.keys():
            out[i, 4] = df[f"NATQ{PLANE_TO_NUM[plane]}"]
            out[i, 5] = df[f"NATQ{PLANE_TO_NUM[plane]}RMS"]
        out[i, 6:] = _gen_kick_calc(measure_input, df, plane)
    return out


def _gen_kick_calc(meas_input, lin, plane):
    """
    Takes either PK2PK/2 for kicker excitation or AMP for AC-dipole excitation
    """
    frame = pd.merge(_get_model_arc_betas(meas_input, plane), lin.loc[:, [f"AMP{plane}", "PK2PK"]],
                     how='inner', left_index=True, right_index=True)
    amps = (frame.loc[:, f"AMP{plane}"].values if meas_input.accelerator.excitation
            else frame.loc[:, 'PK2PK'].values / 2.0)
    meansqrt2j = amps / np.sqrt(frame.loc[:, f"BET{plane}"].values)
    mean2j = np.square(amps) / frame.loc[:, f"BET{plane}"].values
    return np.array([np.mean(meansqrt2j), np.std(meansqrt2j), np.mean(mean2j), np.std(mean2j)])


def _get_model_arc_betas(measure_input, plane):
    accel = measure_input.accelerator
    model = (accel.get_driven_tfs() if accel.excitation else accel.get_model_tfs())
    return model.loc[:, ['S', f"BET{plane}"]].loc[
           accel.get_element_types_mask(model.index, ["arc_bpm"]), :]
