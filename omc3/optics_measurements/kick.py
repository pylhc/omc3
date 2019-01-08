"""
.. module: kick

Created on 29/06/18

:author: Lukas Malina

It computes kick actions.
"""
from os.path import join
import pandas as pd
import numpy as np
import tfs


def calculate_kick(measure_input, input_files, model, mad_ac, beta_d, header_dict):
    """
    Fills the following TfsFiles:
     - getkick.out getkickac.out

    Args:
        measure_input: Optics_input object
        input_files: Stores the input files tfs
        model:  Model tfs panda
        mad_ac:  Model tfs panda with AC-dipole in
        beta_d: measured beta functions
        header_dict: OrderedDict containing information about the analysis

    Returns:
    """
    try:
        tunes_actions = _getkick(input_files, _get_model_arc_betas(measure_input, model), ac=False)
    except IndexError:  # occurs if either no x or no y files exist
        return pd.DataFrame, pd.DataFrame
    column_names = ["DPP", "QX", "QXRMS", "QY", "QYRMS", "NATQX", "NATQXRMS", "NATQY", "NATQYRMS",
                    "sqrt2JX", "sqrt2JXSTD", "sqrt2JY", "sqrt2JYSTD", "2JX", "2JXSTD", "2JY",
                    "2JYSTD"]
    kick_frame = pd.DataFrame(data=tunes_actions, columns=column_names)
    header = _get_header(header_dict, beta_d)
    tfs.write(join(measure_input.outputdir, header['FILENAME']), kick_frame, header)
    actions_x, actions_y = tunes_actions[:, 9:11], tunes_actions[:, 11:13]  # sqrt2jx, sqrt2Jy

    if measure_input.accelerator.excitation:
        column_names_ac = column_names + ["sqrt2JXRES", "sqrt2JXSTDRES", "sqrt2JYRES", "sqrt2JYSTDRES", "2JXRES",
                            "2JXSTDRES", "2JYRES", "2JYSTDRES"]
        tunes_actions_ac = _getkick(input_files, _get_model_arc_betas(measure_input, mad_ac), ac=True)
        x, y = beta_d["X"], beta_d["Y"]
        tunes_actions_ac_res = tunes_actions_ac[:, 9:] / np.array([np.sqrt(x), np.sqrt(x), np.sqrt(y), np.sqrt(y), x, x, y, y])
        kick_frame_ac = pd.DataFrame(data=np.hstack((tunes_actions_ac, tunes_actions_ac_res)), columns=column_names_ac)
        header_ac = _get_header(header_dict, beta_d, ac=True)
        tfs.write(join(measure_input.outputdir, header_ac['FILENAME']), kick_frame_ac, header_ac)
        actions_x, actions_y = tunes_actions_ac[:, 9:11], tunes_actions_ac[:, 11:13]
    return actions_x, actions_y


def _get_model_arc_betas(measure_input, model):
    return model.loc[:, ['S', 'BETX', 'BETY']].loc[
           measure_input.accelerator.get_element_types_mask(model.index, ["arc_bpm"]), :]


def _get_header(header_dict, beta_d, ac=False):
    header = header_dict.copy()
    header['COMMENT'] = "Calculates the kick from the model beta function"
    header['FILENAME'] = 'getkick' + ac * 'ac' + '.out'
    if ac:
        header["RescalingFactor_for_X"] = beta_d["X"]
        header["RescalingFactor_for_Y"] = beta_d["Y"]
    return header


def _getkick(files, model_beta, ac):
    out = np.zeros([len(files["X"]), 17])
    for i in range(len(files["X"])):
        action_x_model = _gen_kick_calc(files["X"][i], model_beta, "X", ac)
        action_y_model = _gen_kick_calc(files["Y"][i], model_beta, "Y", ac)
        # what if the following is not there - except KeyError?
        out[i, 0] = files["X"][i].DPP
        out[i, 1] = files["X"][i].Q1
        out[i, 2] = files["X"][i].Q1RMS
        out[i, 3] = files["Y"][i].Q2
        out[i, 4] = files["Y"][i].Q2RMS
        out[i, 5] = files["X"][i].NATQ1
        out[i, 6] = files["X"][i].NATQ1RMS
        out[i, 7] = files["Y"][i].NATQ2
        out[i, 8] = files["Y"][i].NATQ2RMS
        out[i, 9:] = np.ravel(np.concatenate((action_x_model, action_y_model), axis=1))
    return out


def _gen_kick_calc(lin, beta, plane, ac):
    """
    Takes either PK2PK/2 for kicker excitation or 2 * AMP for AC-dipole excitation(complex spectra)
    """
    if ac:
        frame = pd.merge(beta, lin.loc[:, ['AMP' + plane]], how='inner', left_index=True,
                         right_index=True)
        amps = 2.0 * frame.loc[:, 'AMP' + plane].values  # multiplied by 2.0 due to complex spectra
    else:
        frame = pd.merge(beta, lin.loc[:, ['PK2PK']], how='inner', left_index=True, right_index=True)
        amps = frame.loc[:, 'PK2PK'].values / 2.0
    meansqrt2j = amps / np.sqrt(frame.loc[:, 'BET' + plane].values)
    mean2j = np.square(amps) / frame.loc[:, 'BET' + plane].values
    return np.array([[np.mean(meansqrt2j), np.std(meansqrt2j)], [np.mean(mean2j), np.std(mean2j)]])
