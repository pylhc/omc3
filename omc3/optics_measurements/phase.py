"""
.. module: phase

Created on 18/07/18

:author: Lukas Malina

It computes betatron phase advances and provides structures to store them.
"""
from os.path import join
import numpy as np
import pandas as pd
import tfs
from utils import logging_tools, stats
from optics_measurements.toolbox import df_ang_diff, df_diff, ang_sum
from optics_measurements.constants import EXT, PHASE_NAME, TOTAL_PHASE_NAME, ERR, DELTA, MDL
LOGGER = logging_tools.get_logger(__name__)


def calculate(meas_input, input_files, tunes, header_dict, plane):
    """
    Calculates phase advances and fills the following files:
        getphase(tot)(x/y)(_free).out

    Parameters:
        meas_input: the input object including settings and the accelerator class
        input_files: includes measurement tfs
        tunes: TunesDict object containing measured and model tunes
        header_dict: part of the header common for all output files
        plane: "X" or "Y"

    Returns:
        an instance of PhaseDict filled with the results of get_phases
    """
    if meas_input.compensation == "none" and meas_input.accelerator.excitation:
        model = meas_input.accelerator.get_driven_tfs()
    else:
        model = meas_input.accelerator.get_model_tfs()
    comp_dict = {"equation": dict(eq_comp=tunes),
                 "model": dict(model_comp=meas_input.accelerator.get_driven_tfs()),
                 "none": dict()}
    LOGGER.info("Calculating phase advances")
    LOGGER.info(f"Measured tune in plane {plane} = {tunes[plane]['Q']}")
    phase_d, output_dfs = get_phases(meas_input, input_files, model, plane, **comp_dict[meas_input.compensation])
    headers = _get_headers(header_dict, tunes, free=(meas_input.compensation != "none"))
    _write_output(headers, output_dfs, meas_input.outputdir, plane)
    _write_special_phase_file(plane, phase_d, tunes[plane]["QF"] if (meas_input.compensation != "none") else tunes[plane]["Q"],
                              meas_input.accelerator, meas_input.outputdir)

    return phase_d


def get_phases(meas_input, input_files, model, plane, eq_comp=None, model_comp=None, no_errors=True):
    """
    Computes phase advances among all BPMs.

    Args:
        meas_input: OpticsInput object
        input_files: InputFiles object
        model: model tfs_panda to be used
        plane: "X" or "Y"
        eq_comp: (driven_tune,free_tune,ac2bpm object)
        no_errors: if True measured errors shall not be propagated (only their spread)

    Returns:
        dictionary of DataFrames indexed (BPMi x BPMj) yielding phase advance phi_ij
            "MEAS" measured phase advances
            "ERRMEAS" errors of measured phase advances
            "MODEL" model phase advances

            +------++--------+--------+--------+--------+
            |      ||  BPM1  |  BPM2  |  BPM3  |  BPM4  |
            +======++========+========+========+========+
            | BPM1 ||   0    | phi_12 | phi_13 | phi_14 |
            +------++--------+--------+--------+--------+
            | BPM2 || phi_21 |    0   | phi_23 | phi_24 |
            +------++--------+--------+--------+--------+
            | BPM3 || phi_31 | phi_32 |   0    | phi_34 |
            +------++--------+--------+--------+--------+

            The phase advance between BPM_i and BPM_j can be obtained via:
                phase_advances["MEAS"].loc[BPMi,BPMj]
        list of output data frames(for files)
    """
    phase_frame = pd.DataFrame(model).loc[:, ["S", f"MU{plane}"]]
    how = 'outer' if meas_input.union else 'inner'
    phase_frame = pd.merge(phase_frame,
                           input_files.joined_frame(plane, [f"MU{plane}", f"{ERR}_MU{plane}"],
                                                    dpp_value=0, how=how),
                           how='inner', left_index=True, right_index=True)
    if model_comp is not None:
        phase_frame = pd.merge(phase_frame, pd.DataFrame(model_comp.loc[:, [f"MU{plane}"]].rename(
            columns={f"MU{plane}": f"MU{plane}comp"})),
                      how='inner', left_index=True, right_index=True)
        phase_compensation = df_diff(phase_frame, f"MU{plane}", f"MU{plane}comp")
        phase_frame[input_files.get_columns(phase_frame, f"MU{plane}")] = ang_sum(input_files.get_data(phase_frame, f"MU{plane}"), phase_compensation[:, np.newaxis])

    phases_mdl = phase_frame.loc[:, f"MU{plane}"].values
    phase_advances = {"MODEL": _get_square_data_frame(
            (phases_mdl[np.newaxis, :] - phases_mdl[:, np.newaxis]) % 1.0, phase_frame.index)}
    phases_meas = input_files.get_data(phase_frame, f"MU{plane}") * meas_input.accelerator.get_beam_direction()
    phases_errors = input_files.get_data(phase_frame, f"{ERR}_MU{plane}")


    if eq_comp is not None:
        driven_tune, free_tune, ac2bpmac = eq_comp[plane]["Q"], eq_comp[plane]["QF"], eq_comp[plane]["ac2bpm"]
        k_bpmac = ac2bpmac[2]
        phase_corr = ac2bpmac[1] - phases_meas[k_bpmac] + (0.5 * driven_tune)
        phases_meas = phases_meas + phase_corr[np.newaxis, :]
        r = eq_comp.get_lambda(plane)
        phases_meas[k_bpmac:, :] = phases_meas[k_bpmac:, :] - driven_tune
        psi = (np.arctan((1 - r) / (1 + r) * np.tan(2 * np.pi * phases_meas)) / (2 * np.pi)) % 0.5
        phases_meas = np.where(phases_meas % 1.0 > 0.5, psi + .5, psi)
        phases_meas[k_bpmac:, :] = phases_meas[k_bpmac:, :] + free_tune

    if phases_meas.ndim < 2:
        phase_advances["MEAS"] = _get_square_data_frame(
                (phases_meas[np.newaxis, :] - phases_meas[:, np.newaxis]) % 1.0, phase_frame.index)
        phase_advances["ERRMEAS"] = _get_square_data_frame(
                np.zeros((len(phases_meas), len(phases_meas))), phase_frame.index)
        return phase_advances
    if meas_input.union:
        mask = np.isnan(phases_meas)
        phases_meas[mask], phases_errors[mask] = 0.0, np.inf
        if no_errors:
            phases_errors[~mask] = 1e-10
    elif no_errors:
        phases_errors = None
    phases_3d = phases_meas[np.newaxis, :, :] - phases_meas[:, np.newaxis, :]
    if phases_errors is not None:
        errors_3d = phases_errors[np.newaxis, :, :] + phases_errors[:, np.newaxis, :]
    else:
        errors_3d = None
    phase_advances["MEAS"] = _get_square_data_frame(stats.circular_mean(
            phases_3d, period=1, errors=errors_3d, axis=2) % 1.0, phase_frame.index)
    phase_advances["ERRMEAS"] = _get_square_data_frame(stats.circular_error(
            phases_3d, period=1, errors=errors_3d, axis=2), phase_frame.index)
    return phase_advances, [_create_output_df(phase_advances, phase_frame, plane),
                            _create_output_df(phase_advances, phase_frame, plane, tot=True)]


def _write_output(headers, dfs, output, plane):
    for head, df, name in zip(headers, dfs, (PHASE_NAME, TOTAL_PHASE_NAME)):
        tfs.write(join(output, f"{name}{plane.lower()}{EXT}"), df, head)
        LOGGER.info(f"Phase advance beating in {name}{plane.lower()}{EXT} = {stats.weighted_rms(df.loc[:, f'{DELTA}PHASE{plane}'])}")


def _create_output_df(phase_advances, model, plane, tot=False):
    meas = phase_advances["MEAS"]
    mod = phase_advances["MODEL"]
    err = phase_advances["ERRMEAS"]
    if tot:
        output_data = model.loc[:, ["S", f"MU{plane}"]].iloc[:, :]
        output_data["NAME"] = output_data.index
        output_data = output_data.assign(S2=model.at[model.index[0], "S"], NAME2=model.index[0])
        output_data[f"PHASE{plane}"] = meas.values[0, :]
        output_data[f"{ERR}PHASE{plane}"] = err.values[0, :]
        output_data[f"PHASE{plane}{MDL}"] = mod.values[0, :]
    else:
        output_data = model.loc[:, ["S", f"MU{plane}"]].iloc[:-1, :]
        output_data["NAME"] = output_data.index
        output_data = output_data.assign(S2=model.loc[:, "S"].values[1:], NAME2=model.index[1:].values)
        output_data[f"PHASE{plane}"] = np.diag(meas.values, k=1)
        output_data[f"{ERR}PHASE{plane}"] = np.diag(err.values, k=1)
        output_data[f"PHASE{plane}{MDL}"] = np.diag(mod.values, k=1)
    output_data.rename(columns={f"MU{plane}": f"MU{plane}{MDL}"}, inplace=True)
    output_data[f"{DELTA}PHASE{plane}"] = df_ang_diff(output_data, f"PHASE{plane}", f"PHASE{plane}{MDL}")
    output_data[f"{ERR}{DELTA}PHASE{plane}"] = output_data.loc[:, f"{ERR}PHASE{plane}"].values
    return output_data


def _get_headers(header_dict, tunes, free=False):
    header = header_dict.copy()
    header['Q1'] = tunes["X"]["QF"] if free else tunes["X"]["Q"]
    header['Q2'] = tunes["Y"]["QF"] if free else tunes["Y"]["Q"]
    header_tot = header.copy()
    return [header, header_tot]


def _get_square_data_frame(data, index):
    return pd.DataFrame(data=data, index=index, columns=index)


def _write_special_phase_file(plane, phase_advances, plane_tune, accel, outputdir):
    # TODO REFACTOR AND SIMPLIFY
    plane_mu = "MU" + plane
    meas = phase_advances["MEAS"]
    bd = accel.get_beam_direction()
    elements = accel.get_elements_tfs()
    lines = []
    for elem1, elem2 in accel.get_important_phase_advances():
        mus1 = elements.loc[elem1, plane_mu] - elements.loc[:, plane_mu]
        minmu1 = abs(mus1.loc[meas.index]).idxmin()
        mus2 = elements.loc[:, plane_mu] - elements.loc[elem2, plane_mu]
        minmu2 = abs(mus2.loc[meas.index]).idxmin()
        bpm_phase_advance = meas.loc[minmu1, minmu2]
        model_value = elements.loc[elem2, plane_mu] - elements.loc[elem1, plane_mu]
        if (elements.loc[elem1, "S"] - elements.loc[elem2, "S"]) * bd > 0.0:
            bpm_phase_advance += plane_tune
            model_value += plane_tune
        bpm_err = phase_advances["ERRMEAS"].loc[minmu1, minmu2]
        phase_to_first = -mus1.loc[minmu1]
        phase_to_second = -mus2.loc[minmu2]
        ph_result = ((bpm_phase_advance + phase_to_first + phase_to_second) * bd)
        model_value = (model_value * bd)
        resultdeg = ph_result % .5 * 360
        if resultdeg > 90:
            resultdeg -= 180
        modeldeg = model_value % .5 * 360
        if modeldeg > 90:
            modeldeg -= 180
        model_desc = f"{elem1} to {elem2} MODEL: {model_value % 1:8.4f}     {'':6s} = {modeldeg:6.2f} deg"
        result_desc = f"{elem1} to {elem2} MEAS : {ph_result % 1:8.4f}  +- {bpm_err:6.4f} = "
        f"{resultdeg:6.2f} +- {bpm_err * 360:3.2f} deg ({bpm_phase_advance:8.4f} + {phase_to_first + phase_to_second:8.4f} [{minmu1}, {minmu2}])"
        lines.extend([model_desc, result_desc])
    with open(join(outputdir, f"special_phase_{plane.lower()}.txt"), 'w') as special_phase_writer:
        special_phase_writer.write('Special phase advances\n')
        for line in lines:
            special_phase_writer.write(line + '\n')
