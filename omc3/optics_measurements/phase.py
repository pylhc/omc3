"""
Phase advance
----------------

:module: optics_measurements.phase
:author: Lukas Malina

Computes betatron phase advances and provides structures to store them.
"""
from os.path import join
import numpy as np
import pandas as pd
import tfs
from utils import logging_tools, stats
from optics_measurements.toolbox import df_ang_diff, df_diff, ang_sum
from optics_measurements.constants import EXT, PHASE_NAME, TOTAL_PHASE_NAME, ERR, DELTA, MDL
LOGGER = logging_tools.get_logger(__name__)


def calculate(meas_input, input_files, tunes, plane, no_errors=False):
    """
    Calculates phase advances

    Parameters:
        meas_input: the input object including settings and the accelerator class
        input_files: includes measurement tfs
        tunes: TunesDict object containing measured and model tunes and ac2bpm object
        plane: "X" or "Y"
        no_errors: if True measured errors shall not be propagated (only their spread)

    Returns:
        dictionary of DataFrames indexed (BPMi x BPMj) yielding phase advance phi_ij

         - "MEAS" measured phase advances
         - "ERRMEAS" errors of measured phase advances
         - "MODEL" model phase advances


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
    LOGGER.info("Calculating phase advances")
    LOGGER.info(f"Measured tune in plane {plane} = {tunes[plane]['Q']}")

    df = pd.DataFrame(meas_input.accelerator.get_model_tfs()).loc[:, ["S", f"MU{plane}"]]
    how = 'outer' if meas_input.union else 'inner'
    dpp_value = meas_input.dpp if "dpp" in meas_input.keys() else 0
    df = pd.merge(df, input_files.joined_frame(plane, [f"MU{plane}", f"{ERR}MU{plane}"],
                                               dpp_value=dpp_value, how=how),
                  how='inner', left_index=True, right_index=True)
    phases_mdl = df.loc[:, f"MU{plane}"].to_numpy()
    phase_advances = {"MODEL": _get_square_data_frame(
        (phases_mdl[np.newaxis, :] - phases_mdl[:, np.newaxis]) % 1.0, df.index)}
    if meas_input.compensation == "model":
        df = _compensate_by_model(input_files, meas_input, df, plane)
    phases_meas = input_files.get_data(df, f"MU{plane}") * meas_input.accelerator.get_beam_direction()
    if meas_input.compensation == "equation":
        phases_meas = _compensate_by_equation(phases_meas, plane, tunes)

    phases_errors = input_files.get_data(df, f"{ERR}MU{plane}")
    if phases_meas.ndim < 2:
        phase_advances["MEAS"] = _get_square_data_frame(
                (phases_meas[np.newaxis, :] - phases_meas[:, np.newaxis]) % 1.0, df.index)
        phase_advances["ERRMEAS"] = _get_square_data_frame(
                np.zeros((len(phases_meas), len(phases_meas))), df.index)
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
            phases_3d, period=1, errors=errors_3d, axis=2) % 1.0, df.index)
    phase_advances["ERRMEAS"] = _get_square_data_frame(stats.circular_error(
            phases_3d, period=1, errors=errors_3d, axis=2), df.index)
    return phase_advances, [_create_output_df(phase_advances, df, plane),
                            _create_output_df(phase_advances, df, plane, tot=True)]


def _compensate_by_equation(phases_meas, plane, tunes):
    driven_tune, free_tune, ac2bpmac = tunes[plane]["Q"], tunes[plane]["QF"], tunes[plane]["ac2bpm"]
    k_bpmac = ac2bpmac[2]
    phase_corr = ac2bpmac[1] - phases_meas[k_bpmac] + (0.5 * driven_tune)
    phases_meas = phases_meas + phase_corr[np.newaxis, :]
    r = tunes.get_lambda(plane)
    phases_meas[k_bpmac:, :] = phases_meas[k_bpmac:, :] - driven_tune
    psi = (np.arctan((1 - r) / (1 + r) * np.tan(2 * np.pi * phases_meas)) / (2 * np.pi)) % 0.5
    phases_meas = np.where(phases_meas % 1.0 > 0.5, psi + .5, psi)
    phases_meas[k_bpmac:, :] = phases_meas[k_bpmac:, :] + free_tune
    return phases_meas


def _compensate_by_model(input_files, meas_input, df, plane):
    df = pd.merge(df, pd.DataFrame(meas_input.accelerator.get_driven_tfs().loc[:, [f"MU{plane}"]]),
                  how='inner', left_index=True, right_index=True, suffixes=("", "comp"))
    phase_compensation = df_diff(df, f"MU{plane}", f"MU{plane}comp")
    df[input_files.get_columns(df, f"MU{plane}")] = ang_sum(
        input_files.get_data(df, f"MU{plane}"), phase_compensation[:, np.newaxis])
    return df


def write(dfs, headers, output, plane):
    for head, df, name in zip(headers, dfs, (PHASE_NAME, TOTAL_PHASE_NAME)):
        tfs.write(join(output, f"{name}{plane.lower()}{EXT}"), df, head)
        LOGGER.info(f"Phase advance beating in {name}{plane.lower()}{EXT} = "
                    f"{stats.weighted_rms(df.loc[:, f'{DELTA}PHASE{plane}'])}")


def _create_output_df(phase_advances, model, plane, tot=False):
    meas = phase_advances["MEAS"]
    mod = phase_advances["MODEL"]
    err = phase_advances["ERRMEAS"]
    if tot:
        output_data = model.loc[:, ["S", f"MU{plane}"]].iloc[:, :]
        output_data["NAME"] = output_data.index
        output_data = output_data.assign(S2=model.at[model.index[0], "S"], NAME2=model.index[0])
        output_data[f"PHASE{plane}"] = meas.to_numpy()[0, :]
        output_data[f"{ERR}PHASE{plane}"] = err.to_numpy()[0, :]
        output_data[f"PHASE{plane}{MDL}"] = mod.to_numpy()[0, :]
    else:
        output_data = model.loc[:, ["S", f"MU{plane}"]].iloc[:-1, :]
        output_data["NAME"] = output_data.index
        output_data = output_data.assign(S2=model.loc[:, "S"].to_numpy()[1:], NAME2=model.index[1:].to_numpy())
        output_data[f"PHASE{plane}"] = np.diag(meas.to_numpy(), k=1)
        output_data[f"{ERR}PHASE{plane}"] = np.diag(err.to_numpy(), k=1)
        output_data[f"PHASE{plane}{MDL}"] = np.diag(mod.to_numpy(), k=1)
    output_data.rename(columns={f"MU{plane}": f"MU{plane}{MDL}"}, inplace=True)
    output_data[f"{DELTA}PHASE{plane}"] = df_ang_diff(output_data, f"PHASE{plane}", f"PHASE{plane}{MDL}")
    output_data[f"{ERR}{DELTA}PHASE{plane}"] = output_data.loc[:, f"{ERR}PHASE{plane}"].to_numpy()
    return output_data


def _get_square_data_frame(data, index):
    return pd.DataFrame(data=data, index=index, columns=index)


def write_special(meas_input, phase_advances, plane_tune, plane):
    # TODO REFACTOR AND SIMPLIFY
    accel = meas_input.accelerator
    meas = phase_advances["MEAS"]
    bd = accel.get_beam_direction()
    elements = accel.get_elements_tfs()
    lines = []
    for elem1, elem2 in accel.get_important_phase_advances():
        mus1 = elements.loc[elem1, f"MU{plane}"] - elements.loc[:, f"MU{plane}"]
        minmu1 = abs(mus1.loc[meas.index]).idxmin()
        mus2 = elements.loc[:, f"MU{plane}"] - elements.loc[elem2, f"MU{plane}"]
        minmu2 = abs(mus2.loc[meas.index]).idxmin()
        bpm_phase_advance = meas.loc[minmu1, minmu2]
        model_value = elements.loc[elem2, f"MU{plane}"] - elements.loc[elem1, f"MU{plane}"]
        if (elements.loc[elem1, "S"] - elements.loc[elem2, "S"]) * bd > 0.0:
            bpm_phase_advance += plane_tune
            model_value += plane_tune
        bpm_err = phase_advances["ERRMEAS"].loc[minmu1, minmu2]
        elems_to_bpms = -mus1.loc[minmu1] - mus2.loc[minmu2]
        ph_result = ((bpm_phase_advance + elems_to_bpms) * bd)
        model_value = (model_value * bd) % 1
        model_desc = f"{elem1} to {elem2} MODEL: {model_value:8.4f}     {'':6s} = {_to_deg(model_value):6.2f} deg"
        result_desc = (f"{elem1} to {elem2} MEAS : {ph_result % 1:8.4f}  +- {bpm_err:6.4f} = "
                       f"{_to_deg(ph_result):6.2f} +- {bpm_err * 360:3.2f} deg ({bpm_phase_advance:8.4f} + {elems_to_bpms:8.4f} [{minmu1}, {minmu2}])")
        lines.extend([model_desc, result_desc])
    with open(join(meas_input.outputdir, f"special_phase_{plane.lower()}.txt"), 'w') as spec_phase:
        spec_phase.write('Special phase advances\n')
        for line in lines:
            spec_phase.write(line + '\n')


def _to_deg(phase):  # -90 to 90 degrees
    phase = phase % 0.5 * 360
    if phase < 90:
        return phase
    return phase - 180
