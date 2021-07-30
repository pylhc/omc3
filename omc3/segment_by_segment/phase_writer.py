import tfs
import pandas as pd
import numpy as np
from pathlib import Path


def _create_dict_segment(omc3_format, plane):
    PLANE = plane.upper()
    plane = plane.lower()

    NAME_DICT = dict(
        MU=f"MU{PLANE}",
        PHASE=f"PHASE{PLANE}",
        STDPH=f"ERRDELTAPHASE{PLANE}",
        BET=f"BET{PLANE}",
        ERRBET=f"ERRBET{PLANE}",
        ALF=f"ALF{PLANE}",
        ERRALF=f"ERRALF{PLANE}",
        STDERRPHASE=f"STDERRPHASE{PLANE}",
        MEASPHASE=f"MEASPHASE{PLANE}",
        PROPPHASE=f"PROPPHASE{PLANE}",
        CORPHASE=f"CORPHASE{PLANE}",
        BACKPHASE=f"BACKPHASE{PLANE}",
        BACKCORPHASE=f"BACKCORPHASE{PLANE}",
        ERRPROPPHASE=f"ERRPROPPHASE{PLANE}",
        ERRCORPHASE=f"ERRCORPHASE{PLANE}",
        ERRBACKPHASE=f"ERRBACKPHASE{PLANE}",
        ERRBACKCORPHASE=f"ERRBACKCORPHASE{PLANE}"
    )
    if (omc3_format):
        NAME_DICT['file_phase_measure'] = f"total_phase_{plane}.tfs"
        NAME_DICT['file_beta_measure'] = f"beta_phase_{plane}.tfs"
    else:
        NAME_DICT['file_phase_measure'] = f"getphasetot{plane}_free.out"
        NAME_DICT['file_beta_measure'] = f"getbeta{plane}_free.out"
        NAME_DICT['STDPH'] = f"PHASE{PLANE}"
    return NAME_DICT


def _propagate_error_phase(errb0, erra0, dphi, bet0, alf0):
    return np.sqrt((((1 / 2. * np.cos(4 * np.pi * dphi) * alf0 / bet0) - (1 / 2. * np.sin(4 * np.pi * dphi) / bet0)
                     - (1 / 2. * alf0 / bet0)) * errb0) ** 2 + (
                               (-(1 / 2. * np.cos(4 * np.pi * dphi)) + (1 / 2.)) * erra0) ** 2) / (2 * np.pi)


def create_phase_segment(folder_measure, output_dir, label):
    output_dir = Path(output_dir)
    planes = ['x', 'y']
    omc3_format = True

    for plane in planes:
        NAME_DICT = _create_dict_segment(omc3_format, plane)
        file_phase_measure = Path(NAME_DICT['file_phase_measure'])
        file_beta_measure = Path(NAME_DICT['file_beta_measure'])

        file_phase = tfs.read(folder_measure / file_phase_measure, index="NAME")
        file_beta = tfs.read(folder_measure / file_beta_measure, index="NAME")

        # In order not to have several S and get the correc S.
        file_phase = file_phase.rename(columns={"S": "MODEL_S"})
        file_phase = file_phase.rename(columns={NAME_DICT["PHASE"]: NAME_DICT["MEASPHASE"]})

        twiss_forward = tfs.read(output_dir / f"twiss_{label}.dat",
                                 index="NAME")  # model_propagation = propagated_models.propagation
        twiss_forward_corr = tfs.read(output_dir / f"twiss_{label}_cor.dat",
                                      index="NAME")  # model_back_propagation = propagated_models.back_propagation
        twiss_back = tfs.read(output_dir / f"twiss_{label}_back.dat",
                              index="NAME")  # model_cor = propagated_models.corrected
        twiss_back_corr = tfs.read(output_dir / f"twiss_{label}_cor_back.dat",
                                   index="NAME")  # model_back_cor = propagated_models.corrected_back_propagation

        # At the moment it is only for the BPM positions
        twiss_forward = pd.concat([file_phase, twiss_forward], axis=1, join="inner")
        twiss_forward_corr = pd.concat([file_phase, twiss_forward_corr], axis=1, join="inner")
        twiss_back = pd.concat([file_phase, twiss_back], axis=1, join="inner")
        twiss_back_corr = pd.concat([file_phase, twiss_back_corr], axis=1, join="inner")

        pd_out = twiss_forward.loc[:, ["S", "MODEL_S", NAME_DICT["STDPH"]]]
        pd_out = pd_out.rename(columns={NAME_DICT["STDPH"]: NAME_DICT["STDERRPHASE"]})

        tmp_pd = twiss_forward.loc[:, [NAME_DICT["MU"], NAME_DICT["MEASPHASE"]]]
        first_row = tmp_pd.iloc[[0]].to_numpy()[0]
        last_row = tmp_pd.iloc[[-1]].to_numpy()[0]

        twiss_forward_ph = tmp_pd.apply(lambda row: (row - first_row) % 1, axis=1)
        measure_back_ph = tmp_pd.apply(lambda row: (row - last_row) % 1, axis=1)

        pd_out[NAME_DICT["MEASPHASE"]] = twiss_forward_ph[NAME_DICT["MEASPHASE"]]
        pd_out[NAME_DICT["PROPPHASE"]] = twiss_forward_ph[NAME_DICT["MEASPHASE"]] - twiss_forward_ph[NAME_DICT["MU"]]

        tmp_pd = twiss_forward_corr.loc[:, [NAME_DICT["MU"]]]
        first_row = tmp_pd.iloc[[0]].to_numpy()[0]
        twiss_forward_corr_ph = tmp_pd.apply(lambda row: (row - first_row) % 1, axis=1)
        pd_out[NAME_DICT["CORPHASE"]] = (twiss_forward_corr_ph[NAME_DICT["MU"]] - twiss_forward_ph[NAME_DICT["MU"]])

        tmp_pd = twiss_back.loc[:, [NAME_DICT["MU"]]]
        first_row = tmp_pd.iloc[[-1]].to_numpy()[0]
        twiss_back_ph = tmp_pd.apply(lambda row: (first_row - row) % 1, axis=1)

        tmp_pd = twiss_back_corr.loc[:, [NAME_DICT["MU"]]]
        first_row = tmp_pd.iloc[[-1]].to_numpy()[0]
        twiss_back_corr_ph = tmp_pd.apply(lambda row: (first_row - row) % 1, axis=1)

        # First and last BPM
        start_bpm = twiss_forward.index[0]
        end_bpm = twiss_forward.index[-1]

        betx_start = file_beta[NAME_DICT["BET"]].loc[start_bpm]
        betx_end = file_beta[NAME_DICT["BET"]].loc[end_bpm]

        err_betx_start = file_beta[NAME_DICT["ERRBET"]].loc[start_bpm]
        err_betx_end = file_beta[NAME_DICT["ERRBET"]].loc[end_bpm]

        alfx_start = file_beta[NAME_DICT["ALF"]].loc[start_bpm]
        alfx_end = -file_beta[NAME_DICT["ALF"]].loc[end_bpm]

        err_alfx_start = file_beta[NAME_DICT["ERRALF"]].loc[start_bpm]
        err_alfx_end = file_beta[NAME_DICT["ERRALF"]].loc[end_bpm]

        pd_out[NAME_DICT["BACKPHASE"]] = ((measure_back_ph[NAME_DICT["MEASPHASE"]] -
                                           twiss_back_ph[NAME_DICT["MU"]]) % 1).apply(lambda x: x - 1 if x > 0.5 else x)
        pd_out[NAME_DICT["BACKCORPHASE"]] = twiss_back[NAME_DICT["MU"]] - twiss_back_corr[NAME_DICT["MU"]]

        pd_out[NAME_DICT["ERRPROPPHASE"]] = _propagate_error_phase(err_betx_start, err_alfx_start,
                                                                   twiss_forward_ph[NAME_DICT["MU"]], betx_start,
                                                                   alfx_start)
        pd_out[NAME_DICT["ERRPROPPHASE"]] = np.sqrt(
            pd_out[NAME_DICT["ERRPROPPHASE"]] ** 2 + pd_out[NAME_DICT["STDERRPHASE"]] ** 2)

        pd_out[NAME_DICT["ERRCORPHASE"]] = _propagate_error_phase(err_betx_start, err_alfx_start,
                                                                  twiss_forward_corr_ph[NAME_DICT["MU"]], betx_start,
                                                                  alfx_start)
        pd_out[NAME_DICT["ERRCORPHASE"]] = np.sqrt(
            pd_out[NAME_DICT["ERRCORPHASE"]] ** 2 + pd_out[NAME_DICT["ERRPROPPHASE"]] ** 2)

        pd_out[NAME_DICT["ERRBACKPHASE"]] = _propagate_error_phase(err_betx_end, err_alfx_end,
                                                                   twiss_back_ph[NAME_DICT["MU"]], betx_end, alfx_end)
        pd_out[NAME_DICT["ERRBACKPHASE"]] = np.sqrt(
            pd_out[NAME_DICT["ERRBACKPHASE"]] ** 2 + pd_out[NAME_DICT["STDERRPHASE"]] ** 2)

        pd_out[NAME_DICT["ERRBACKCORPHASE"]] = _propagate_error_phase(err_betx_end, err_alfx_end,
                                                                      twiss_back_corr_ph[NAME_DICT["MU"]], betx_end,
                                                                      alfx_end)
        pd_out[NAME_DICT["ERRBACKCORPHASE"]] = np.sqrt(
            pd_out[NAME_DICT["ERRBACKPHASE"]] ** 2 + pd_out[NAME_DICT["ERRBACKCORPHASE"]] ** 2)

        tfs.write(output_dir / f"sbsphase{plane}_{label}.out", pd_out, save_index="NAME")
