"""
Converts most important measurements produced by GetLLM into a more unified form
to allow straight forward comparison.
"""
from os.path import join, isfile
from datetime import datetime
from omc3.utils import logging_tools
from collections import OrderedDict

import tfs
from generic_parser.entrypoint_parser import entrypoint, EntryPointParameters, save_options_to_config

from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import EXT, ERR, DELTA, MDL
from omc3.optics_measurements.constants import AMP_BETA_NAME, BETA_NAME, PHASE_NAME, TOTAL_PHASE_NAME, DISPERSION_NAME, NORM_DISP_NAME, ORBIT_NAME
from omc3.optics_measurements.toolbox import df_ang_diff, df_diff, df_ratio, df_rel_diff, df_err_sum
from omc3.definitions import formats


LOGGER = logging_tools.get_logger(__name__)
OLD_EXT = ".out"
DEFAULT_CONFIG_FILENAME = "old_measurement_converter_{time:s}.ini"


def converter_params():
    params = EntryPointParameters()
    params.add_parameter(name="outputdir", required=True, help="Output directory.")
    params.add_parameter(name="suffix", type=str, default="_free2", choices=("", "_free", "_free2"),
                         help="Choose compensation suffix. ")
    return params


@entrypoint(converter_params(), strict=True)
def converter_entrypoint(opt):
    """
    Converts optics output files from GetLLM to new form.

    Converter Kwargs:
      - **outputdir**: Output directory.

        Flags: **--outputdir**
        Required: ``True``
      - **suffix** *(str)*: Choose compensation suffix.

        Flags: **--suffix**
        Default: ``_free2``

    """
    save_options_to_config(join(opt.outputdir, DEFAULT_CONFIG_FILENAME.format(
        time=datetime.utcnow().strftime(formats.TIME))), OrderedDict(sorted(opt.items())))
    convert_old_directory_to_new(opt)


def convert_old_directory_to_new(opt):
    # TODO check if new files are present?
    for plane in PLANES:
        convert_old_beta_from_amplitude(opt, plane)
        convert_old_beta_from_phase(opt, plane)
        convert_old_phase(opt, plane)
        convert_old_total_phase(opt, plane)
        # TODO phase vs phasetot inconsistent naming NAME S , first and second BPMs swapped locations
        convert_old_closed_orbit(opt, plane)
        convert_old_dispersion(opt, plane)
    convert_old_coupling(opt)
    convert_old_normalised_dispersion(opt, "X")



def convert_old_beta_from_amplitude(opt, plane, old_file_name="ampbeta", new_file_name=AMP_BETA_NAME):
    """
    getampbetax.out: *NAME S COUNT BETX BETXSTD BETXMDL MUXMDL BETXRES BETXSTDRES
    """
    old_file=join(opt.outputdir, f"get{old_file_name}{plane.lower()}{opt.suffix}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    df.rename(columns={f"BET{plane}STD": f"{ERR}BET{plane}",
                       f"BET{plane}STDRES": f"{ERR}BET{plane}RES"},
              inplace=True)
    df[f"{DELTA}BET{plane}"] = df_rel_diff(df, f"BET{plane}", f"BET{plane}{MDL}")
    df[f"{ERR}{DELTA}BET{plane}"] = df_ratio(df, f"{ERR}BET{plane}", f"BET{plane}{MDL}")
    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_beta_from_phase(opt, plane, old_file_name="beta", new_file_name=BETA_NAME):
    """
    getbetax.out: *NAME S COUNT BETX SYSBETX STATBETX ERRBETX CORR_ALFABETA ALFX SYSALFX STATALFX ERRALFX BETXMDL ALFXMDL MUXMDL NCOMBINATIONS
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{plane.lower()}{opt.suffix}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    if "CORR_ALFABETA" in df.columns.to_numpy():
        df.drop(columns=[f"STATBET{plane}", f"SYSBET{plane}", "CORR_ALFABETA",
                         f"STATALF{plane}", f"SYSALF{plane}"], inplace=True)
    else:
        df[f"{ERR}BET{plane}"] = df_err_sum(df, f"{ERR}BET{plane}", f"STDBET{plane}")
        df[f"{ERR}ALF{plane}"] = df_err_sum(df, f"{ERR}ALF{plane}", f"STDALF{plane}")

    df[f"{DELTA}BET{plane}"] = df_rel_diff(df, f"BET{plane}", f"BET{plane}{MDL}")
    df[f"{ERR}{DELTA}BET{plane}"] = df_ratio(df, f"{ERR}BET{plane}", f"BET{plane}{MDL}")
    df[f"{DELTA}ALF{plane}"] = df_diff(df, f"ALF{plane}", f"ALF{plane}{MDL}")
    df[f"{ERR}{DELTA}ALF{plane}"] = df.loc[:, f"{ERR}ALF{plane}"].values
    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_phase(opt, plane, old_file_name="phase", new_file_name=PHASE_NAME):
    """
    getphasex.out: *NAME NAME2 S S1 COUNT PHASEX STDPHX PHXMDL MUXMDL
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{plane.lower()}{opt.suffix}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    df.rename(columns={f"STDPH{plane}": f"{ERR}PHASE{plane}",
                       f"PH{plane}{MDL}": f"PHASE{plane}{MDL}", "S1": "S2"},
              inplace=True)
    df[f"{DELTA}PHASE{plane}"] = df_ang_diff(df, f"PHASE{plane}", f"PHASE{plane}{MDL}")
    df[f"{ERR}{DELTA}PHASE{plane}"] = df.loc[:, f"{ERR}PHASE{plane}"].values
    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_total_phase(opt, plane, old_file_name="phasetot", new_file_name=TOTAL_PHASE_NAME):
    """
    getphasex.out: *NAME NAME2 S S1 COUNT PHASEX STDPHX PHXMDL MUXMDL
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{plane.lower()}{opt.suffix}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    df.rename(columns={f"STDPH{plane}": f"{ERR}PHASE{plane}",
                       f"PH{plane}{MDL}": f"PHASE{plane}{MDL}", "S1": "S2"},
              inplace=True)
    df[f"{DELTA}PHASE{plane}"] = df_ang_diff(df, f"PHASE{plane}", f"PHASE{plane}{MDL}")
    df[f"{ERR}{DELTA}PHASE{plane}"] = df.loc[:, f"{ERR}PHASE{plane}"].values
    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_closed_orbit(opt, plane, old_file_name="CO", new_file_name=ORBIT_NAME):
    """
    getCOx.out: *NAME S COUNT X STDX XMDL MUXMDL
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{plane.lower()}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    df.rename(columns={f"STD{plane}": f"{ERR}{plane}"},
              inplace=True)
    df[f"{DELTA}{plane}"] = df_diff(df, f"{plane}", f"{plane}{MDL}")
    df[f"{ERR}{DELTA}{plane}"] = df.loc[:, f"{ERR}{plane}"].values
    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_dispersion(opt, plane, old_file_name="D", new_file_name=DISPERSION_NAME):
    """
    getDx.out: *NAME S COUNT DX STDDX DPX DXMDL DPXMDL MUXMDL
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{plane.lower()}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    df.rename(columns={f"STDD{plane}": f"{ERR}D{plane}"},
              inplace=True)
    df[f"{DELTA}D{plane}"] = df_diff(df, f"D{plane}", f"D{plane}{MDL}")
    df[f"{ERR}{DELTA}D{plane}"] = df.loc[:, f"{ERR}D{plane}"].values
    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_normalised_dispersion(opt, plane, old_file_name="ND", new_file_name=NORM_DISP_NAME):
    """
    getNDx.out: *NAME S COUNT NDX STDNDX DX DPX NDXMDL DXMDL DPXMDL MUXMDL
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{plane.lower()}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    df.rename(columns={f"STDND{plane}": f"{ERR}ND{plane}"}, inplace=True)
    df[f"{DELTA}ND{plane}"] = df_diff(df, f"ND{plane}", f"ND{plane}{MDL}")
    df[f"{ERR}{DELTA}ND{plane}"] = df.loc[:, f"{ERR}ND{plane}"].values
    if f"D{plane}" in df.columns:
        df.rename(columns={f"STDD{plane}": f"{ERR}D{plane}"}, inplace=True)
        df[f"{DELTA}D{plane}"] = df_diff(df, f"D{plane}", f"D{plane}{MDL}")

    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_coupling(opt, old_file_name="couple"):
    """
    getcouple.out: * NAME S COUNT F1001W FWSTD1 F1001R F1001I F1010W FWSTD2 F1010R F1010I Q1001 Q1001STD Q1010 Q1010STD MDLF1001R MDLF1001I MDLF1010R MDLF1010I
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{opt.suffix}{OLD_EXT}")
    new_file_name = "coupling_f"
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    dfs = {"1001": df.loc[:,["S", "COUNT", "F1001W", "FWSTD1", "F1001R", "F1001I", "Q1001",
                             "Q1001STD", "MDLF1001R", "MDLF1001I"]],
           "1010": df.loc[:, ["S", "COUNT", "F1010W", "FWSTD2", "F1010R", "F1010I", "Q1010",
                              "Q1010STD", "MDLF1010R", "MDLF1010I"]]}

    for i, rdt in enumerate(("1001", "1010")):
        dfs[rdt].drop(columns=[f"MDLF{rdt}R", f"MDLF{rdt}I"], inplace=True)
        dfs[rdt].rename(columns={f"F{rdt}W": "AMP", f"FWSTD{i+1}": f"{ERR}AMP", f"Q{rdt}": "PHASE",
                                 f"Q{rdt}STD": f"{ERR}PHASE", f"F{rdt}R": "REAL", f"F{rdt}I": "IMAG"}, inplace=True)
        tfs.write(join(opt.outputdir, f"{new_file_name}{rdt}{EXT}"), dfs[rdt])


if __name__ == '__main__':
    converter_entrypoint()
