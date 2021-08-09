"""
Converts most important measurements produced by GetLLM into a more unified form
to allow straight forward comparison.
"""
from collections import OrderedDict
from datetime import datetime
from os.path import isfile, join
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import tfs
from generic_parser.entrypoint_parser import (
    EntryPointParameters,
    entrypoint,
    save_options_to_config,
)

from omc3.definitions import formats
from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import (AMP_BETA_NAME, BETA_NAME, DELTA, DISPERSION_NAME, ERR,
                                                EXT, MDL, NORM_DISP_NAME, ORBIT_NAME, PHASE_NAME,
                                                TOTAL_PHASE_NAME)
from omc3.optics_measurements.toolbox import df_ang_diff, df_diff, df_err_sum, df_ratio, df_rel_diff
from omc3.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)
OLD_EXT = ".out"
DEFAULT_CONFIG_FILENAME = "old_measurement_converter_{time:s}.ini"


def converter_params():
    params = EntryPointParameters()
    params.add_parameter(name="outputdir", required=True, help="Output directory.")
    params.add_parameter(
        name="suffix",
        type=str,
        default="_free2",
        choices=("", "_free", "_free2"),
        help="Choose compensation suffix. ",
    )
    return params


@entrypoint(converter_params(), strict=True)
def converter_entrypoint(opt: EntryPointParameters) -> None:
    """
    TODO: write here
    Converts optics output files from GetLLM to new form.

    Converter Kwargs:
      - **outputdir**: Output directory.

        Flags: **--outputdir**
        Required: ``True``
      - **suffix** *(str)*: Choose compensation suffix.

        Flags: **--suffix**
        Default: ``_free2``

    """
    save_options_to_config(
        join(
            opt.outputdir,
            DEFAULT_CONFIG_FILENAME.format(time=datetime.utcnow().strftime(formats.TIME)),
        ),
        OrderedDict(sorted(opt.items())),
    )
    convert_old_directory_to_new(opt)


def convert_old_directory_to_new(opt: EntryPointParameters) -> None:
    """
    Looks ni the provided directory for expected ``BetaBeat.src`` output files, converts it to the output
    format used by ``omc3`` and  write them to the new location.

    Args:
        opt (EntryPointParameters): The entrypoint parameters parsed from the command line.
    """
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


def convert_old_beta_from_amplitude(
    opt: EntryPointParameters,
    plane: str,
    old_file_name: str = "ampbeta",
    new_file_name: str = AMP_BETA_NAME,
) -> None:
    """
    Looks in the provided directory for expected beta from amplitude file from ``BetaBeat.src`` for a given
    plane, converts it to the output format used by ``omc3`` and  write them to the new location.

    The file naming should be getampbeta(x,y).out, with the following expected columns: NAME, S, COUNT,
    BETX, BETXSTD, BETXMDL, MUXMDL, BETXRES, BETXSTDRES.

    Args:
        opt (EntryPointParameters): The entrypoint parameters parsed from the command line.
        plane (str): the transverse plane for which to look for the output file.
        old_file_name (str): the standard naming for the old output file.
        new_file_name (str): the standard naming for the new converted file.
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{plane.lower()}{opt.suffix}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    df.rename(
        columns={f"BET{plane}STD": f"{ERR}BET{plane}", f"BET{plane}STDRES": f"{ERR}BET{plane}RES"},
        inplace=True,
    )
    df[f"{DELTA}BET{plane}"] = df_rel_diff(df, f"BET{plane}", f"BET{plane}{MDL}")
    df[f"{ERR}{DELTA}BET{plane}"] = df_ratio(df, f"{ERR}BET{plane}", f"BET{plane}{MDL}")
    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_beta_from_phase(
    opt: EntryPointParameters,
    plane: str,
    old_file_name: str = "beta",
    new_file_name: str = BETA_NAME,
) -> None:
    """
    Looks in the provided directory for expected beta from phase file from ``BetaBeat.src`` for a given
    plane, converts it to the output format used by ``omc3`` and  write them to the new location.

    The file naming should be getbeta(x,y).out, with the following expected columns: NAME, S, COUNT,
     BETX, SYSBETX, STATBETX, ERRBETX, CORR_ALFABETA, ALFX, SYSALFX, STATALFX, ERRALFX, BETXMDL, ALFXMDL,
     MUXMDL, NCOMBINATIONS.

    Args:
        opt (EntryPointParameters): The entrypoint parameters parsed from the command line.
        plane (str): the transverse plane for which to look for the output file.
        old_file_name (str): the standard naming for the old output file.
        new_file_name (str): the standard naming for the new converted file.
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{plane.lower()}{opt.suffix}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    if "CORR_ALFABETA" in df.columns.to_numpy():
        df.drop(
            columns=[
                f"STATBET{plane}",
                f"SYSBET{plane}",
                "CORR_ALFABETA",
                f"STATALF{plane}",
                f"SYSALF{plane}",
            ],
            inplace=True,
        )
    else:
        df[f"{ERR}BET{plane}"] = df_err_sum(df, f"{ERR}BET{plane}", f"STDBET{plane}")
        df[f"{ERR}ALF{plane}"] = df_err_sum(df, f"{ERR}ALF{plane}", f"STDALF{plane}")

    df[f"{DELTA}BET{plane}"] = df_rel_diff(df, f"BET{plane}", f"BET{plane}{MDL}")
    df[f"{ERR}{DELTA}BET{plane}"] = df_ratio(df, f"{ERR}BET{plane}", f"BET{plane}{MDL}")
    df[f"{DELTA}ALF{plane}"] = df_diff(df, f"ALF{plane}", f"ALF{plane}{MDL}")
    df[f"{ERR}{DELTA}ALF{plane}"] = df.loc[:, f"{ERR}ALF{plane}"].values
    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_phase(
    opt: EntryPointParameters,
    plane: str,
    old_file_name: str = "phase",
    new_file_name: str = PHASE_NAME,
) -> None:
    """
    Looks in the provided directory for expected phase file from ``BetaBeat.src`` for a given
    plane, converts it to the output format used by ``omc3`` and  write them to the new location.

    The file naming should be getphase(x,y).out, with the following expected columns: NAME, NAME2, S, S1,
    COUNT, PHASEX, STDPHX, PHXMDL, MUXMDL.

    Args:
        opt (EntryPointParameters): The entrypoint parameters parsed from the command line.
        plane (str): the transverse plane for which to look for the output file.
        old_file_name (str): the standard naming for the old output file.
        new_file_name (str): the standard naming for the new converted file.
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{plane.lower()}{opt.suffix}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    df.rename(
        columns={
            f"STDPH{plane}": f"{ERR}PHASE{plane}",
            f"PH{plane}{MDL}": f"PHASE{plane}{MDL}",
            "S1": "S2",
        },
        inplace=True,
    )
    df[f"{DELTA}PHASE{plane}"] = df_ang_diff(df, f"PHASE{plane}", f"PHASE{plane}{MDL}")
    df[f"{ERR}{DELTA}PHASE{plane}"] = df.loc[:, f"{ERR}PHASE{plane}"].values
    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_total_phase(
    opt: EntryPointParameters,
    plane: str,
    old_file_name: str = "phasetot",
    new_file_name: str = TOTAL_PHASE_NAME,
) -> None:
    """
    Looks in the provided directory for expected total phase file from ``BetaBeat.src`` for a given
    plane, converts it to the output format used by ``omc3`` and  write them to the new location.

    The file naming should be getphasetot(x,y).out, with the following expected columns: NAME, NAME2, S,
    S1, COUNT, PHASEX, STDPHX, PHXMDL, MUXMDL.

    Args:
        opt (EntryPointParameters): The entrypoint parameters parsed from the command line.
        plane (str): the transverse plane for which to look for the output file.
        old_file_name (str): the standard naming for the old output file.
        new_file_name (str): the standard naming for the new converted file.
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{plane.lower()}{opt.suffix}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    df.rename(
        columns={
            f"STDPH{plane}": f"{ERR}PHASE{plane}",
            f"PH{plane}{MDL}": f"PHASE{plane}{MDL}",
            "S1": "S2",
        },
        inplace=True,
    )
    df[f"{DELTA}PHASE{plane}"] = df_ang_diff(df, f"PHASE{plane}", f"PHASE{plane}{MDL}")
    df[f"{ERR}{DELTA}PHASE{plane}"] = df.loc[:, f"{ERR}PHASE{plane}"].values
    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_closed_orbit(
    opt: EntryPointParameters,
    plane: str,
    old_file_name: str = "CO",
    new_file_name: str = ORBIT_NAME,
) -> None:
    """
    Looks in the provided directory for expected closed orbit file from ``BetaBeat.src`` for a given
    plane, converts it to the output format used by ``omc3`` and  write them to the new location.

    The file naming should be getCO(x,y).out, with the following expected columns: NAME, S, COUNT,
    X, STDX, XMDL, MUXMDL.

    Args:
        opt (EntryPointParameters): The entrypoint parameters parsed from the command line.
        plane (str): the transverse plane for which to look for the output file.
        old_file_name (str): the standard naming for the old output file.
        new_file_name (str): the standard naming for the new converted file.
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{plane.lower()}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    df.rename(columns={f"STD{plane}": f"{ERR}{plane}"}, inplace=True)
    df[f"{DELTA}{plane}"] = df_diff(df, f"{plane}", f"{plane}{MDL}")
    df[f"{ERR}{DELTA}{plane}"] = df.loc[:, f"{ERR}{plane}"].values
    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_dispersion(
    opt: EntryPointParameters,
    plane: str,
    old_file_name: str = "D",
    new_file_name: str = DISPERSION_NAME,
) -> None:
    """
    Looks in the provided directory for expected dispersion file from ``BetaBeat.src`` for a given
    plane, converts it to the output format used by ``omc3`` and  write them to the new location.

    The file naming should be getD(x,y).out, with the following expected columns: NAME, S, COUNT, DX,
    STDDX, DPX, DXMDL, DPXMDL, MUXMDL.

    Args:
        opt (EntryPointParameters): The entrypoint parameters parsed from the command line.
        plane (str): the transverse plane for which to look for the output file.
        old_file_name (str): the standard naming for the old output file.
        new_file_name (str): the standard naming for the new converted file.
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{plane.lower()}{OLD_EXT}")
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    df.rename(columns={f"STDD{plane}": f"{ERR}D{plane}"}, inplace=True)
    df[f"{DELTA}D{plane}"] = df_diff(df, f"D{plane}", f"D{plane}{MDL}")
    df[f"{ERR}{DELTA}D{plane}"] = df.loc[:, f"{ERR}D{plane}"].values
    tfs.write(join(opt.outputdir, f"{new_file_name}{plane.lower()}{EXT}"), df)


def convert_old_normalised_dispersion(
    opt: EntryPointParameters,
    plane: str,
    old_file_name: str = "ND",
    new_file_name: str = NORM_DISP_NAME,
) -> None:
    """
    Looks in the provided directory for expected normalized dispersion file from ``BetaBeat.src`` for a
    given plane, converts it to the output format used by ``omc3`` and  write them to the new location.

    The file naming should be getND(x,y).out, with the following expected columns: NAME, S, COUNT, NDX,
    STDNDX, DX, DPX, NDXMDL, DXMDL, DPXMDL, MUXMDL.

    Args:
        opt (EntryPointParameters): The entrypoint parameters parsed from the command line.
        plane (str): the transverse plane for which to look for the output file.
        old_file_name (str): the standard naming for the old output file.
        new_file_name (str): the standard naming for the new converted file.
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


def convert_old_coupling(opt: EntryPointParameters, old_file_name: str = "couple") -> None:
    """
    Looks in the provided directory for expected coupling file from ``BetaBeat.src``, converts it to the
    output format used by ``omc3`` and  write them to the new location.

    The file naming should be getcouple(x,y).out, with the following expected columns: NAME, S, COUNT,
    F1001W, FWSTD1, F1001R, F1001I, F1010W, FWSTD2, F1010R, F1010I, Q1001, Q1001STD, Q1010, Q1010STD,
    MDLF1001R, MDLF1001I, MDLF1010R, MDLF1010I.

    Args:
        opt (EntryPointParameters): The entrypoint parameters parsed from the command line.
        old_file_name (str): the standard naming for the old output file.
    """
    old_file = join(opt.outputdir, f"get{old_file_name}{opt.suffix}{OLD_EXT}")
    new_file_name = "coupling_f"
    if not isfile(old_file):
        return
    df = tfs.read(old_file)
    dfs = {
        "1001": df.loc[
            :,
            [
                "S",
                "COUNT",
                "F1001W",
                "FWSTD1",
                "F1001R",
                "F1001I",
                "Q1001",
                "Q1001STD",
                "MDLF1001R",
                "MDLF1001I",
            ],
        ],
        "1010": df.loc[
            :,
            [
                "S",
                "COUNT",
                "F1010W",
                "FWSTD2",
                "F1010R",
                "F1010I",
                "Q1010",
                "Q1010STD",
                "MDLF1010R",
                "MDLF1010I",
            ],
        ],
    }

    for i, rdt in enumerate(("1001", "1010")):
        dfs[rdt].drop(columns=[f"MDLF{rdt}R", f"MDLF{rdt}I"], inplace=True)
        dfs[rdt].rename(
            columns={
                f"F{rdt}W": "AMP",
                f"FWSTD{i+1}": f"{ERR}AMP",
                f"Q{rdt}": "PHASE",
                f"Q{rdt}STD": f"{ERR}PHASE",
                f"F{rdt}R": "REAL",
                f"F{rdt}I": "IMAG",
            },
            inplace=True,
        )
        tfs.write(join(opt.outputdir, f"{new_file_name}{rdt}{EXT}"), dfs[rdt])


if __name__ == "__main__":
    converter_entrypoint()
