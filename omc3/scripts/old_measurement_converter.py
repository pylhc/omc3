"""
Converts most important measurements produced by GetLLM into a more unified form
to allow straight forward comparison.
"""
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

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
from omc3.utils import contexts, iotools, logging_tools

LOGGER = logging_tools.get_logger(__name__)
OLD_EXT: str = ".out"
DEFAULT_CONFIG_FILENAME: str = "old_measurement_converter_{time:s}.ini"


def converter_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="inputdir",
        required=True,
        type=str,
        help="Directory with BetaBeat.src output files."
    )
    params.add_parameter(
        name="outputdir", required=True, type=str, help="Output directory for converted files."
    )
    params.add_parameter(
        name="suffix",
        type=str,
        default="_free2",
        choices=("", "_free", "_free2"),
        help="Compensation suffix used in the provided BetaBeat.src output. Defaults to '_free2'.",
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
        Path(opt.outputdir) / DEFAULT_CONFIG_FILENAME.format(time=datetime.utcnow().strftime(formats.TIME)),
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
    iotools.create_dirs(str(Path(opt.outputdir).absolute()))

    with contexts.timeit(lambda spanned: LOGGER.info(f"Total time for conversion: {spanned}s")):
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
    old_file_path = Path(opt.inputdir) / f"get{old_file_name}{plane.lower()}{opt.suffix}{OLD_EXT}"
    if not old_file_path.is_file():
        LOGGER.debug(f"Expected BetaBeat.src output at '{old_file_path.absolute()}' is not a file, skipping")

    dframe = tfs.read(old_file_path)
    dframe = dframe.rename(
        columns={f"BET{plane}STD": f"{ERR}BET{plane}", f"BET{plane}STDRES": f"{ERR}BET{plane}RES"},
    )
    dframe[f"{DELTA}BET{plane}"] = df_rel_diff(dframe, f"BET{plane}", f"BET{plane}{MDL}")
    dframe[f"{ERR}{DELTA}BET{plane}"] = df_ratio(dframe, f"{ERR}BET{plane}", f"BET{plane}{MDL}")
    tfs.write(Path(opt.outputdir) / f"{new_file_name}{plane.lower()}{EXT}", dframe)


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
    old_file_path = Path(opt.inputdir) / f"get{old_file_name}{plane.lower()}{opt.suffix}{OLD_EXT}"
    if not old_file_path.is_file():
        LOGGER.debug(f"Expected BetaBeat.src output at '{old_file_path.absolute()}' is not a file, skipping")

    dframe = tfs.read(old_file_path)
    if "CORR_ALFABETA" in dframe.columns.to_numpy():
        dframe = dframe.drop(
            columns=[f"STATBET{plane}", f"SYSBET{plane}", "CORR_ALFABETA", f"STATALF{plane}", f"SYSALF{plane}"]
        )
    else:
        dframe[f"{ERR}BET{plane}"] = df_err_sum(dframe, f"{ERR}BET{plane}", f"STDBET{plane}")
        dframe[f"{ERR}ALF{plane}"] = df_err_sum(dframe, f"{ERR}ALF{plane}", f"STDALF{plane}")

    dframe[f"{DELTA}BET{plane}"] = df_rel_diff(dframe, f"BET{plane}", f"BET{plane}{MDL}")
    dframe[f"{ERR}{DELTA}BET{plane}"] = df_ratio(dframe, f"{ERR}BET{plane}", f"BET{plane}{MDL}")
    dframe[f"{DELTA}ALF{plane}"] = df_diff(dframe, f"ALF{plane}", f"ALF{plane}{MDL}")
    dframe[f"{ERR}{DELTA}ALF{plane}"] = dframe.loc[:, f"{ERR}ALF{plane}"].to_numpy()
    tfs.write(Path(opt.outputdir) / f"{new_file_name}{plane.lower()}{EXT}", dframe)


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
    old_file_path = Path(opt.inputdir) / f"get{old_file_name}{plane.lower()}{opt.suffix}{OLD_EXT}"
    if not old_file_path.is_file():
        LOGGER.debug(f"Expected BetaBeat.src output at '{old_file_path.absolute()}' is not a file, skipping")

    dframe = tfs.read(old_file_path)
    dframe = dframe.rename(
        columns={
            f"STDPH{plane}": f"{ERR}PHASE{plane}",
            f"PH{plane}{MDL}": f"PHASE{plane}{MDL}",
            "S1": "S2",
        },
    )
    dframe[f"{DELTA}PHASE{plane}"] = df_ang_diff(dframe, f"PHASE{plane}", f"PHASE{plane}{MDL}")
    dframe[f"{ERR}{DELTA}PHASE{plane}"] = dframe.loc[:, f"{ERR}PHASE{plane}"].to_numpy()
    tfs.write(Path(opt.outputdir) / f"{new_file_name}{plane.lower()}{EXT}", dframe)


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
    old_file_path = Path(opt.inputdir) / f"get{old_file_name}{plane.lower()}{opt.suffix}{OLD_EXT}"
    if not old_file_path.is_file():
        LOGGER.debug(f"Expected BetaBeat.src output at '{old_file_path.absolute()}' is not a file, skipping")

    dframe = tfs.read(old_file_path)
    dframe = dframe.rename(
        columns={
            f"STDPH{plane}": f"{ERR}PHASE{plane}",
            f"PH{plane}{MDL}": f"PHASE{plane}{MDL}",
            "S1": "S2",
        },
    )
    dframe[f"{DELTA}PHASE{plane}"] = df_ang_diff(dframe, f"PHASE{plane}", f"PHASE{plane}{MDL}")
    dframe[f"{ERR}{DELTA}PHASE{plane}"] = dframe.loc[:, f"{ERR}PHASE{plane}"].to_numpy()
    tfs.write(Path(opt.outputdir) / f"{new_file_name}{plane.lower()}{EXT}", dframe)


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
    old_file_path = Path(opt.inputdir) / f"get{old_file_name}{plane.lower()}{OLD_EXT}"
    if not old_file_path.is_file():
        LOGGER.debug(f"Expected BetaBeat.src output at '{old_file_path.absolute()}' is not a file, skipping")

    dframe = tfs.read(old_file)
    dframe = dframe.rename(columns={f"STD{plane}": f"{ERR}{plane}"})
    dframe[f"{DELTA}{plane}"] = df_diff(dframe, f"{plane}", f"{plane}{MDL}")
    dframe[f"{ERR}{DELTA}{plane}"] = dframe.loc[:, f"{ERR}{plane}"].to_numpy()
    tfs.write(Path(opt.outputdir) / f"{new_file_name}{plane.lower()}{EXT}", dframe)


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
    old_file_path = Path(opt.inputdir) / f"get{old_file_name}{plane.lower()}{OLD_EXT}"
    if not old_file_path.is_file():
        LOGGER.debug(f"Expected BetaBeat.src output at '{old_file_path.absolute()}' is not a file, skipping")

    dframe = tfs.read(old_file)
    dframe = dframe.rename(columns={f"STDD{plane}": f"{ERR}D{plane}"})
    dframe[f"{DELTA}D{plane}"] = df_diff(dframe, f"D{plane}", f"D{plane}{MDL}")
    dframe[f"{ERR}{DELTA}D{plane}"] = dframe.loc[:, f"{ERR}D{plane}"].to_numpy()
    tfs.write(Path(opt.outputdir) / f"{new_file_name}{plane.lower()}{EXT}", dframe)


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
    old_file_path = Path(opt.inputdir) / f"get{old_file_name}{plane.lower()}{OLD_EXT}"
    if not old_file_path.is_file():
        LOGGER.debug(f"Expected BetaBeat.src output at '{old_file_path.absolute()}' is not a file, skipping")

    dframe = tfs.read(old_file)
    dframe = dframe.rename(columns={f"STDND{plane}": f"{ERR}ND{plane}"})
    dframe[f"{DELTA}ND{plane}"] = df_diff(dframe, f"ND{plane}", f"ND{plane}{MDL}")
    dframe[f"{ERR}{DELTA}ND{plane}"] = dframe.loc[:, f"{ERR}ND{plane}"].to_numpy()

    if f"D{plane}" in dframe.columns:
        dframe = dframe.rename(columns={f"STDD{plane}": f"{ERR}D{plane}"})
        dframe[f"{DELTA}D{plane}"] = df_diff(dframe, f"D{plane}", f"D{plane}{MDL}")

    tfs.write(Path(opt.outputdir) / f"{new_file_name}{plane.lower()}{EXT}", dframe)


def convert_old_coupling(opt: EntryPointParameters, old_file_name: str = "couple", new_file_name: str = "coupling_f") -> None:
    """
    Looks in the provided directory for expected coupling file from ``BetaBeat.src``, converts it to the
    output format used by ``omc3`` and  write them to the new location.

    The file naming should be getcouple(x,y).out, with the following expected columns: NAME, S, COUNT,
    F1001W, FWSTD1, F1001R, F1001I, F1010W, FWSTD2, F1010R, F1010I, Q1001, Q1001STD, Q1010, Q1010STD,
    MDLF1001R, MDLF1001I, MDLF1010R, MDLF1010I.

    Args:
        opt (EntryPointParameters): The entrypoint parameters parsed from the command line.
        old_file_name (str): the standard naming for the old output file.
        new_file_name (str): the standard naming for the new converted file.
    """
    old_file_path = Path(opt.inputdir) / f"get{old_file_name}{opt.suffix}{OLD_EXT}"
    if not old_file_path.is_file():
        LOGGER.debug(f"Expected BetaBeat.src output at '{old_file_path.absolute()}' is not a file, skipping")

    dframe = tfs.read(old_file)
    rdt_dfs = {
        "1001": dframe.loc[: , ["S", "COUNT", "F1001W", "FWSTD1", "F1001R", "F1001I",
                                "Q1001", "Q1001STD", "MDLF1001R", "MDLF1001I"]],
        "1010": dframe.loc[: , ["S", "COUNT", "F1010W", "FWSTD2", "F1010R", "F1010I",
                                "Q1010", "Q1010STD", "MDLF1010R", "MDLF1010I"]],
    }

    for i, rdt in enumerate(("1001", "1010")):
        rdt_dfs[rdt] = rdt_dfs[rdt].drop(columns=[f"MDLF{rdt}R", f"MDLF{rdt}I"])
        rdt_dfs[rdt] = rdt_dfs[rdt].rename(
            columns={
                f"F{rdt}W": "AMP",
                f"FWSTD{i+1}": f"{ERR}AMP",
                f"Q{rdt}": "PHASE",
                f"Q{rdt}STD": f"{ERR}PHASE",
                f"F{rdt}R": "REAL",
                f"F{rdt}I": "IMAG",
            }
        )
        tfs.write(Path(opt.outputdir) / f"{new_file_name}{rdt}{EXT}", rdt_dfs[rdt])


if __name__ == "__main__":
    converter_entrypoint()
