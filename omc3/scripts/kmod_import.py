""" 
Import K-Modulation Results
---------------------------

Imports K-Mod data and writes them into a file containing beta data, 
in the same format as beta-from-phase or beta-from-amplitude.
This data can then be easily used for the same purposes, e.g. global correction.


**Arguments:**

*--Required--*

- **meas_files** *(PathOrStr)*:

    Paths to the Kmod results files to import.
    Can be either the TFS-files directly or a path to a folder containing them.


- **model** *(PathOrStr)*:

    Path to the model twiss file, or a folder containing 'twiss_elemtents.dat'. 
    The model determines which elements to keep.


- **beam** *(int)*:

    Beam for which to import.


*--Optional--*

- **output_dir** *(PathOrStr)*:

    Path to the directory where to write the output files.

"""
from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING

import tfs
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint

from omc3.definitions.constants import PLANES
from omc3.model.constants import TWISS_ELEMENTS_DAT
from omc3.optics_measurements.constants import (
    BEAM,
    BEAM_DIR,
    BETA,
    BETA_KMOD_FILENAME,
    BETA_STAR_FILENAME,
    BETASTAR,
    BETAWAIST,
    DELTA,
    ERR,
    EXT,
    LABEL,
    MDL,
    NAME,
    S_LOCATION,
    TIME,
    TUNE,
    WAIST,
    S,
)
from omc3.optics_measurements.constants import (
    KMOD_PHASE_ADV as PHASEADV,
)
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, PathOrStrOrDataFrame, save_config

if TYPE_CHECKING:
    from collections.abc import Sequence
    from generic_parser import DotDict

LOG = logging_tools.get_logger(__name__)


BPM_RESULTS_ID: str = "BPM"
BETASTAR_RESULTS_ID: str = "BETASTAR"


def _get_params() -> EntryPointParameters:
    """
    Creates and returns the parameters for the kmod_output function.
    """
    params = EntryPointParameters()
    params.add_parameter(
        name="measurements",
        required=True,
        nargs="+",
        type=PathOrStrOrDataFrame,
        help="Paths to the K-modulation results files to import. "
             "Can be either the TFS-files directly or a path to a folder containing them."
    )
    params.add_parameter(
        name="model",
        required=True,
        type=PathOrStrOrDataFrame,
        help="Path to model twiss file, or a folder containing 'twiss_elemtents.dat'. "
             "The model determines which elements to keep."
    )
    params.add_parameter(
        name="beam",
        required=True,
        type=int,
        help="Beam for which to import."
    )
    params.add_parameter(
        name="output_dir",
        type=PathOrStr,
        help="Path to the directory where to write the output files."
    )
    return params


@entrypoint(_get_params(), strict=True)
def import_kmod_data(opt: DotDict) -> dict[str, tfs.TfsDataFrame]:
    """
    Reads model and measurement files to calculate differences in beta functions 
    and writes the results to output files.
    
    Args:
        measurements (Sequence[Path|str]):
            A sequence of k-modulation results files to import. 
            This can include either single measurements (e.g., 'lsa_results.tfs'),
            averaged results (e.g., 'averaged_bpm_beam1_ip1_beta0.22m.tfs') or a 
            path to the folder containing multiple of such files.
        
        model (Path|str):
            Path to the model twiss file, or a folder containing 'twiss_elements.dat'.
            Determines which elements to keep, i.e. `twiss.dat` keeps only the BPMs,
            `twiss_elements.dat` keeps BPMs and IPs.
        
        output_dir (Path|str):
            Path to the output directory, i.e. the optics-measurement directory 
            into which to import these K-Modulation results.
    
    Returns:
        Dictionary of kmod-DataFrames by planes and IDs (BPM, BETASTAR).
    """
    LOG.info("Starting Kmod Import.")

    # Prepare output dir    
    if opt.output_dir is not None:
        opt.output_dir = Path(opt.output_dir)
        opt.output_dir.mkdir(exist_ok=True)
        save_config(opt.output_dir, opt, __file__)

    # read data
    df_model = read_model_df(opt.model)
    bpm_results_list, betastar_results_list = _read_kmod_results(opt.measurements, beam=opt.beam)

    # create new dataframes
    dfs = {}
    if len(bpm_results_list):
        dfs.update(convert_bpm_results(bpm_results_list, df_model))

    if len(betastar_results_list):
        dfs.update(convert_betastar_results(betastar_results_list, df_model, beam=opt.beam))
    
    # write output
    if opt.output_dir is not None:
        _write_output(dfs, opt.output_dir)


# BPM Results Conversion ---

def convert_bpm_results(
    bpm_results_list: Sequence[tfs.TfsDataFrame], 
    df_model: tfs.TfsDataFrame, 
    ) -> dict[str, tfs.TfsDataFrame]:
    """ Convert K-Modulation BPM results to kmod-DataFrames
    that can be placed in the optics-measurement directory.

    Args:
        bpm_results_list (Sequence[tfs.TfsDataFrame]): List of BPM results.
        df_model (tfs.TfsDataFrame): Model data. 

    Returns:
        dict[str, tfs.TfsDataFrame]: Dictionary of kmod-DataFrames by planes and with BPM ID as key.
    """
    LOG.debug("Converting K-modulation BPM results")

    # merge files
    bpm_results_list = [df.set_index(NAME, drop=True) if NAME in df.columns else df for df in bpm_results_list]
    kmod_results = tfs.concat(bpm_results_list, join="inner",)
    df_model = _sync_model_index(kmod_results, df_model) 

    dfs = {}
    for plane in PLANES:
        beta_kmod = tfs.TfsDataFrame(index=df_model.index)
        
        # copy s and beta
        beta_kmod.loc[:, S] = df_model.loc[:, S]
        beta_kmod.loc[:, f"{BETA}{plane}{MDL}"] = df_model.loc[:, f"{BETA}{plane}"]
        beta_kmod.loc[:, f"{BETA}{plane}"] = kmod_results.loc[:, f"{BETA}{plane}"]
        beta_kmod.loc[:, f"{ERR}{BETA}{plane}"] = kmod_results.loc[:, f"{ERR}{BETA}{plane}"]

        # model-delta and beta-beating
        beta_kmod.loc[:, f"{DELTA}{BETA}{plane}{MDL}"] = (
            beta_kmod[f"{BETA}{plane}"] - beta_kmod[f"{BETA}{plane}{MDL}"]
        )
        beta_kmod.loc[:, f"{DELTA}{BETA}{plane}"] = (
            beta_kmod[f"{DELTA}{BETA}{plane}{MDL}"] / beta_kmod[f"{BETA}{plane}{MDL}"]
        )
        beta_kmod.loc[:, f"{ERR}{DELTA}{BETA}{plane}"] = (
            beta_kmod[f"{ERR}{BETA}{plane}"] / beta_kmod[f"{BETA}{plane}{MDL}"]
        )

        # tune
        beta_kmod.headers[f"{TUNE}1"] = df_model.headers[f"{TUNE}1"] % 1
        beta_kmod.headers[f"{TUNE}2"] = df_model.headers[f"{TUNE}2"] % 1

        beta_kmod = beta_kmod.sort_values(by=S)

        dfs[f"{BPM_RESULTS_ID}{plane}"] = beta_kmod

    return dfs


# BetaStar Results Conversion ---

def convert_betastar_results(
    betastar_results_list: Sequence[tfs.TfsDataFrame], 
    df_model: tfs.TfsDataFrame,
    beam: int, 
    ) -> dict[str, tfs.TfsDataFrame]:
    """ Convert K-Modulation BetaStar results to kmod-DataFrames
    that can be placed in the optics-measurement directory. 
    
    Args:
        betastar_results_list (Sequence[tfs.TfsDataFrame]): List of BetaStar results.
        df_model (tfs.TfsDataFrame): Model data.
        beam (int): Beam to import.
    
    Returns:
        dict[str, tfs.TfsDataFrame]: Dictionary of kmod-DataFrames by planes and with BETASTAR ID as key.
    """
    LOG.debug("Converting K-modulation BetaStar results")

    # merge files and set index
    kmod_results = tfs.concat(betastar_results_list, join="inner",)
    if BEAM in kmod_results.columns or kmod_results.index.name == BEAM:  # averaged file
        if beam is None:
            raise ValueError("Need to give beam when importing averaged betastar files.")

        try: 
            # as column
            kmod_results = kmod_results.loc[kmod_results[BEAM] == beam, :]
            kmod_results = kmod_results.drop(columns=[BEAM])
        except KeyError:
            # already as index
            kmod_results = kmod_results.loc[beam, :]

        kmod_results = kmod_results.set_index(NAME, drop=True)
    
    else:
        kmod_results.index = kmod_results[LABEL].apply(lambda s: f"IP{s[-1]}")
        kmod_results = kmod_results.drop(columns=[LABEL, TIME])

    df_model = _sync_model_index(kmod_results, df_model) 

    dfs = {}
    for plane in PLANES:
        beta_kmod = tfs.TfsDataFrame(index=df_model.index)
        
        # copy s and beta from model
        beta_kmod.loc[:, S] = df_model.loc[:, S]
        beta_kmod.loc[:, f"{BETASTAR}{plane}{MDL}"] = df_model.loc[:, f"{BETA}{plane}"]

        # copy columns with errors from results
        columns = [
            f"{column}{plane}" for column in (BETASTAR, BETAWAIST, WAIST, PHASEADV)] + [f"{WAIST}{plane}{S_LOCATION}"
        ]
        for column in columns:
            beta_kmod.loc[:, f"{column}"] = kmod_results[f"{column}"]
            beta_kmod.loc[:, f"{ERR}{column}"] = kmod_results[f"{ERR}{column}"]


        # model-delta and beta-beating
        beta_kmod.loc[:, f"{DELTA}{BETASTAR}{plane}{MDL}"] = (
            beta_kmod[f"{BETASTAR}{plane}"] - beta_kmod[f"{BETASTAR}{plane}{MDL}"]
        )
        beta_kmod.loc[:, f"{DELTA}{BETASTAR}{plane}"] = (
            beta_kmod[f"{DELTA}{BETASTAR}{plane}{MDL}"] / beta_kmod[f"{BETASTAR}{plane}{MDL}"]
        )
        beta_kmod.loc[:, f"{ERR}{DELTA}{BETASTAR}{plane}"] = (
            beta_kmod[f"{ERR}{BETASTAR}{plane}"] / beta_kmod[f"{BETASTAR}{plane}{MDL}"]
        )

        # tune
        beta_kmod.headers[f"{TUNE}1"] = df_model.headers[f"{TUNE}1"] % 1
        beta_kmod.headers[f"{TUNE}2"] = df_model.headers[f"{TUNE}2"] % 1

        beta_kmod = beta_kmod.sort_values(by=S)

        dfs[f"{BETASTAR_RESULTS_ID}{plane}"] = beta_kmod

    return dfs


def _sync_model_index(kmod_results: tfs.TfsDataFrame, df_model: tfs.TfsDataFrame):
    missing_elements = kmod_results.index.difference(df_model.index)
    if len(missing_elements):
        msg =  (
            f"Elements {missing_elements} not found in the model. "
            "Make sure to use the `elements` file as input!"
        )
        raise NameError(msg)

    return df_model.loc[kmod_results.index, :]


# IO ---

def read_model_df(model_path: Path | str | tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """ Read model twiss file, 
    either directly or twiss_elements.dat from a folder. 
    """
    try:
        model_path = Path(model_path)
    except TypeError:
        return model_path  # is a TfsDataFrame

    if model_path.is_dir():
        return tfs.read(model_path / TWISS_ELEMENTS_DAT, index=NAME)
   
    return tfs.read(model_path, index=NAME)


def _read_kmod_results(paths: Sequence[Path | str], beam: int    
    ) -> tuple[list[tfs.TfsDataFrame], list[tfs.TfsDataFrame]]:
    """ Read K-modulation results from a list of paths and sort into bpm and betastar types. """
    # read all files ---
    all_dfs = []
    for path in paths:
        try:
            path = Path(path)
        except TypeError: 
            # is a TfsDataFrame
            all_dfs.append(path) 
            continue
        
        if path.is_dir():
            # If the given path was a K-Mod output directory, the tfs might be in sub-dirs per beam
            beam_dir = path / f"{BEAM_DIR}{beam}"
            if beam_dir.exists():
                all_dfs.extend(tfs.read(file_path) for file_path in beam_dir.glob(f"*{EXT}"))
                continue

            # otherwise, read all files in the given dir
            all_dfs.extend(tfs.read(file_path) for file_path in path.glob(f"*{EXT}"))
            continue
        
        # Not a folder, must be a file
        all_dfs.append(tfs.read(path))
    
    # sort into bpm and betastar --
    bpm_results = _filter_bpm_results(all_dfs, beam=beam)
    betastar_results = _filter_betastar_results(all_dfs)
    LOG.debug(
        f"Found {len(bpm_results)} BPM results and {len(betastar_results)} betastar results for beam {beam}."
    )
    return bpm_results, betastar_results


def _is_bpm_df(df: tfs.TfsDataFrame, beam: int) -> bool:
    """ Check if the given df is a BPM results file for the given beam. """
    # bpm files must have a beta column
    if f"{BETA}X" not in df.columns:
        return False
    
    # They should also have at least one element (e.g. the BPM) matching the beam
    elements = df.index if NAME not in df.columns else df[NAME]
    return elements.str.match(fr".*\.B{beam}$", flags=re.IGNORECASE).any()


def _filter_bpm_results(dfs: Sequence[tfs.TfsDataFrame], beam: int) -> list[tfs.TfsDataFrame]:
    return [df for df in dfs if _is_bpm_df(df, beam)]


def _filter_betastar_results(dfs: Sequence[tfs.TfsDataFrame]) -> list[tfs.TfsDataFrame]:
    return [df for df in dfs if f"{BETASTAR}X" in df.columns]


def _write_output(dfs: dict[str, tfs.TfsDataFrame], output_dir: Path):
    """ Write output files if data exists. """
    for id_, filename in ((BETASTAR_RESULTS_ID, BETA_STAR_FILENAME), (BPM_RESULTS_ID, BETA_KMOD_FILENAME)):
        for plane in PLANES:
            df = dfs.get(f"{id_}{plane}")
            if df is None:
                continue
            outfile = output_dir / f"{filename}{plane.lower()}{EXT}"
            LOG.info(f"Writing output file {outfile}")
            tfs.write(outfile, df, save_index=NAME)


# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    import_kmod_data()
