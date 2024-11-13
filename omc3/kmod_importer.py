""" 
Full Import of K-Modulation Results
-----------------------------------

Performs the full import procedure of the "raw" K-Modulation results,
which come from the K-modulation GUI. Each `measurement` needs to be the path
to the main output folder of a K-modulation run, containing `B1` and `B2` folders.

The results are first sorted by IP and averaged. The averaged results are
written into a sub-folder of the given `output_dir`.

If data for both beams is present, these averages are then used to calculate the 
luminosity imbalance between each combination of IPs. 
These results are again written out into the same sub-folder of the given `output_dir`.

Finally, the averaged results for the given `beam` are then written out into 
the `beta_kmod` and `betastar` tfs-files in the `output_dir`.


**Arguments:**

*--Required--*

- **meas_paths** *(PathOrStr)*:

    Directories of K-modulation results to import. 
    These need to be the paths to the root-folders containing B1 and B2 sub-dirs.


- **model** *(PathOrStr)*:

    Path to the model.


- **beam** *(int)*:

    Beam for which to import.
    

- **output_dir** *(PathOrStr)*:

    Path to the directory where to write the output files.


*--Optional--*

- **show_plots**:

    Show the plots.

    action: ``store_true``


"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint

from omc3.optics_measurements.constants import (
    BEAM_DIR,
    BETA,
    ERR,
    EXT,
    IMBALANCE,
    LSA_FILE_NAME,
    LUMINOSITY,
    NAME,
)
from omc3.scripts.kmod_average import average_kmod_results
from omc3.scripts.kmod_import import import_kmod_data, read_model_df
from omc3.scripts.kmod_lumi_imbalance import IPS, calculate_lumi_imbalance
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config

if TYPE_CHECKING:
    from collections.abc import Sequence
    from generic_parser import DotDict

LOG = logging_tools.get_logger(__name__)


AVERAGE_DIR = "kmod_averaged"


def _get_params():
    """
    Creates and returns the parameters for the kmod_output function.
    
    """
    params = EntryPointParameters()
    params.add_parameter(
        name="meas_paths",
        required=True,
        nargs='+',
        type=PathOrStr,
        help="Directories of K-modulation results to import. "
             "These need to be the paths to the root-folders containing B1 and B2 sub-dirs."
    )
    params.add_parameter(
        name="model",
        required=True,
        type=PathOrStr,
        help="Path to the model."
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
        required=True,
        help="Path to the directory where to write the output files."
    )
    params.add_parameter(
        name="show_plots", 
        action="store_true", 
        help="Show the plots."
    )
    return params


@entrypoint(_get_params(), strict=True)
def import_kmod_results(opt: DotDict) -> None:
    """
    Performs the full import procedure of the "raw" K-Modulation results.
    
    Args:
        meas_paths (Sequence[Path|str]):
            Directories of K-modulation results to import.
            These need to be the paths to the root-folders containing B1 and B2 sub-dirs.
        
        model (Path|str):
            Path to the model Twiss file.
        
        beam (int):
            Beam for which to import.
        
        output_dir (Path|str):
            Path to the output directory, i.e. the optics-measurement directory 
            into which to import these K-Modulation results.
        
        show_plots (bool):
            If True, show the plots. Default: False.

    
    Returns:
        Dictionary of kmod-DataFrames by planes.
    """
    LOG.info("Starting full K-modulation import.")

    # Prepare IO ---    
    opt.output_dir = Path(opt.output_dir)
    opt.output_dir.mkdir(exist_ok=True)
    save_config(opt.output_dir, opt, __file__)

    average_output_dir = opt.output_dir / AVERAGE_DIR
    average_output_dir.mkdir(exist_ok=True)

    df_model = read_model_df(opt.model)
    
    # Perform averaging and import ---
    averaged_results = average_all_results(
        meas_paths=opt.meas_paths, 
        df_model=df_model, 
        beam=opt.beam, 
        output_dir=average_output_dir
    )
    
    calculate_all_lumi_imbalances(
        averaged_results, 
        df_model=df_model, 
        output_dir=average_output_dir
    )

    results_list = [
        df 
        for ip in averaged_results.keys() 
        for df in (
            averaged_results[ip][0], # beta-star results
            averaged_results[ip][opt.beam]  # bpm results of the specific beam
        )
    ]
    import_kmod_data(
        model=df_model,
        measurements=results_list,
        output_dir=opt.output_dir,
        beam=opt.beam,
    )


# Averaging ---

def average_all_results(
    meas_paths: Sequence[Path | str],
    df_model: tfs.TfsDataFrame,
    beam: int,
    output_dir: Path | str,
    show_plots: bool = False,
    ) -> dict[str, dict[int, tfs.TfsDataFrame]]:
    """ Averages all kmod results.

    Args:
        meas_paths (Sequence[Path | str]): Paths to the K-modulation results. 
        df_model (tfs.TfsDataFrame): DataFrame with the model. 
        beam (int): Beam for which to average. 
        output_dir (Path | str, optional): Path to the output directory. Defaults to None.
        show_plots (bool, optional): If True, show the plots. Defaults to False.

    Returns:
        dict[int, tfs.TfsDataFrame]: Averaged kmod results, sorted by IP. 
    """
    sorted_paths = _sort_paths_by_ip(meas_paths, beam)

    averaged_results = {}
    for ip, paths in sorted_paths.items():
        LOG.debug(f"Averaging IP {ip}")

        average = average_kmod_results(
            ip=int(ip[-1]),
            betastar=_get_betastar(df_model, ip),
            meas_paths=paths,
            output_dir=output_dir,
            plot=True,
            show_plots=show_plots
        )
        averaged_results[ip] = average

    return averaged_results


def _sort_paths_by_ip(paths: Sequence[str | Path], beam: int) -> dict[str, list[str | Path]]:
    """ Sorts the kmod results files by IP. 
    
    Identification of the IP is done by reading the `lsa_results.tfs` files.
    """
    sorted_paths = defaultdict(list)
    for path in paths:
        path = Path(path)
        beam_dir = path / f"{BEAM_DIR}{beam}"
        lsa_results = beam_dir / f"{LSA_FILE_NAME}{EXT}"
        df = tfs.read(lsa_results, index=NAME)
        ip = [ip.upper() for ip in IPS if ip.upper() in df.index][0]
        sorted_paths[ip].append(path)
    return sorted_paths


# Lumi Imbalance ---

def calculate_all_lumi_imbalances(
    averaged_results: dict[str, dict[int, tfs.TfsDataFrame]], 
    df_model: tfs.TfsDataFrame,
    output_dir: Path | str = None
    ) -> None:
    """ Calculates the luminosity imbalance between two IPs.
    
    Args:
        averaged_results (dict[str, dict[int, tfs.TfsDataFrame]]): Averaged kmod results, sorted by IP. 
        df_model (tfs.TfsDataFrame): DataFrame with the model. 
        output_dir (Path | str, optional): Path to the output directory. Defaults to None.

    Returns:
        tfs.TfsDataFrame: DataFrame with the luminosity imbalance.
    """
    sets_of_ips = list(combinations(averaged_results.keys(), 2))
    for (ipA, ipB) in sets_of_ips:
        LOG.debug(f"Calculating lumi imbalance between {ipA} and {ipB}")
        betastar = _get_betastar(df_model, ipA)  # does not really matter which IP, for output name only

        # Calculate luminosity imbalance
        data = {ip.lower(): averaged_results[ip][0] for ip in (ipA, ipB)}
        try:
            df = calculate_lumi_imbalance(**data, output_dir=output_dir, betastar=betastar)
        except KeyError as e:
            # Most likely because not all data available (e.g. only one beam).
            LOG.debug(f"Could not calculate lumi imbalance between {ipA} and {ipB}. Skipping.", exc_info=e)
            continue

        # Print luminosity imbalance
        imb, err_imb = df.headers[f"{LUMINOSITY}{IMBALANCE}"], df.headers[f"{ERR}{LUMINOSITY}{IMBALANCE}"]
        LOG.info(f"Luminosity imbalance between {ipA} and {ipB}: {imb:.2e} +/- {err_imb:.2e}")


def _get_betastar(df_model: tfs.TfsDataFrame, ip: str) -> list[float, float]:
    # return [round(bstar, 3) for bstar in df_model.loc[ip, [f"{BETA}X", f"{BETA}Y"]]]
    return df_model.loc[ip, [f"{BETA}X", f"{BETA}Y"]].tolist()


# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    import_kmod_data()
