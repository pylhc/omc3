""" 
K-Mod Luminosity Imbalance
--------------------------

Calculate the luminosity imbalance from the averaged K-modulation results,
based on the effective betas for two IPs, i.e. assuming that apart from the 
beta-function all other parameters are identical for the two IPs.

This script needs the data in the format of the average script `kmod_average.py`,
as the imbalance is calculated over both beams and this data is only available 
at that point.
The dataframes therefore need to have the columns `BEAM`, `BETSTAR` and `ERRBETSTAR`,
and data for both beams needs to be available.

.. warning::
        You need to provide the data for exactly two of the four IP's.


**Arguments:**

*--Optional--*

- **ip1** *(PathOrStrOrDataFrame)*:

    Path or DataFrame of the averaged beta-star results of IP1.


- **ip2** *(PathOrStrOrDataFrame)*:

    Path or DataFrame of the averaged beta-star results of IP2.


- **ip5** *(PathOrStrOrDataFrame)*:

    Path or DataFrame of the averaged beta-star results of IP5.


- **ip8** *(PathOrStrOrDataFrame)*:

    Path or DataFrame of the averaged beta-star results of IP8.


- **betastar** *(float)*:

    Model beta-star values (x, y) of measurements. Only used for filename.


- **output_dir** *(PathOrStr)*:

    Path to the directory where to write the output files.

"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tfs
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint

from omc3.optics_measurements.constants import (
    BEAM,
    BETASTAR,
    EFFECTIVE_BETAS_FILENAME,
    ERR,
    EXT,
    IMBALANCE,
    LUMINOSITY,
    MDL,
    NAME,
)
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, PathOrStrOrDataFrame, save_config

if TYPE_CHECKING:
    from generic_parser import DotDict

LOG = logging_tools.get_logger(__name__)
IPS: tuple[str, ...] = ("ip1", "ip2", "ip5", "ip8")

def _get_params() -> EntryPointParameters:
    """
    A function to initialize and return parameters for kmod luminosity calculation.
    """
    params = EntryPointParameters()
    for ip in IPS:
        params.add_parameter(
            name=ip,
            type=PathOrStrOrDataFrame,
            help=f"Path or DataFrame of the averaged beta-star results of {ip.upper()}.",
        )
    params.add_parameter(
        name="betastar",
        type=float,
        nargs="+",
        help="Model beta-star values (x, y) of measurements. Only used for filename.",
    )
    params.add_parameter(
        name="output_dir",
        type=PathOrStr,
        help="Path to the directory where to write the output files.",
    )
    return params


@entrypoint(_get_params(), strict=True)
def calculate_lumi_imbalance(opt: DotDict) -> tfs.TfsDataFrame:
    """ Perform the luminosity imbalance calculation. """
    output_path = opt.output_dir 
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        opt_cp = copy.copy(opt)
        for ip in IPS:
            if not isinstance(opt_cp[ip], (Path, str)):
                opt_cp[ip] = "(was provided as DataFrame)"
        save_config(output_path, opt_cp, __file__)

    dfs = _read_and_check_dataframes(opt)
    df = get_lumi_imbalance_df(**dfs)

    if output_path is not None:
        if opt.betastar is None:
            raise ValueError("Betastar not given. Cannot write out results.")
        if len(opt.betastar) == 1:
            opt.betastar = [opt.betastar[0], opt.betastar[0]]

        tfs.write(
            output_path / f"{EFFECTIVE_BETAS_FILENAME.format(betastar_x=opt.betastar[0], betastar_y=opt.betastar[1])}{EXT}", 
            df, 
            save_index=NAME
        )
    
    return df


def _read_and_check_dataframes(opt: DotDict) -> dict[str, tfs.TfsDataFrame]:
    """
    Read the given DataFrames if needed, check them for validity and return a dictionary.
    """
    dfs = {}
    for ip in IPS:
        df = opt.get(ip, None)
        if df is None:
            continue

        if isinstance(df, (Path, str)):
            try:
                df = tfs.read(df, index=BEAM)
            except KeyError as e:
                msg = (
                    f"Dataframe '{df}' does not contain a '{BEAM}' column."
                    "You need to run the `kmod_average` script on data for both beams "
                    "before you can calulate the luminosity imbalance."
                )
                raise KeyError(msg) from e

        if BEAM in df.columns:
            df = df.set_index(BEAM)

        for beam in (1, 2):
            if beam not in df.index:
                msg = (
                    f"Dataframe '{df}' does not seem to contain data per beam."
                    "You need to run the `kmod_average` script on data for both beams "
                    "before you can calulate the luminosity imbalance."
                )
                raise KeyError(msg)

        dfs[ip] = df
    
    if len(dfs) != 2:
        msg = (
            "Lumi inbalance can only be calculated for exactly two IPs, "
            f"but instead {len(dfs)} were given."
        )
        raise ValueError(msg)

    return dfs


def get_lumi_imbalance_df(**kwargs) -> tfs.TfsDataFrame:
    """
    Calculate the effective beta stars and the luminosity imbalance from the input dataframes.

    Args:
        ipA (tfs.TfsDataFrame): a `TfsDataFrame` with the averaged results from a kmod analysis, for IP_A.
        ipB (tfs.TfsDataFrame): a `TfsDataFrame` with the averaged results from a kmod analysis, for IP_B.
        (Actually, any name that ends with an integer is fine.)
    
    Returns:
        tfs.TfsDataFrame with effective beta stars per IP and the luminosity imbalance added to the header.
    """
    df_effective_betas = tfs.TfsDataFrame()
    for ip, df in kwargs.items():
        df_effective_betas.loc[ip.upper(), [f'{BETASTAR}', f'{ERR}{BETASTAR}']] = get_effective_beta_star_w_err(df)
    
    ip_a, ip_b = df_effective_betas.index
    lumi_imb, lumi_imb_err = get_imbalance_w_err(
        *tuple(df_effective_betas.loc[ip_a, :]), 
        *tuple(df_effective_betas.loc[ip_b, :])
    )
    
    df_effective_betas.headers[f'{LUMINOSITY}{IMBALANCE}'] = lumi_imb
    df_effective_betas.headers[f'{ERR}{LUMINOSITY}{IMBALANCE}'] = lumi_imb_err
    return df_effective_betas


def get_imbalance_w_err(ipA_beta: float, ipA_beta_err: float, ipB_beta: float, ipB_beta_err: float) -> tuple[float, float]:
    """
    Calculate the luminosity imbalance IP_A / IP_B  and its error, based on the effective betas for IP_A and IP_B.
    This implies, that all the other beam parameters (see e.g. Eq(17): https://cds.cern.ch/record/941318/files/p361.pdf) 
    are the same for both IPs.
    """
    result = ipB_beta / ipA_beta  # inverse due to beta in the denominator for lumi
    err = result * np.sqrt((ipB_beta_err/ipB_beta)**2 + (ipA_beta_err/ipA_beta)**2)
    return result, err


def get_effective_beta_star_w_err(df_ip: tfs.TfsDataFrame) -> tuple[float]:
    """ Calculates the effective beta*, i.e. the denominator of the luminosity as given in e.g. 
    Eq(17): https://cds.cern.ch/record/941318/files/p361.pdf , without any constants 
    as we assume here that these are equal for both IPs, and we only care about the ratio. 
    """
    b1x, b1y, b2x, b2y = _get_betastar_beams(df_ip)
    db1x, db2x, db1y, db2y = _get_betastar_beams(df_ip, errors=True) 

    # Effective beta:
    sqrt_x = np.sqrt(b1x + b2x)
    sqrt_y = np.sqrt(b1y + b2y)
    beta = 0.5 * sqrt_x * sqrt_y  # division by 2 because of averaging, see Eq. 16 -> Eq. 17 in reference

    # Error propagation:
    dbeta_db1x = dbeta_db2x = 0.25 * sqrt_y / sqrt_x
    dbeta_db1y = dbeta_db2y = 0.25 * sqrt_x / sqrt_y 
    sigma = np.sqrt((dbeta_db1x * db1x) ** 2 + (dbeta_db1y * db1y) ** 2 +
                    (dbeta_db2x * db2x) ** 2 + (dbeta_db2y * db2y) ** 2)
    return beta, sigma


def _get_betastar_beams(df_ip: tfs.TfsDataFrame, errors: bool = False) -> tuple[float, float, float, float]:
    """ Get betastar x and y for both beam 1 and beam 2. Order: b1x, b1y, b2x, b2y """
    return (*_get_betastar_xy(df_ip, 1, errors), *_get_betastar_xy(df_ip, 2, errors))


def _get_betastar_xy(df_ip: tfs.TfsDataFrame, beam: int, errors: bool = False) -> tuple[float, float]:
    """ Get betastar x and y for the given beam. """
    if errors:
        return tuple(df_ip.loc[beam, [f'{ERR}{BETASTAR}X', f'{ERR}{BETASTAR}Y']])
    return tuple(df_ip.loc[beam, [f'{BETASTAR}X', f'{BETASTAR}Y']])


# Commandline Entry Point ------------------------------------------------------

if __name__ == "__main__":
    calculate_lumi_imbalance()