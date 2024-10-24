""" 
K-Mod Luminosity Imbalance
--------------------------

Calculate the luminosity imbalance from the k-mod results.

**Arguments:**

*--Required--*

- **ip1** *(PathOrStrOrDataFrame)*:

    Path or DataFrame of the beta-star results of IP1.


- **ip5** *(PathOrStrOrDataFrame)*:

    Path or DataFrame of the beta-star results of IP5.


*--Optional--*

- **output_dir** *(PathOrStr)*:

    Path to the directory where to write the output files.

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tfs
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint

from omc3.kmod.constants import (
    BEAM,
    BETASTAR,
    EFFECTIVE_BETAS_FILENAME,
    ERR,
    EXT,
    IMBALACE,
    IP,
    LUMINOSITY,
    MDL,
)
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, PathOrStrOrDataFrame, save_config

if TYPE_CHECKING:
    from generic_parser import DotDict

LOG = logging_tools.get_logger(__name__)


def _get_params():
    """
    A function to initialize and return parameters for kmod luminosity calculation.
    """
    params = EntryPointParameters()
    params.add_parameter(
        name="ip1",
        required=True,
        type=PathOrStrOrDataFrame,
        help="Path or DataFrame of the beta-star results of IP1.",
    )
    params.add_parameter(
        name="ip5",
        required=True,
        type=PathOrStrOrDataFrame,
        help="Path or DataFrame of the beta-star results of IP5.",
    )
    params.add_parameter(
        name="output_dir",
        type=PathOrStr,
        help="Path to the directory where to write the output files.",
    )
    return params


@entrypoint(_get_params(), strict=True)
def calculate_lumi_imbalance_entrypoint(opt: DotDict) -> tfs.TfsDataFrame:
    output_path = opt.output_dir 
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        if isinstance(opt.ip1, (Path, str)) and isinstance(opt.ip5, (Path, str)):
            save_config(output_path, opt, __file__)

    dfs = {}
    for ip in ("ip1", "ip5"):
        df = opt[ip]
        if isinstance(df, (Path, str)):
            df = tfs.read(df, index=BEAM)
        dfs[f"df_{ip}"] = df

    df = get_lumi_imbalance(**dfs)
    betastar_x, betastar_y = dfs["df_ip1"].loc[1, [f'{BETASTAR}X{MDL}', f'{BETASTAR}Y{MDL}']].values

    if output_path is not None:
        tfs.write(
            output_path / f"{EFFECTIVE_BETAS_FILENAME.format(betastar_x=betastar_x, betastar_y=betastar_y)}{EXT}", df, 
            save_index=IP
        )
    
    return df


def get_lumi_imbalance(df_ip1: tfs.TfsDataFrame, df_ip5: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """
    Calculate the effective beta stars and the luminosity imbalance from the input dataframes.

    Args:
        df_ip1 (tfs.TfsDataFrame): a `TfsDataFrame` with the results from a kmod analysis, for IP1.
        df_ip5 (tfs.TfsDataFrame): a `TfsDataFrame` with the results from a kmod analysis, for IP5.
    
    Returns:
        tfs.TfsDataFrame with effective beta stars per IP and the luminosity imbalance added to the header.
    """
    df_effective_betas = tfs.TfsDataFrame()
    for ip, df in ((1, df_ip1), (5, df_ip5)):
        df_effective_betas.loc[ip, [f'{BETASTAR}', f'{ERR}{BETASTAR}']] = get_effective_beta_star_w_err(df)
    
    lumi_imb, lumi_imb_err = get_imbalance_w_err(
        *tuple(df_effective_betas.loc[1, :]), 
        *tuple(df_effective_betas.loc[5, :])
    )
    df_effective_betas.headers[f'{LUMINOSITY}{IMBALACE}'] = lumi_imb
    df_effective_betas.headers[f'{ERR}{LUMINOSITY}{IMBALACE}'] = lumi_imb_err
    return df_effective_betas


def get_imbalance_w_err(ip1_beta: float, ip1_beta_err: float, ip5_beta: float, ip5_beta_err: float) -> tuple[float, float]:
    """
    Calculate the luminosity imbalance IP1 / IP5  and its error.
    """
    result = ip5_beta / ip1_beta  # due to beta in the denominator for lumi
    err = result * np.sqrt((ip5_beta_err/ip5_beta)**2 + (ip1_beta_err/ip1_beta)**2)
    return result, err


def get_effective_beta_star_w_err(df_ip: tfs.TfsDataFrame) -> tuple[float]:
    """ Calculates the effective beta*, 
    i.e. the denominator of the luminosity (e.g. Eq(17): https://cds.cern.ch/record/941318/files/p361.pdf)
    without any constants, as we only need it for the ratio anyway.  
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


def _get_betastar_xy(df_ip: tfs.TfsDataFrame, beam: int, errors: bool = False) -> tuple[float, float]:
    """ Get betastar x and y for the given beam. """
    if errors:
        return tuple(df_ip.loc[beam, [f'{ERR}{BETASTAR}X', f'{ERR}{BETASTAR}Y']])
    return tuple(df_ip.loc[beam, [f'{BETASTAR}X', f'{BETASTAR}Y']])


def _get_betastar_beams(df_ip: tfs.TfsDataFrame, errors: bool = False) -> tuple[float, float, float, float]:
    """ Get betastar x and y for both beam 1 and beam 2. Order: b1x, b1y, b2x, b2y """
    return (*_get_betastar_xy(df_ip, 1, errors), *_get_betastar_xy(df_ip, 2, errors))


if __name__ == "__main__":
    calculate_lumi_imbalance_entrypoint()