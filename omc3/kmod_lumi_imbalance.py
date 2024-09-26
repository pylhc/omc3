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
        if isinstance(opt.ip1, (Path, str)):
            save_config(output_path, opt, __file__)

    if isinstance(opt.ip1, (Path, str)):
        df_ip1 = tfs.read(opt.ip1, index=BEAM)
    else:
        df_ip1 = opt.ip1
    if isinstance(opt.ip5, (Path, str)):
        df_ip5 = tfs.read(opt.ip5, index=BEAM)
    else:
        df_ip5 = opt.ip5

    df = get_lumi_imbalance(df_ip1, df_ip5)
    betastar = df_ip1.loc[1, f'{BETASTAR}{MDL}']

    if output_path is not None:
        tfs.write(output_path / f"{EFFECTIVE_BETAS_FILENAME.format(betastar=betastar)}{EXT}", df, save_index=IP)
    
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
    df_effective_betas.loc[1, [f'{BETASTAR}', f'{ERR}{BETASTAR}']] = get_conv_beta_star_w_err(df_ip1)
    df_effective_betas.loc[5, [f'{BETASTAR}', f'{ERR}{BETASTAR}']] = get_conv_beta_star_w_err(df_ip5)
    
    lumi_imb, lumi_imb_err = get_imbalance_w_err(*tuple(df_effective_betas.loc[1, :]), *tuple(df_effective_betas.loc[5, :]))
    df_effective_betas.headers[f'{LUMINOSITY}{IMBALACE}'] = lumi_imb
    df_effective_betas.headers[f'{ERR}{LUMINOSITY}{IMBALACE}'] = lumi_imb_err
    return df_effective_betas


def get_imbalance_w_err(ip1: float, ip1_err: float, ip5: float, ip5_err: float) -> tuple[float, float]:
    """
    Calculate the luminosity imbalance and its error.
    """
    result = ip5 / ip1
    err = result * np.sqrt((ip5_err/ip5)**2 + (ip1_err/ip1)**2)
    return result, err


def get_conv_beta_star_w_err(df_ip: tfs.TfsDataFrame) -> tuple[float]:
    b1x, b1y, b2x, b2y = _get_betastar_beams(df_ip)
    db1x, db2x, db1y, db2y = _get_betastar_beams(df_ip, errors=True) 

    beta = np.sqrt(b1x + b2x) * np.sqrt(b1y + b2y) / 2

    # Error propagation:
    dbeta_dx = np.sqrt(b1y + b2y) / (4 * np.sqrt(b1x + b2x))
    dbeta_dy = np.sqrt(b1y + b2y) / (4 * np.sqrt(b1x + b2x))
    dbeta_dw = np.sqrt(b1x + b2x) / (4 * np.sqrt(b1y + b2y))
    dbeta_dz = np.sqrt(b1x + b2x) / (4 * np.sqrt(b1y + b2y))
    sigma = np.sqrt((dbeta_dx * db1x) ** 2 + (dbeta_dy * db1y) ** 2 +
                    (dbeta_dw * db2x) ** 2 + (dbeta_dz * db2y) ** 2)
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