import tfs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from generic_parser import DotDict
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr

LOG = logging_tools.get_logger(__name__)


def kmod_luminosity_params():
    """
    A function to initialize and return parameters for kmod luminosity calculation.
    """
    params = EntryPointParameters()
    params.add_parameter(name="ip1_path",
                         required=True,
                         type=PathOrStr,
                         help="Path to the beta star results of IP1.")
    params.add_parameter(name="ip5_path",
                         required=True,
                         type=PathOrStr,
                         help="Path to the beta star results of IP5.")
    params.add_parameter(name="output_dir",
                         required=True,
                         type=PathOrStr,
                         help="Path to the directory where to write the output files.")
    return params


@entrypoint(kmod_luminosity_params(), strict=True)
def calculate_lumi_imbalance_entrypoint(opt: DotDict) -> None:
    opt.ip1_path = Path(opt.ip1_path)
    opt.ip5_path = Path(opt.ip5_path)
    opt.output_dir = Path(opt.output_dir)
    get_lumi_imbalance(opt)


def get_lumi_imbalance(opt):
    """
    Calculate the effective beta stars and the luminosity imbalance from the input dataframes.
    
    Parameters:
    - opt: the options object containing the paths and output directory
    
    Returns:
    - None
    """
    df_ip1 = tfs.read(opt.ip1_path).set_index('BEAM')
    df_ip5 = tfs.read(opt.ip5_path).set_index('BEAM')
    betastar = df_ip1.loc[1, 'BETSTARMDL']
    ip1, ip1_err = get_conv_beta_star_w_err(df_ip1)
    ip5, ip5_err = get_conv_beta_star_w_err(df_ip5)

    eff_betas = {}
    eff_betas['IP1_EFF'] = ip1
    eff_betas['IP1_EFF_ERR'] = ip1_err
    eff_betas['IP5_EFF'] = ip5
    eff_betas['IP5_EFF_ERR'] = ip5_err
    
    lumi_imb, lumi_imb_err = get_imbalance_w_err(eff_betas['IP5_EFF'], 
                                                 eff_betas['IP1_EFF'], 
                                                 eff_betas['IP5_EFF_ERR'], 
                                                 eff_betas['IP1_EFF_ERR'])
    eff_betas['LUMI_IMB'] = lumi_imb
    eff_betas['LUMI_IMB_ERR'] = lumi_imb_err
    
    eff_betas = pd.DataFrame([eff_betas])
    tfs.write(opt.output_dir / f"effective_betas_{betastar}m.tfs", eff_betas)


def get_imbalance_w_err(ip5, ip1, ip5_err, ip1_err):
    result = ip5 / ip1
    err = result * np.sqrt((ip5_err/ip5)**2 + (ip1_err/ip1)**2)
    return result, err


def get_conv_beta_star(df_ip):
    b1x, b2x, b1y, b2y = df_ip.loc[1, 'BETSTARX'], df_ip.loc[2, 'BETSTARX'], df_ip.loc[1, 'BETSTARY'], df_ip.loc[2, 'BETSTARY']
    return (np.sqrt(b1x+b2x) * np.sqrt(b1y+b2y))/2


def get_conv_beta_star_w_err(df_ip):
    b1x, b2x, b1y, b2y = df_ip.loc[1, 'BETSTARX'], df_ip.loc[2, 'BETSTARX'], df_ip.loc[1, 'BETSTARY'], df_ip.loc[2, 'BETSTARY']
    db1x, db2x, db1y, db2y = df_ip.loc[1, 'ERRBETSTARX'], df_ip.loc[2, 'ERRBETSTARX'], df_ip.loc[1, 'ERRBETSTARY'], df_ip.loc[2, 'ERRBETSTARY']

    beta = np.sqrt(b1x + b2x) * np.sqrt(b1y + b2y) / 2
    dbeta_dx = np.sqrt(b1y + b2y) / (4 * np.sqrt(b1x + b2x))
    dbeta_dy = np.sqrt(b1y + b2y) / (4 * np.sqrt(b1x + b2x))
    dbeta_dw = np.sqrt(b1x + b2x) / (4 * np.sqrt(b1y + b2y))
    dbeta_dz = np.sqrt(b1x + b2x) / (4 * np.sqrt(b1y + b2y))
    sigma = np.sqrt((dbeta_dx * db1x) ** 2 + (dbeta_dy * db1y) ** 2 +
                    (dbeta_dw * db2x) ** 2 + (dbeta_dz * db2y) ** 2)
    return beta, sigma


if __name__ == "__main__":
    calculate_lumi_imbalance_entrypoint()