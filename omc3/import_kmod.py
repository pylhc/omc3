import tfs
import numpy as np
import pandas as pd
from pathlib import Path

from generic_parser import DotDict
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr
from omc3.definitions.constants import PLANES

LOG = logging_tools.get_logger(__name__)


def kmod_output_params():
    """
    A function to create and return the parameters for the kmod_output function.
    """
    params = EntryPointParameters()
    params.add_parameter(name="meas_paths",
                         required=True,
                         nargs='+',
                         help="Kmod results files to import")
    params.add_parameter(name="model",
                         required=True,
                         type=PathOrStr,
                         help="Path to the model.")
    params.add_parameter(name="output_dir",
                         required=True,
                         type=PathOrStr,
                         help="Path to the directory where to write the output files.")
    return params


@entrypoint(kmod_output_params(), strict=True)
def input_kmod_for_global_entrypoint(opt: DotDict) -> None:
    LOG.info("Starting Kmod import for global correction.")
    
    opt.meas_paths = [Path(m) for m in opt.meas_paths]
    opt.output_dir = Path(opt.output_dir)
    opt.output_dir.mkdir(exist_ok=True)
    import_kmod(opt)


def import_kmod(opt):
    """
    Reads model and measurement files to calculate differences in beta functions and writes the results to output files.
    
    Parameters:
    opt (object): An object containing the model and measurement file paths, and the output directory.
    
    Returns:
    None
    """
    tw = tfs.read(opt.model)

    kmod_list = []
    for ip_path in opt.meas_paths:
        kmod_list.append(tfs.read(ip_path))

    kmod_results = pd.concat(kmod_list)
    
    common_bpms = np.intersect1d(kmod_results['NAME'], tw['NAME'])
    kmod_results = kmod_results.set_index('NAME')
    kmod_results = kmod_results.loc[common_bpms]
    tw = tw.set_index('NAME')

    for plane in PLANES:
        beta_kmod = kmod_results[[f'BET{plane}', f'ERRBET{plane}']]
        beta_kmod[f'S'] = tw.loc[common_bpms, 'S']
        beta_kmod[f'BET{plane}MDL'] = tw.loc[common_bpms, f'BET{plane}']
        beta_kmod[f'DELTABET{plane}MDL'] = beta_kmod[f'BET{plane}'] - beta_kmod[f'BET{plane}MDL']
        beta_kmod[f'DELTABET{plane}'] = beta_kmod[f'DELTABET{plane}MDL']/beta_kmod[f'BET{plane}MDL']
        beta_kmod[f'ERRDELTABET{plane}'] = beta_kmod[f'ERRBET{plane}']/beta_kmod[f'BET{plane}MDL']
        beta_kmod = beta_kmod.reset_index()
        beta_kmod.headers['Q1'] = tw.headers['Q1']%1
        beta_kmod.headers['Q2'] = tw.headers['Q2']%1
        beta_kmod = beta_kmod.sort_values(by='S')
        tfs.write(opt.output_dir / f'beta_kmod_{plane.lower()}.tfs', beta_kmod)



if __name__ == "__main__":
    input_kmod_for_global_entrypoint()


