""" 
Import K-Modulation Results
---------------------------

Imports K-Mod data and writes them into a file containing beta data, 
in the same format as beta-from-phase or beta-from-amplitude.
This data can then be easily used for the same purposes, e.g. global correction.

**Arguments:**

*--Required--*

- **meas_paths** *(PathOrStr)*:

    Kmod results files to import.


- **model** *(PathOrStr)*:

    Path to the model.


*--Optional--*

- **output_dir** *(PathOrStr)*:

    Path to the directory where to write the output files.

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint

from omc3.definitions.constants import PLANES
from omc3.kmod.constants import BETA_FILENAME, EXT
from omc3.optics_measurements.constants import (  # using opitcs constants as we create similar file
    BETA,
    DELTA,
    ERR,
    MDL,
    NAME,
    TUNE,
    S,
)
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr

if TYPE_CHECKING:
    from generic_parser import DotDict

LOG = logging_tools.get_logger(__name__)


def _get_params():
    """
    Creates and returns the parameters for the kmod_output function.
    
    """
    params = EntryPointParameters()
    params.add_parameter(name="meas_paths",
                         required=True,
                         nargs='+',
                         type=PathOrStr,
                         help="Kmod results files to import.")
    params.add_parameter(name="model",
                         required=True,
                         type=PathOrStr,
                         help="Path to the model.")
    params.add_parameter(name="output_dir",
                         type=PathOrStr,
                         help="Path to the directory where to write the output files.")
    return params


@entrypoint(_get_params(), strict=True)
def import_kmod_data(opt: DotDict) -> None:
    """
    Reads model and measurement files to calculate differences in beta functions 
    and writes the results to output files.
    
    Args:
        meas_paths (Sequence[Path|str]):
            A sequence of kmod BPM results files to import. This can include either single 
            measurements (e.g., 'lsa_results.tfs') or averaged results 
            (e.g., 'averaged_bpm_beam1_ip1_beta0.22m.tfs').
        
        model (Path|str):
            Path to the model Twiss file.
        
        output_dir (Path|str):
            Path to the output directory, i.e. the optics-measurement directory 
            into which to import these K-Modulation results.
    
    Returns:
        Dictionary of kmod-DataFrames by planes.
    """
    LOG.info("Starting Kmod import for global correction.")

    # Prepare output dir    
    if opt.output_dir is not None:
        opt.output_dir = Path(opt.output_dir)
        opt.output_dir.mkdir(exist_ok=True)

    # read data
    df_model = tfs.read(opt.model, index=NAME)
    kmod_list = [tfs.read(ip_path, index=NAME) for ip_path in opt.meas_paths]

    # get common bpms
    kmod_results = tfs.concat(kmod_list, join='inner', )
    common_bpms = kmod_results.index.intersection(df_model.index)
    kmod_results = kmod_results.loc[common_bpms, :]
    df_model = df_model.loc[common_bpms, :]

    # create new dataframes
    beta_kmod = {}
    for plane in PLANES:
        beta_kmod = tfs.TfsDataFrame(index=df_model.index)
        
        # copy s, name and beta
        beta_kmod.loc[:, S] = df_model.loc[:, S]
        beta_kmod.loc[:, f'{BETA}{plane}{MDL}'] = df_model.loc[:, f'{BETA}{plane}']
        beta_kmod.loc[:, f'{BETA}{plane}'] = kmod_results[f'{BETA}{plane}']
        beta_kmod.loc[:, f'{ERR}{BETA}{plane}'] = kmod_results[f'{ERR}{BETA}{plane}']

        # model-delta and beta-beating
        beta_kmod.loc[:, f'{DELTA}{BETA}{plane}{MDL}'] = beta_kmod[f'{BETA}{plane}'] - beta_kmod[f'{BETA}{plane}{MDL}']
        beta_kmod.loc[:, f'{DELTA}{BETA}{plane}'] = beta_kmod[f'{DELTA}{BETA}{plane}{MDL}'] / beta_kmod[f'{BETA}{plane}{MDL}']
        beta_kmod.loc[:, f'{ERR}{DELTA}{BETA}{plane}'] = beta_kmod[f'{ERR}{BETA}{plane}'] / beta_kmod[f'{BETA}{plane}{MDL}']

        # tune
        beta_kmod.headers[f'{TUNE}1'] = df_model.headers[f'{TUNE}1'] % 1
        beta_kmod.headers[f'{TUNE}2'] = df_model.headers[f'{TUNE}2'] % 1

        beta_kmod = beta_kmod.sort_values(by=S)
        
        if opt.output_dir is not None:
            tfs.write(opt.output_dir / f'{BETA_FILENAME}{plane.lower()}{EXT}', beta_kmod, save_index=NAME)


if __name__ == "__main__":
    import_kmod_data()
