import tfs
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from generic_parser import DotDict
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint
from omc3.model import manager
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config

LOG = logging_tools.get_logger(__name__)


def kmod_output_params():
    params = EntryPointParameters()
    params.add_parameter(name="meas_paths",
                         required=True,
                         nargs='+',
                         help="Kmod results files to import")
    params.add_parameter(name="model",
                         required=True,
                         type=PathOrStr,
                         help="Path to the model.", )
    params.add_parameter(name="output_dir",
                         required=True,
                         type=PathOrStr,
                         help="Path to the directory where to write the output files.", )
    return params


@entrypoint(kmod_output_params())
def input_kmod_for_global_entrypoint(opt: DotDict, accel_opt) -> None:

    "Input kmod results for global corrections"
    LOG.info("Starting Kmod import for global correction.")

    tw = tfs.read(opt.model)

    kmod_list = []
    for ip_path in opt.meas_paths:
        kmod_list.append(tfs.read(ip_path))

    kmod_results = pd.concat(kmod_list)

    common_bpms = np.intersect1d(kmod_results['NAME'], tw['NAME'])
    kmod_results = kmod_results.set_index('NAME')
    kmod_results = kmod_results.loc[common_bpms]
    tw = tw.set_index('NAME')

    print(kmod_results)
    print(common_bpms)

    beta_kmod_x = kmod_results[['S', 'BETX', 'ERRBETX']]
    beta_kmod_y = kmod_results[['S', 'BETY', 'ERRBETY']]

    beta_kmod_x['BETXMDL'] = tw.loc[common_bpms, 'BETX']
    beta_kmod_y['BETYMDL'] = tw.loc[common_bpms, 'BETY']

    beta_kmod_x['DELTABETXMDL'] = beta_kmod_x['BETX'] - beta_kmod_x['BETXMDL']
    beta_kmod_y['DELTABETYMDL'] = beta_kmod_y['BETY'] - beta_kmod_y['BETYMDL']

    beta_kmod_x = beta_kmod_x.reset_index()
    beta_kmod_y = beta_kmod_y.reset_index()
    tfs.write(opt.output_dir / 'beta_kmod_x.tfs', beta_kmod_x)
    tfs.write(opt.output_dir / 'beta_kmod_y.tfs', beta_kmod_y)



if __name__ == "__main__":
    input_kmod_for_global_entrypoint()


