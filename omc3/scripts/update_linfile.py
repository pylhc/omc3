from collections import OrderedDict
from contextlib import suppress

from generic_parser import entrypoint, EntryPointParameters
from generic_parser.entrypoint_parser import save_options_to_config

from definitions import formats
from omc3.harpy.frequency import PLANES
from omc3.plotting.spectrum_utils import load_spectrum_data
import numpy as np
from omc3.utils import logging_tools
from pathlib import Path


LOG = logging_tools.get_logger(__name__)


def get_params():
    return EntryPointParameters(
        files=dict(required=True,
                   nargs='+',
                   help=("List of paths to the spectrum files. The files need to be given"
                         " without their '.lin'/'.amps[xy]','.freqs[xy]' endings. "
                         " (So usually the path of the TbT-Data file.)")
                   ),
        range=dict(required=True,
                   nargs=2,
                   type=np.numerictypes.allTypes,
                   ),
        bpms=dict(nargs='+',
                  help='List of BPMs which need to be updated. If not given it will be all of them.'),
        rename_suffix=dict(type=str,
                           help=("Additional suffix for output lin-file. "
                                 "Will be inserted between filename and extension. "
                                 "If not, the original file is overwritten."),
                           default=''
                           ),
        target=dict(
            type=str,
            help="Target column of the lin-file to be updated by found peak.",
            choices=["TUNE", "NATTUNE"],
            default="NATTUNE",
        ),
        planes=dict(
            nargs='+',
            type=str,
            help="Which planes.",
            choices=PLANES,
            default=PLANES,
        ),
    )


@entrypoint(get_params())
def main(opt):
    _save_options_to_config(opt)
    LOG.info("Updating Linfiles.")

    for file_path in (Path(f) for f in opt.files):
        data = load_spectrum_data(file_path, opt.bpms)




def _save_options_to_config(opt):
    with suppress(IOError):
        save_options_to_config(formats.get_config_filename(__file__),
                               OrderedDict(sorted(opt.items())))


