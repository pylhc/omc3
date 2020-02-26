from collections import OrderedDict
from contextlib import suppress
from pathlib import Path

import numpy as np
from generic_parser import entrypoint, EntryPointParameters
from generic_parser.entrypoint_parser import save_options_to_config

from omc3.definitions import formats
from omc3.harpy.frequency import PLANES  # TODO change to constants
from omc3.plotting.spectrum_utils import load_spectrum_data, LIN, get_bpms
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


def get_params():
    return EntryPointParameters(
        files=dict(
            required=True,
            nargs='+',
            help=("List of paths to the spectrum files. The files need to be given"
                  " without their '.lin'/'.amps[xy]','.freqs[xy]' endings. "
                  " (So usually the path of the TbT-Data file.)")
        ),
        range=dict(
            required=True,
            nargs=2,
            type=np.numerictypes.allTypes,
            help="Frequency range in which the highest peak should be found."
        ),
        bpms=dict(
            nargs='+',
            help=('List of BPMs which need to be updated. '
                  'If not given it will be all of them.')
        ),
        planes=dict(
            nargs='+',
            type=str,
            help="Which planes.",
            choices=PLANES,
            default=PLANES,
        ),
        rename_suffix=dict(
            type=str,
            help=("Additional suffix for output lin-file. "
                  "Will be inserted between filename and extension. "
                  "If empty, the original file is overwritten."),
            default=''
        ),
        not_found_action=dict(
            type=str,
            choices=['error', 'nan', 'remove', 'keep'],
            help=('Defines what to do, if no line was found in given range.'
                  "'error': throws a ValueError; 'nan': sets the value to 'NAN'; "
                  "'remove': removes the bpm; 'keep': keeps the old values."),
            default='error'
        ),
    )


@entrypoint(get_params())
def main(opt):
    _save_options_to_config(opt)
    LOG.info("Updating Natural Tunes in Lin-Files.")

    for file_path in (Path(f) for f in opt.files):
        data = load_spectrum_data(file_path, opt.bpms)
        bpms = get_bpms(data[LIN], opt.bpms, file_path)
        data = _update_lin_columns(data, bpms, opt.range)


def _update_lin_columns(data, bpms, range):
    for plane in PLANES:
        for bpm in bpms[plane]:


def _save_options_to_config(opt):
    with suppress(IOError):
        save_options_to_config(formats.get_config_filename(__file__),
                               OrderedDict(sorted(opt.items())))


