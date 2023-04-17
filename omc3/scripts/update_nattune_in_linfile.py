"""
Update Natural Tune in Lin-Files
--------------------------------

Script to update the natural tune in lin files, based on the spectrum data (amps and freqs) and a
given frequency interval.


**Arguments:**

*--Required--*

- **files**: List of paths to the spectrum files.
  The files need to be given without their '.lin'/'.amps[xy]','.freqs[xy]' endings.
  (So usually the path of the TbT-Data file.)

- **interval** *(float)*: Frequency interval in which the highest peak should be found.


*--Optional--*

- **bpms**: List of BPMs which need to be updated. If not given it will be all of them.

- **not_found_action** *(str)*: Defines what to do, if no line was found in given interval.
  'error': throws a ValueError; 'remove': removes the bpm; 'ignore': keeps the old values.

  Choices: ``['error', 'remove', 'ignore']``
  Default: ``error``
- **planes** *(str)*: Which planes.

  Choices: ``('X', 'Y')``
  Default: ``['X', 'Y']``
- **rename_suffix** *(str)*: Additional suffix for output lin-file.
  Will be inserted between filename and extension.
  If empty, the original file is overwritten - unless they are old files,
  then the omc3 filename convention will be used.

  Default: ``None``
"""

from collections import OrderedDict
from contextlib import suppress
from pathlib import Path

import pandas as pd
import tfs
from generic_parser import entrypoint, EntryPointParameters
from generic_parser.entrypoint_parser import save_options_to_config

from omc3.definitions import formats
from omc3.definitions.constants import PLANES
from omc3.harpy.constants import COL_NATTUNE, COL_NATAMP, COL_NAME, FILE_LIN_EXT
from omc3.harpy.handler import _compute_headers
from omc3.plotting.spectrum.utils import (load_spectrum_data, get_bpms,
                                          LIN, AMPS, FREQS
                                          )
from omc3.utils.logging_tools import get_logger, list2str

LOG = get_logger(__name__)


def get_params():
    return EntryPointParameters(
        files=dict(
            required=True,
            nargs='+',
            help=("List of paths to the spectrum files. The files need to be given"
                  " without their '.lin'/'.amps[xy]','.freqs[xy]' endings. "
                  " (So usually the path of the TbT-Data file.)")
        ),
        interval=dict(
            required=True,
            nargs=2,
            type=float,
            help="Frequency interval in which the highest peak should be found."
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
            default=list(PLANES),
        ),
        rename_suffix=dict(
            type=str,
            help=("Additional suffix for output lin-file. "
                  "Will be inserted between filename and extension. "
                  "If empty, the original file is overwritten - unless they "
                  "are old files, then the omc3 filename convention will be "
                  "used."),
            default=''
        ),
        not_found_action=dict(
            type=str,
            choices=['error', 'remove', 'ignore'],
            help=('Defines what to do, if no line was found in given interval.'
                  "'error': throws a ValueError; 'remove': removes the bpm; "
                  "'ignore': keeps the old values."),
            default='error'
        ),
    )


@entrypoint(get_params(), strict=True)
def main(opt):
    _save_options_to_config(opt)
    LOG.info("Updating Natural Tunes in Lin-Files.")

    gathered = [None] * len(opt.files)
    for idx_file, file_path in enumerate(Path(f) for f in opt.files):
        data = load_spectrum_data(file_path, opt.bpms, opt.planes)
        bpms = get_bpms(data[LIN], opt.bpms, file_path, opt.planes)

        data = _update_lin_columns(data, bpms,
                                   opt.planes, opt.interval, opt.not_found_action,
                                   file_path.name)

        data = _update_lin_header(data, opt.planes, file_path.name)

        _save_linfiles(data[LIN], file_path, opt.planes, opt.rename_suffix)
        gathered[idx_file] = data[LIN]
    return gathered


# Update -----------------------------------------------------------------------


def _update_lin_columns(data, bpms, planes, interval, not_found_action, filename):
    for plane in planes:
        col_nattune = f'{COL_NATTUNE}{plane.upper()}'
        col_natamp = f'{COL_NATAMP}{plane.upper()}'

        for bpm in bpms[plane]:
            freqs, amps = data[FREQS][plane][bpm], data[AMPS][plane][bpm]
            peak = _get_peak_in_interval(freqs, amps, interval)

            if peak is None:
                msg = (f'No lines found for bpm {bpm} in plane {plane} '
                       f'in interval {list2str(interval)} for file-id "{filename}".')
                if not_found_action == 'error':
                    raise ValueError(msg)
                LOG.warning(msg)

                if not_found_action == 'remove':
                    data[LIN][plane] = data[LIN][plane].drop(bpm, axis='index')

            else:
                freq_peak, amp_peak = peak
                data[LIN][plane].loc[bpm, col_nattune] = freq_peak
                data[LIN][plane].loc[bpm, col_natamp] = amp_peak
                LOG.debug(f"{filename}.{bpm}.{plane} nattune set to"
                          f"f={freq_peak}, A={amp_peak}")
    return data


def _get_peak_in_interval(freqs, amps, interval):
    data_series = pd.Series(data=amps.to_numpy(), index=freqs.to_numpy())
    data_series = data_series.sort_index()
    try:
        f_peak = data_series.loc[slice(*sorted(interval))].idxmax()
    except ValueError:
        return None
    else:
        return f_peak, data_series.loc[f_peak]


def _update_lin_header(data, planes, filename):
    for plane in planes:
        LOG.debug(f"Updating headers for {filename}, plane {plane}")
        data[LIN][plane].headers.update(_compute_headers(data[LIN][plane]))
    return data


# Output -----------------------------------------------------------------------


def _save_linfiles(lin_data, file_path, planes, id_suffix):
    for plane in planes:
        out_path = file_path.with_name(f'{file_path.name}{id_suffix}{FILE_LIN_EXT.format(plane=plane.lower())}')
        tfs.write(str(out_path), lin_data[plane], save_index=COL_NAME)


def _save_options_to_config(opt):
    with suppress(IOError):
        save_options_to_config(formats.get_config_filename(__file__),
                               OrderedDict(sorted(opt.items())))


# Script Mode ------------------------------------------------------------------


if __name__ == '__main__':
    main()
