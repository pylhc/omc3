"""
Update Natural Tune in Lin-Files
----------------------------------

Script to update the natural tune in lin files, based on the spectrum data
(amps and freqs) and a given frequency range.


**Arguments:**

*--Required--*

- **files**: List of paths to the spectrum files.
  The files need to be given without their '.lin'/'.amps[xy]','.freqs[xy]' endings.
  (So usually the path of the TbT-Data file.)

- **range** *(float)*: Frequency range in which the highest peak should be found.


*--Optional--*

- **bpms**: List of BPMs which need to be updated. If not given it will be all of them.

- **not_found_action** *(str)*: Defines what to do, if no line was found in given range.
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

  Default: ````

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
from omc3.harpy.constants import FILE_LIN_EXT
from omc3.plotting.spectrum.utils import (load_spectrum_data, get_bpms,
                                          LIN, AMPS, FREQS
                                          )
from omc3.utils.logging_tools import get_logger, list2str

LOG = get_logger(__name__)


# TODO: create constants in measure optics and use these (jdilly)
COL_NATTUNE = "NATTUNE{plane}"
COL_NATAMP = "NATAMP{plane}"
COL_NAME = "NAME"


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
            type=float,
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
            help=('Defines what to do, if no line was found in given range.'
                  "'error': throws a ValueError; 'remove': removes the bpm; "
                  "'ignore': keeps the old values."),
            default='error'
        ),
    )


@entrypoint(get_params(), strict=True)
def main(opt):
    _save_options_to_config(opt)
    LOG.info("Updating Natural Tunes in Lin-Files.")

    for file_path in (Path(f) for f in opt.files):
        data = load_spectrum_data(file_path, opt.bpms, opt.planes)
        bpms = get_bpms(data[LIN], opt.bpms, file_path, opt.planes)

        data = _update_lin_columns(data, bpms,
                                   opt.planes, opt.range, opt.not_found_action,
                                   file_path.name)

        _save_linfiles(data[LIN], file_path, opt.planes, opt.rename_suffix)


def _update_lin_columns(data, bpms, planes, range, not_found_action, filename):
    for plane in planes:
        col_nattune = COL_NATTUNE.format(plane=plane.upper())
        col_natamp = COL_NATAMP.format(plane=plane.upper())

        for bpm in bpms[plane]:
            freqs, amps = data[FREQS][plane][bpm], data[AMPS][plane][bpm]
            peak = _get_peak_in_range(freqs, amps, range)

            if peak is None:
                msg = (f'No lines found for bpm {bpm} in plane {plane} '
                       f'in range {list2str(range)} for file-id "{filename}".')
                if not_found_action == 'error':
                    raise ValueError(msg)
                LOG.warning(msg)

                if not_found_action == 'remove':
                    data[LIN][plane].drop(bpm, axis='index')

            else:
                freq_peak, amp_peak = next(peak.items())
                data[LIN][plane].loc[bpm, col_natamp] = freq_peak
                data[LIN][plane].loc[bpm, col_nattune] = amp_peak
                LOG.debug(f"{filename}.{bpm}.{plane} nattune set to"
                          f"f={freq_peak}, A={amp_peak}")
    return data


def _get_peak_in_range(freqs, amps, range):
    data_series = pd.Series(data=amps.to_numpy(), index=freqs.to_numpy())
    data_series = data_series.sort_index()
    try:
        f_peak = data_series.loc[slice(*sorted(range))].idxmax()
    except ValueError:
        return None
    else:
        return data_series.loc[[f_peak]]


def _save_linfiles(lin_data, file_path, planes, suffix):
    file_path = file_path.with_suffix('')
    for plane in planes:
        out_path = file_path.with_name(file_path.name + suffix).with_suffix(FILE_LIN_EXT.format(plane=plane.lower()))
        tfs.write(str(out_path), lin_data[plane], save_index=COL_NAME)


def _save_options_to_config(opt):
    with suppress(IOError):
        save_options_to_config(formats.get_config_filename(__file__),
                               OrderedDict(sorted(opt.items())))


if __name__ == '__main__':
    main(files=['tests/inputs/spec_test.sdds'], range=[0.2, 0.3], rename_suffix='_mytest')