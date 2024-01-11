"""
Check Peak-to-Peak
------------------

Performs a quick check on the peak-to-peak of the given
turn-by-turn files.
"""
import numbers
import re
from pathlib import Path
from typing import Union, Sequence, Dict

import numpy as np
import pandas as pd
import turn_by_turn as tbt
from generic_parser import EntryPointParameters, entrypoint
from turn_by_turn.utils import generate_average_tbtdata

from omc3.definitions.constants import PLANES, UNIT_IN_METERS
from omc3.harpy.constants import DEFOCUSSING_MONITORS
from omc3.hole_in_one import HARPY_DEFAULTS
from omc3.tbt_converter import _file_name_without_sdds
from omc3.utils.logging_tools import get_logger

LOG = get_logger(__name__)


def get_params():
    params = EntryPointParameters()
    params.add_parameter(name="files", required=True, nargs='+',
                         help="Files for analysis")
    params.add_parameter(name="beam", type=int,
                         help="LHC beam number.")
    params.add_parameter(name="tbt_datatype",
                         default=HARPY_DEFAULTS["tbt_datatype"],
                         choices=list(tbt.io.DATA_READERS.keys()),
                         help="Choose the datatype from which to import.")
    params.add_parameter(name="unit", type=str, default="mm",
                         choices=list(UNIT_IN_METERS.keys()),
                         help="Unit in which to log the peak-to-peak values.")
    params.add_parameter(name="input_unit", type=str, default="mm",
                         choices=list(UNIT_IN_METERS.keys()),
                         help="Unit of the tbt input data.")
    return params


@entrypoint(get_params(), strict=True)
def peak_to_peak(opt):
    """Main function to log peak-to-peak values from SDDS files.

    *--Required--*

    - **files**:

        Files for analysis


    *--Optional--*

    - **beam** *(int)*:

        LHC beam number.


    - **input_unit** *(str)*:

        Unit of the tbt input data.

        choices: ``['km', 'm', 'mm', 'um', 'nm', 'pm', 'fm', 'am']``

        default: ``mm``


    - **tbt_datatype**:

        Choose the datatype from which to import.

        choices: ``['lhc', 'iota', 'esrf', 'ptc', 'trackone']``

        default: ``lhc``


    - **unit** *(str)*:

        Unit in which to log the peak-to-peak values.

        choices: ``['km', 'm', 'mm', 'um', 'nm', 'pm', 'fm', 'am']``

        default: ``mm``

    """
    for input_file in opt.files:
        LOG.debug(f"Calculating pk2pk for file: {input_file}")
        input_file = Path(input_file)
        name = _file_name_without_sdds(input_file)
        beam = _get_beam(opt.beam, filename=name)
        bpms = DEFOCUSSING_MONITORS[beam]
        tbt_data = tbt.read_tbt(input_file, datatype=opt.tbt_datatype)
        for bunch, positions in zip(tbt_data.bunch_ids, tbt_data.matrices):
            LOG.debug(f"Bunch: {bunch}")
            pk2pk = get_pk2pk(positions, bpms)
            _log_pk2pk(pk2pk, name, bunch, opt.input_unit, opt.unit)

        if tbt_data.nbunches > 1:
            tbt_data_av = generate_average_tbtdata(tbt_data)
            positions = tbt_data_av.matrices[0]
            pk2pk = get_pk2pk(positions, bpms)
            _log_pk2pk(pk2pk, name, '(average)', opt.input_unit, opt.unit)


def get_pk2pk(data: tbt.TransverseData, bpms: Dict[str, Sequence[str]]) -> Dict[str, float]:
    """Get the filtered peak-to-peak from the current tbt-data from the given bpms."""
    pk2pk = {p: None for p in PLANES}
    for plane in PLANES:
        positions: pd.DataFrame = getattr(data, plane)

        # Get only the desired BPMs
        mask = _get_index_mask(positions.index, bpms[plane])
        if not any(mask):
            LOG.debug(f"None of the required pk2pk BPMs are present "
                      f"in plane {plane} of the current tbt data.")
            continue
        positions = positions.loc[mask, :]

        # Filter BPMs with exact zeros
        exact_zeros = get_exact_zero_mask(positions)
        if all(exact_zeros):
            LOG.debug(f"Exact zeros found in all BPMs "
                      f"in plane {plane} of the current tbt data.")
            continue
        positions = positions.loc[~exact_zeros, :]

        # Calculate peak-to-peak from remaining
        pk2pk_per_bpm = positions.max(axis='columns') - positions.min(axis='columns')
        pk2pk[plane] = pk2pk_per_bpm.mean()
    return pk2pk


def _log_pk2pk(pk2pk: Dict[str, float], name: str = None, other: Union[int, str] = None, input_unit: str = "m", unit: str = "mm"):
    other_str = other if isinstance(other, str) else ""
    if isinstance(other, numbers.Integral):
        other_str = f", Bunch {other: d}"

    unit_scale = UNIT_IN_METERS[input_unit] / UNIT_IN_METERS[unit]
    pk2pk_str = " ".join(f"   {plane}: {p2p*unit_scale:.4f} {unit}" for plane, p2p in pk2pk.items() if p2p is not None)
    pk2pk_warn = ", ".join(plane for plane, p2p in pk2pk.items() if p2p is None)

    LOG.info(f"Peak-to-Peak values for {name}{other_str}: {pk2pk_str} ")
    if pk2pk_warn:
        LOG.warning(f"No Peak-to-Peak values for {name} {other_str} in planes {pk2pk_warn}")


# Utils ------------------------------------------------------------------------

def _get_beam(beam: int, filename: Union[Path, str]) -> int:
    """Get the beam from either given option or try to get it from the filename."""
    if beam is None:
        try:
            beam = infer_beam_from_filename(filename)
        except AttributeError:
            raise NameError(f"No beam option given and could not infer beam from filename {filename}. "
                            f"Please provide a beam number. ")
    LOG.debug(f"Assuming Beam {beam}")
    return beam


def _get_index_mask(index: pd.Index, bpms: Sequence[str]) -> Sequence[bool]:
    """Get the boolean mask for the index/columns of the desired bpms."""
    mask = np.zeros_like(index, dtype=bool)
    not_found_bpms = []
    for bpm in bpms:
        new_mask = index.str.startswith(bpm)
        if any(new_mask):
            mask |= new_mask
        else:
            not_found_bpms.append(bpm)

    if len(not_found_bpms):
        LOG.debug(f"Some BPMs were not found in current tbt data: {not_found_bpms}")
    return mask


def get_exact_zero_mask(positions: pd.DataFrame):
    """Finds bpms containing exact zeros in the data."""
    exact_zeros = (positions == 0).any(axis='columns')
    if any(exact_zeros):
        LOG.debug(f"Exact zeros found in bpms {positions.columns[exact_zeros]}")
    return exact_zeros


def infer_beam_from_filename(filename: Union[str, Path]) -> int:
    """Regex to find 'beam\\d' in filename and return the beam."""
    return int(re.search(r"Beam(\d)", str(filename), flags=re.IGNORECASE).group(1))


# Script Mode ------------------------------------------------------------------

if __name__ == '__main__':
    peak_to_peak()