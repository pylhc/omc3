"""
SuperKEKB BPM Synchronization
-----------------------------

This script resyncs the BPMs from the `LER` and `HER` rings of `SuperKEKB`.
Those BPMs are often not aligned time-wise and the values can be off by a few turns.
The resynchronization is done by looking up the phase advance of each BPM to retrieve the turn
offset.
This requires a frequency and an optics analysis of the unsynchronized turn by turn data.

The script takes as input the original turn by turn data, the optics directory containing the
results of the optics analysis, as well as the output filename where the turn by turn data will be
written, in ASCII SDDS format.


Arguments:

*--Required--*

- **input** *(Path,str,TbtData)*:

    Input turn by turn data to be resynchronized.
    Can take the form of a `Path` or `str` to a file or directly a `TbtData` object.

    flags: **['--input']**

- **optics_dir** *(Path,str)*:

    Optics analysis path of the unsynchronized data, must contain the `total_phase_{x,y}.tfs` files.

    flags: **['--optics_dir']**

- **output_file** *(Path,str)*:

    Output file path where to write the synchronized turn by turn data.
    The directory will be created if necessary.

    flags: **['--output_file']**

- **ring** *(str)*:

    Ring name, either `LER` or `HER`.

    flags: **['--ring']**

    choices: ``('LER', 'HER')``

*--Optional--*

- **tbt_datatype** *(str)*:
    Datatype of the `turn_by_turn` data provided as input.

    flags: **['--tbt_datatype']**

    choices: list of datatypes supported by `turn_by_turn`, in `turn_by_turn.io.TBT_MODULES`

    default: ``lhc``

- ** overwrite** *(bool)*:
    Whether to overwrite the output file if it already exists.

    flags: **['--overwrite']**

    default: ``False``
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import tfs
import turn_by_turn as tbt
from generic_parser import EntryPointParameters, entrypoint
from generic_parser.dict_parser import ArgumentError

from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import DELTA, EXT, NAME, PHASE, TOTAL_PHASE_NAME, TUNE
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr

if TYPE_CHECKING:
    from pandas import Series

LOGGER = logging_tools.get_logger(__name__)

# ----- Some very specific constants -----

# Available rings
RINGS: Final[set[Literal["LER", "HER"]]] = {"LER", "HER"}

# Phase file containing the phase advance of the BPMs
PHASE_FILE: Final[str] = f"{TOTAL_PHASE_NAME}" + "{plane}" + f"{EXT}"  # to be formatted by 'plane'
DEFAULT_DATATYPE: Final[Literal["lhc"]] = "lhc"


# ----- Entrypoint parsing ----- #


def _get_params() -> dict:
    """
    Parse Commandline Arguments and return them as options.

    Returns:
        dict
    """

    return EntryPointParameters(
        input={
            "required": True,
            "help": "Input turn by turn data to be resynchronized. Can take the form of a `Path` to"
            "a file or directly a `TbtData` object.",
        },
        optics_dir={
            "type": PathOrStr,
            "required": True,
            "help": "Optics path, must contain the `total_phase_{x,y}.tfs` files.",
        },
        output_file={
            "type": PathOrStr,
            "required": True,
            "help": "Output file path where to write the turn by turn data. The directory will be"
            "created if necessary.",
        },
        ring={
            "type": str,
            "required": True,
            "choices": RINGS,
            "help": (f"Ring name, from {RINGS}"),
        },
        tbt_datatype={
            "type": str,
            "required": False,
            "choices": list(tbt.io.TBT_MODULES.keys()),
            "default": DEFAULT_DATATYPE,
            "help": "Datatype of the TurnByTurn data",
        },
        overwrite={
            "type": bool,
            "required": False,
            "default": False,
            "help": "Whether to overwrite the output file if it already exists.",
        },
    )


# ----- Resync Functionality ----- #


def sync_tbt(original_tbt: tbt.TbtData, optics_dir: Path, ring: str) -> tbt.TbtData:
    """Resynchronize the BPMS in the the turn by turn data based on the phase advance.
    Args:
        original_tbt (tbt.TbtData): Original turn by turn data to be synchronized.
        optics_dir (Path): Original optics directory containing the phase advance files.
        ring (str): Ring name, either `LER` or `HER`.
    Returns:
        tbt.TbtData: Resynchronized turn by turn data.
    """
    # Copy the original turn by turn data
    synced_tbt = deepcopy(original_tbt)

    # HER and LER are in opposite direction for the phase
    ring_dir = 1 if ring == "HER" else -1

    # Some BPMs can exist in a plane but not the other, we need to check both planes to be sure
    already_processed = set()
    for plane in PLANES:
        phase_df = tfs.read(optics_dir / PHASE_FILE.format(plane=plane.lower()))
        qx = phase_df.headers[f"{TUNE}1"]
        qy = phase_df.headers[f"{TUNE}2"]
        bpms: Series = phase_df[NAME]  # using omc3 constants
        dphase: Series = phase_df[f"{DELTA}{PHASE}{plane.upper()}"]

        tune = (1 - qx) if plane == "X" else (1 - qy)

        # The phase advance divided by the tune will tell us how off the BPM is
        ntune: Series = dphase / tune
        abs_ntune: Series = ntune.abs()

        # If the ratio ntune is close to 1, that's one turn, otherwise, it's likely -2 turns
        mag: np.ndarray = np.select([abs_ntune >= 0.8, abs_ntune >= 0.1], [1, -2], default=0)
        # The final number of turns also depends on the sign of the phase
        final_correction: np.ndarray = (mag * np.sign(ntune) * ring_dir).astype(int)

        # Iterate through all the BPMs and check their phase advance
        for idx, bpm in enumerate(bpms):
            # Check if we've seen that BPM before in the other plane
            if bpm in already_processed:
                continue
            already_processed.add(bpm)

            if (bpm_correction := final_correction[idx]) == 0:
                continue

            # Shift the data
            LOGGER.info(
                f"  {bpm:15s} -> turn correction of {bpm_correction} (ntune={ntune[idx]:.2f})"
            )
            for plane in PLANES:
                matrix = synced_tbt.matrices[0][plane]
                orig_row = original_tbt.matrices[0][plane].loc[bpm]
                matrix.loc[bpm] = orig_row.shift(bpm_correction, fill_value=0)

    return synced_tbt


# ----- CLI Mode ----- #


@entrypoint(_get_params(), strict=True)
def main(opt):
    # Open the TbT file if needed
    if isinstance(opt.input, (Path, str)):
        original_tbt = tbt.read(opt.input, datatype=opt.tbt_datatype)
    elif isinstance(opt.input, tbt.TbtData):
        original_tbt = opt.input
    else:
        raise ArgumentError("input must be either a Path, str or a TbtData object")

    opt.optics_dir = Path(opt.optics_dir)
    opt.output_file = Path(opt.output_file)

    # Check the overwrite flag
    if (opt.output_file).exists() and not opt.overwrite:
        LOGGER.warning(f"File {opt.output_file} already exists, aborting.")
        raise FileExistsError(f"File {opt.output_file} already exists, aborting.")
    if (opt.output_file).exists() and opt.overwrite:
        LOGGER.warning(f"Overwriting file {opt.output_file}.")

    # Synchronise TbT
    LOGGER.info(f"Resynchronizing {opt.optics_dir.name}...")
    synced_tbt = sync_tbt(original_tbt, opt.optics_dir, opt.ring)  # type: ignore

    # Save the resynced turn by turn data
    opt.output_file.parent.mkdir(exist_ok=True, parents=True)
    tbt.write(opt.output_file, synced_tbt)


if __name__ == "__main__":
    main()
