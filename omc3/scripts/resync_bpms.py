"""
SuperKEKB BPM Synchronization
-----------------------------

This script resyncs the BPMs from the `LER` and `HER` rings of `SuperKEKB`.
Those BPMs are often not aligned time-wise and the values can be off by a few turns.
The resynchronization is done by looking up the phase advance of each BPM to retrieve the turn
offset: a BPM `n` turns off reads a total phase `n * (1 - Q)` (modulo 1) away from the model.
This requires a frequency and an optics analysis of the unsynchronized turn by turn data.

Offsets of up to two turns are fitted. With a tune close to 0.5, offsets `n` and `n +/- 2` are
only `|2Q mod 1|` apart in phase (0.06 for the HER at Qx = 0.53), so a BPM is only shifted when
its phase is more than `tolerance` away from the model and a turn offset explains it to within
twice that. BPMs that no offset explains are left as they are, with a warning.
Running the script a second time on its own output is not a no-op for BPMs with genuine phase
errors close to `tolerance`: run it once, on the raw data.

By default the offsets are read from the kicked plane only, the one with the smallest median phase
error bar, as the phase of the other plane is mostly noise.

The shifted turns at both ends of the data are cropped, so all BPMs keep the same turns.
BPMs that are not in the optics (cleaned, or bad) cannot be checked and are dropped.

The script takes as input the original turn by turn data, the optics directory containing the
results of the optics analysis, as well as the output filename where the turn by turn data will be
written, in SDDS format.


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

- **overwrite** *(bool)*:
    Whether to overwrite the output file if it already exists.

    flags: **['--overwrite']**

    default: ``False``

- **planes** *(str)*:
    Planes to process in the order given, the first one decides for the BPMs present in several.
    Usually `X`, `Y`, `XY` or `YX`. If not given, only the plane with the smallest median phase error bar is used.

    flags: **['--planes']**

    choices: ``('X', 'Y', 'XY', 'YX')``

- **tolerance** *(float)*:
    Phase tolerance, in units of 2pi. A BPM whose phase is within `tolerance` of the model is
    considered synchronized; a turn offset is accepted if it leaves less than twice that.

    flags: **['--tolerance']**

    default: ``0.05``
"""

from __future__ import annotations

from copy import deepcopy
from itertools import permutations
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import tfs
import turn_by_turn as tbt
from generic_parser import EntryPointParameters, entrypoint
from generic_parser.dict_parser import ArgumentError

from omc3.definitions.constants import PLANE_TO_NUM, PLANES
from omc3.optics_measurements.constants import (
    DELTA,
    ERR,
    EXT,
    NAME,
    PHASE,
    TOTAL_PHASE_NAME,
    TUNE,
)
from omc3.optics_measurements.toolbox import ang_diff, ang_interval_check
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr

if TYPE_CHECKING:
    from logging import Logger

LOGGER: Logger = logging_tools.get_logger(__name__)

# ----- Some very specific constants -----

# Available rings
RINGS: Final[set[Literal["LER", "HER"]]] = {"LER", "HER"}

# Phase file containing the phase advance of the BPMs
PHASE_FILE: Final[str] = (
    f"{TOTAL_PHASE_NAME}" + "{plane}" + f"{EXT}"
)  # to be formatted by 'plane'
DEFAULT_DATATYPE: Final = "lhc"

# Turn offsets that can be fitted, the phase cannot tell larger ones apart
TURN_OFFSETS: Final[np.ndarray] = np.array([-2, -1, 1, 2])

# Below the HER's 2-turn step in phase (0.06) and above its arc phase drift (~0.04)
DEFAULT_TOLERANCE: Final[float] = 0.05


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
        planes={
            "type": str,
            "required": False,
            "choices": [  # create all the possible permutations from the planes list
                "".join(p)
                for r in range(1, len(PLANES) + 1)
                for p in permutations(PLANES, r)
            ],
            "help": "Planes to process in the given order. If not given, only the plane with the "
            "smallest median phase error bar is used.",
        },
        tolerance={
            "type": float,
            "required": False,
            "default": DEFAULT_TOLERANCE,
            "help": "Phase tolerance in units of 2pi: BPMs within it are synchronized, "
            "a turn offset is accepted if it leaves less than twice that.",
        },
    )


# ----- Resync Functionality ----- #


def turn_offsets(
    dphase: np.ndarray, tune: float, tolerance: float = DEFAULT_TOLERANCE
) -> tuple[np.ndarray, np.ndarray]:
    """Turn offset `n` of each BPM from its phase error to the model, `dphase = n * (1 - tune)` mod 1.

    Args:
        dphase (np.ndarray): Phase errors of the BPMs, in units of 2pi.
        tune (float): Fractional tune of the plane.
        tolerance (float): Phase errors within it are synchronized BPMs, an offset is accepted
            if it leaves less than twice that.
    Returns:
        The offsets, 0 for the synchronized BPMs and for those no offset explains, and a mask of
        the latter.
    """
    residuals = np.abs(ang_diff(dphase[:, None], TURN_OFFSETS * (1 - tune)))
    in_sync = np.abs(ang_interval_check(dphase)) <= tolerance
    explained = residuals.min(axis=1) <= 2 * tolerance  # False for a NaN phase error
    offsets = TURN_OFFSETS[residuals.argmin(axis=1)]
    return np.where(~in_sync & explained, offsets, 0), ~in_sync & ~explained


def kicked_plane(optics_dir: Path) -> str:
    """The plane with the smallest median phase error bar, the phase of the other one is mostly noise."""

    def median_error(plane: str) -> float:
        phase_df = tfs.read(optics_dir / PHASE_FILE.format(plane=plane.lower()))
        return phase_df[f"{ERR}{DELTA}{PHASE}{plane}"].median()

    return min(PLANES, key=median_error)


def sync_tbt(
    original_tbt: tbt.TbtData,
    optics_dir: Path,
    ring: str,
    planes: str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tbt.TbtData:
    """Resynchronize the BPMs in the turn by turn data based on the phase advance.
    Args:
        original_tbt (tbt.TbtData): Original turn by turn data to be synchronized.
        optics_dir (Path): Original optics directory containing the phase advance files.
        ring (str): Ring name, either `LER` or `HER`.
        planes (str): Planes to look at in the given order, the first one decides for the BPMs
            present in several. Defaults to the kicked plane only, see `kicked_plane`.
        tolerance (float): Phase tolerance in units of 2pi, see `turn_offsets`.
    Returns:
        tbt.TbtData: Resynchronized turn by turn data, cropped to the turns all BPMs have, for
        all bunches.
    """
    # HER and LER are in opposite direction for the phase
    ring_dir = 1 if ring == "HER" else -1

    planes = planes or kicked_plane(optics_dir)

    # BPMs can be in one plane only, the first plane that has a BPM decides for it
    shifts: dict[str, int] = {}
    LOGGER.info(f"Processing planes {planes}")
    for plane in planes:
        phase_df = tfs.read(optics_dir / PHASE_FILE.format(plane=plane.lower()))
        tune = phase_df.headers[f"{TUNE}{PLANE_TO_NUM[plane]}"]
        if tolerance >= abs(ang_interval_check(2 * tune)):
            LOGGER.warning(
                f"  The tolerance {tolerance} is above the two-turn step in phase of plane {plane} "
                f"at Q = {tune:.3f}: BPMs two turns off will be seen as synchronized."
            )
        dphase = phase_df[f"{DELTA}{PHASE}{plane}"].to_numpy()
        offsets, unexplained = turn_offsets(dphase, tune, tolerance)

        for bpm, error, offset, bad in zip(phase_df[NAME], dphase, offsets, unexplained):
            if bpm in shifts:
                continue
            shifts[bpm] = int(offset) * ring_dir
            if bad:
                LOGGER.warning(
                    f"  {bpm:15s} -> no turn offset explains its phase error of {error:.3f}, "
                    "left as is"
                )
            elif offset:
                LOGGER.info(
                    f"  {bpm:15s} -> turn correction of {shifts[bpm]} (phase error {error:.3f})"
                )

    # BPMs cleaned from the optics have an unknown turn offset, they cannot be kept
    unchecked = original_tbt.matrices[0][PLANES[0]].index.difference(list(shifts))
    if len(unchecked):
        LOGGER.warning(
            f"  {len(unchecked)} BPMs are not in the optics, their turn offset cannot be checked, "
            f"dropped: {', '.join(unchecked)}"
        )

    # Shift the data of every bunch, and drop the turns that are not there for all BPMs
    first = max([0, *shifts.values()])
    last = original_tbt.nturns + min([0, *shifts.values()])
    synced_tbt = deepcopy(original_tbt)
    for bunch in synced_tbt.matrices:
        for plane in PLANES:
            matrix = bunch[plane].drop(index=unchecked)
            for bpm, shift in shifts.items():
                if shift:
                    matrix.loc[bpm] = matrix.loc[bpm].shift(shift)
            setattr(bunch, plane, matrix.iloc[:, first:last].set_axis(range(last - first), axis=1))
    synced_tbt.nturns = last - first

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
    synced_tbt = sync_tbt(original_tbt, opt.optics_dir, opt.ring, opt.planes, opt.tolerance)

    # Save the resynced turn by turn data
    opt.output_file.parent.mkdir(exist_ok=True, parents=True)
    tbt.write(opt.output_file, synced_tbt)


if __name__ == "__main__":
    main()
