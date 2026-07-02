"""
Parallel
--------

Private helpers for harpy's per-bunch parallel orchestration:

- Resource estimation (peak resident memory per bunch, available RAM, usable cores),
- Worker count strategy,
- BLAS thread-capping utilities.

The functions are kept free of side effects (aside from querying ``psutil`` for host RAM
and cores) so the sizing logic can be unit-tested in isolation. The actual dispatch /
orchestration lives in :func:`omc3.harpy.handler.analyse_bunches`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import psutil
from threadpoolctl import threadpool_limits

from omc3.utils import logging_tools

if TYPE_CHECKING:
    from logging import Logger

    from generic_parser import DotDict

LOGGER: Logger = logging_tools.get_logger(__name__)

# ----- Some constants ----- #

_BYTES_FLOAT64 = 8
_BYTES_COMPLEX128 = 16

# Fraction of available RAM the worker pool is allowed
# to use, with a little bit of margin
SAFETY_MARGIN = 0.9

# Empirically measured floor on the optics1 server (64-core, 536 BPMs, lin-only path):
# Python + NumPy + libraries + one bunch of data + the SVD working set. Overestimating
# is safe (fewer workers, no OOM), so this doubles as the whole footprint of the narrow-mask path.
_BASELINE_RSS_BYTES = int(1.5e9)

# Generous upper bound on the number of frequency lines selected by the narrow
# (resonance-band) mask; its transient is negligible next to the baseline.
_NARROW_MASK_BINS = 1 << 14  # equivalent to 2**14

# ----- System information ----- #


def usable_cores() -> int:
    """
    Tries to return the number of cores actually available to this process.

    Prefers approaches that honour the CPU affinity mask (taskset, cgroup cpuset,
    HPC pinning, ...) so a pinned process does not oversubscribe. Tries, in order:

    - ``os.process_cpu_count`` (Python 3.13+, honours affinity and ``PYTHON_CPU_COUNT``),
    - ``os.sched_getaffinity`` (Linux only, honours affinity),
    - ``psutil.Process().cpu_affinity`` (Windows/Linux only, honours affinity),
    - ``os.cpu_count`` (total logical CPUs, ignores affinity).

    Always returns at least 1.
    """
    if hasattr(os, "process_cpu_count"):  # Python 3.13+, honours affinity + PYTHON_CPU_COUNT
        return os.process_cpu_count() or 1

    try:
        return len(os.sched_getaffinity(0))  # Linux only, honours affinity
    except AttributeError:  # macOS / Windows: no sched_getaffinity
        pass

    try:  # macOS has no affinity concept -> AttributeError
        return len(psutil.Process().cpu_affinity()) or 1  # works on Windows
    except (AttributeError, NotImplementedError, OSError):
        pass
    return os.cpu_count() or 1


def available_ram_bytes() -> int | None:
    """
    Returns currently available RAM in bytes (memory that can be
    given to processes without swapping), via ``psutil``. Returns
    or ``None`` if it cannot be determined.
    """
    try:
        return psutil.virtual_memory().available
    except (OSError, RuntimeError, ValueError):
        return None
