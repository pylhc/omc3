"""
Parallel
--------

Private helper module for harpy's per-bunch parallel orchestration:

- Resource estimation (peak resident memory per bunch, available RAM, usable cores),
- Dispatched workers count strategy.

The functions are kept free of side effects, aside from querying ``psutil`` for host RAM
and cores. The actual dispatch / orchestration (and the BLAS thread-capping around it) lives
in :func:`omc3.harpy.handler.analyse_bunches_parallel`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import psutil

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
SAFETY_MARGIN = 0.85

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
        return len(os.sched_getaffinity(0))  # Linux only, honours affinity |  # ty:ignore[unresolved-attribute]
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


# ----- Orchestration preparation ----- #


def estimate_peak_rss_bytes_per_bunch(harpy_input: DotDict, n_bpms: int) -> int:
    """
    Analytic estimate of the peak resident set sizememory of a single
    ``run_per_bunch`` call, in bytes. This is the number of bytes we estimate
    one worker will peak at.

    The dominant transient for this is the coefficient array built when calling
    :func:`omc3.harpy.frequency.windowed_padded_rfft` (which runs allocates memory
    via `np.dot(u, s_vt_freq[:, mask])`, plus its `np.abs` copy).

    Its width is the number of selected frequency lines: the full `2 ** turn_bits`
    when `full_spectra` are written (default in the CCC) or cleaning is disabled
    (all-ones mask), and a negligible narrow band otherwise.

    It scales with ``turn_bits`` and ``n_bpms``, not with the number of turns (the
    SVD reduces to ``sing_val`` rows before the zero-padded FFT).

    Args:
        harpy_input (DotDict): Harpy analysis settings from which to query ``turn_bits``,
            ``to_write`` and ``clean``.
        n_bpms (int): number of BPMs (use the pre-clean count as a safe upper bound).

    Returns:
        Estimated peak resident set size (RSS) memory in bytes.
    """
    # Neglect the measurement array length (6600 for ACD, 40 000 for ADT) vs zero padding
    padded: int = 1 << harpy_input.turn_bits  # equivalent to 2 ** turn_bits, frequency bins

    # Determine if we have the extra memory usage from full-spectra
    full: bool = ("full_spectra" in harpy_input.to_write) or (not harpy_input.clean)

    # Witdh, in frequency bins, of the coefficient array dominating the memory peak,
    # i.e. how many rfft bins survive get_freq_mask() into the `coefs` array.
    # - full: the whole half spectrum (2**turn_bits) is materialised, when full_spectra is required.
    # - otherwise: get_freq_mask() keeps only narrow ± tolerance bands around the tunes / resonances,
    #              a few thousand bins. We have set _NARROW_MASK_BINS to safely over-estimate that.
    n_mask: int = padded if full else _NARROW_MASK_BINS

    # At the FFT peak we have two arrays of shape (n_bpms, n_mask) simultaneously:
    # - coefs = np.dot(u, s_vt_freq[:, mask]), a complex128 array
    # - np.abs(coefs), its float64 copy
    transient: int = n_bpms * n_mask * (_BYTES_COMPLEX128 + _BYTES_FLOAT64)

    # Return the estimated peak + minimum memory occupancy (data loading etc)
    return _BASELINE_RSS_BYTES + transient


def decide_n_workers(
    harpy_input: DotDict,
    n_bunches: int,
    n_bpms: int,
    *,
    requested: int = 0,
    margin: float = SAFETY_MARGIN,
) -> int:
    """
    Choose the number of worker processes for per-bunch analysis. Each bunch is sent
    to a worker. This assumes the number of BLAS threads is limited (for `numpy.svd`
    operations for instance where it maxes out the threads and doesn't do much with
    it).

    Note
    ----
        The `requested` argument mirrors the `n_jobs` for the harpy parameters. Passing
        `0` selects automatically (all usable cores, RAM permitting); `1` runs serially
        (which is the old behaviour) and providing an integer value `N` caps the pool at
        either `N` processes or the automatically determined number (RAM permitting).

        The result is `max(1, min(cores_cap, n_bunches, ram_cap))`, where `ram_cap` is
        `floor(margin * available_ram / peak_rss_per_bunch)` (or unconstrained when the
        available RAM could not be determined).

    Args:
        harpy_input (DotDict): Harpy analysis settings, forwarded to
            :func:`estimate_peak_rss_bytes_per_bunch` for the per-bunch RSS estimate.
        n_bunches (int): Number of bunches to process. Caps the pool since each bunch
            goes to at most one worker.
        n_bpms (int): Number of BPMs (use the pre-clean count as a safe upper bound),
            used for the per-bunch RSS estimate.
        requested (int): the `n_jobs` option passed to ``harpy``. See the note above.
        margin (float): fraction of available RAM the worker pool may occupy, leaving
            the remainder as headroom against under-estimating the peak and for other
            running processes.

    Returns:
        The number of worker processes to use, always at least 1.
    """
    # Determine a few parameters
    cores_cap: int = requested if requested > 0 else usable_cores()
    peak_rss: int = estimate_peak_rss_bytes_per_bunch(harpy_input, n_bpms)
    available_ram: int | None = available_ram_bytes()

    # Determine the available RAM to use, including the safety margin
    # If unknown RAM -> do not let it constrain the choice
    if available_ram is None:
        ram_cap: int = n_bunches
        available_ram_str, ram_cap_str = "unknown", "n/a"
    else:
        ram_cap = int(margin * available_ram / peak_rss)
        available_ram_str, ram_cap_str = f"{available_ram / 1e9:.0f}GB", str(ram_cap)

    # Compute the number of workers / jobs to dispatch, log it and return
    n_jobs: int = max(1, min(cores_cap, n_bunches, ram_cap))
    LOGGER.debug(
        f"Parallel harpy: n_jobs={n_jobs} (cores_cap={cores_cap}, bunches={n_bunches}, "
        f"ram_cap={ram_cap_str}, peak/bunch~{peak_rss / 1e9:.1f}GB, available~{available_ram_str})"
    )
    return n_jobs
