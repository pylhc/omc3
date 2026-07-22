"""
Unit tests for the harpy parallel orchestration strategy in omc3.harpy._parallel.
"""

import logging
import os
from types import SimpleNamespace

import pytest
from generic_parser import DotDict

from omc3.harpy import _parallel


def _min_harpy_input(turn_bits=20, to_write=("lin",), clean=True) -> DotDict:
    """Just the ones we query from harpy input."""
    return DotDict(turn_bits=turn_bits, to_write=list(to_write), clean=clean)



# ----- Detection of Available RAM ----- #


@pytest.mark.basic
def test_available_ram_bytes_returns_none_on_error(monkeypatch):
    """
    If psutil cannot read memory, check that available_ram_bytes()
    reports None (and RAM remains unconstrained).
    """
    def _raise():
        raise OSError

    monkeypatch.setattr(_parallel.psutil, "virtual_memory", _raise)
    assert _parallel.available_ram_bytes() is None


# ----- Detection of Usable Cores ----- #

# Our coverage CI runs on Python 3.14+, where usable_cores() returns early inside of
# os.process_cpu_count(). These tests remove the attribute to force (and cover) each
# fallback: os.sched_getaffinity -> psutil affinity -> os.cpu_count.


@pytest.mark.basic
def test_usable_cores_via_sched_getaffinity(monkeypatch):
    """
    Without os.process_cpu_count(), the Linux os.sched_getaffinity path is used.
    """
    monkeypatch.delattr(os, "process_cpu_count", raising=False)  # simulate lower Python
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2}, raising=False)
    assert _parallel.usable_cores() == 3


@pytest.mark.basic
def test_usable_cores_via_psutil_affinity(monkeypatch):
    """
    Without process_cpu_count() and sched_getaffinity(), psutil's affinity is used.
    """
    monkeypatch.delattr(os, "process_cpu_count", raising=False)  # simulate lower Python
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)  # simulate non-Linux
    monkeypatch.setattr(
        _parallel.psutil, "Process", lambda: SimpleNamespace(cpu_affinity=lambda: [0, 1, 2, 3])
    )
    assert _parallel.usable_cores() == 4


@pytest.mark.basic
def test_usable_cores_falls_back_to_cpu_count(monkeypatch):
    """
    When no affinity source is available, fall back to os.cpu_count().
    """
    def _raise():
        raise NotImplementedError

    monkeypatch.delattr(os, "process_cpu_count", raising=False)  # simulate lower Python
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)  # simulate non-Linux
    monkeypatch.setattr(_parallel.psutil, "Process", lambda: SimpleNamespace(cpu_affinity=_raise))
    monkeypatch.setattr(os, "cpu_count", lambda: 7)  # patch as well as we can't predict the CI machine
    assert _parallel.usable_cores() == 7


# ----- Estimation of Peak RSS per Bunch ----- #


@pytest.mark.basic
def test_peak_rss_scales_linearly_with_bpms():
    base: int = _parallel._BASELINE_RSS_BYTES
    hin: DotDict = _min_harpy_input(to_write=["lin", "full_spectra"])
    p500: int = _parallel.estimate_peak_rss_bytes_per_bunch(hin, 500)
    p1000: int = _parallel.estimate_peak_rss_bytes_per_bunch(hin, 1000)
    p2000: int = _parallel.estimate_peak_rss_bytes_per_bunch(hin, 2000)
    # The transient (everything above the constant baseline) scales with n_bpms.
    assert (p1000 - base) == 2 * (p500 - base)
    assert (p2000 - base) == 4 * (p500 - base)


@pytest.mark.basic
def test_peak_rss_doubles_with_turn_bits():
    """Increasing turn_bits by 1 should double the peak RSS since it's used as a power of 2."""
    base: int = _parallel._BASELINE_RSS_BYTES
    p20: int = _parallel.estimate_peak_rss_bytes_per_bunch(_min_harpy_input(20, ["lin", "full_spectra"]), 536)
    p21: int = _parallel.estimate_peak_rss_bytes_per_bunch(_min_harpy_input(21, ["lin", "full_spectra"]), 536)
    p22: int = _parallel.estimate_peak_rss_bytes_per_bunch(_min_harpy_input(22, ["lin", "full_spectra"]), 536)
    assert (p21 - base) == 2 * (p20 - base)
    assert (p22 - base) == 2 * (p21 - base)


@pytest.mark.basic
def test_full_spectra_much_heavier_than_lin_only():
    """
    When providing full_spectra (as in the CCC) the estimated peak RSS should drastically increase.
    See inside the estimate_peak_rss_bytes_per_bunch docstring and comments for why.
    """
    lin: int = _parallel.estimate_peak_rss_bytes_per_bunch(_min_harpy_input(to_write=["lin"]), 536)
    full: int = _parallel.estimate_peak_rss_bytes_per_bunch(_min_harpy_input(to_write=["lin", "full_spectra"]), 536)
    assert full > 5 * lin


@pytest.mark.basic
def test_no_clean_triggers_full_transient():
    """Similarly, no cleaning should also drastically up the estimate."""
    narrow: int = _parallel.estimate_peak_rss_bytes_per_bunch(_min_harpy_input(clean=True, to_write=["lin"]), 536)
    no_clean: int = _parallel.estimate_peak_rss_bytes_per_bunch(_min_harpy_input(clean=False, to_write=["lin"]), 536)
    assert no_clean > 5 * narrow


# ----- Determination of Number of Parallel Workers ----- #


@pytest.mark.basic
def test_decide_n_jobs_core_bound(monkeypatch):
    """So much RAM the number of cores limits us."""
    monkeypatch.setattr(_parallel, "usable_cores", lambda: 8)  # 8 cores only
    monkeypatch.setattr(_parallel, "available_ram_bytes", lambda: int(1e12))  # 1000GB of RAM!
    assert _parallel.decide_n_workers(_min_harpy_input(), 100, 536, requested=0) == 8


@pytest.mark.basic
def test_decide_n_jobs_bunch_bound(monkeypatch):
    """So few bunches we should just dispatch one worker for each."""
    monkeypatch.setattr(_parallel, "usable_cores", lambda: 64)  # 64 cores
    monkeypatch.setattr(_parallel, "available_ram_bytes", lambda: int(1e12))  # 1000GB of RAM!
    assert _parallel.decide_n_workers(_min_harpy_input(), 3, 536, requested=0) == 3


@pytest.mark.basic
def test_decide_n_jobs_ram_bound(monkeypatch):
    """Available RAM would have us out of memory so it determines the number of workers."""
    monkeypatch.setattr(_parallel, "usable_cores", lambda: 64)  # 64 cores
    monkeypatch.setattr(_parallel, "available_ram_bytes", lambda: int(100e9))  # 100 GB RAM
    harpy_input: DotDict = _min_harpy_input(to_write=["lin", "full_spectra"])  # ~15 GB/worker
    peak: int = _parallel.estimate_peak_rss_bytes_per_bunch(harpy_input, 536)
    n_workers: int = _parallel.decide_n_workers(harpy_input, 64, 536, requested=0)
    assert n_workers == int(_parallel.ALLOWED_RAM_PORTION * 100e9 / peak)
    assert n_workers < 10  # RAM-bound, well below the 64 cores / number of bunches (we properly throttle)


@pytest.mark.basic
def test_decide_n_jobs_requested_serial(monkeypatch):
    """Ensure the provided --n_jobs is used if RAM permits it."""
    monkeypatch.setattr(_parallel, "usable_cores", lambda: 64)  # 64 cores
    monkeypatch.setattr(_parallel, "available_ram_bytes", lambda: int(1e12))  # 1000GB of RAM!
    assert _parallel.decide_n_workers(_min_harpy_input(), 100, 536, requested=1) == 1


@pytest.mark.basic
def test_decide_n_jobs_requested_still_ram_clamped(monkeypatch):
    """Ensure the provided --n_jobs is bypassed if RAM limits it."""
    monkeypatch.setattr(_parallel, "usable_cores", lambda: 64)  # 64 cores
    monkeypatch.setattr(_parallel, "available_ram_bytes", lambda: int(30e9))  # 30 GB of RAM
    harpy_input = _min_harpy_input(to_write=["lin", "full_spectra"])  # ~15 GB/worker -> ~1 fits with safety margin
    assert _parallel.decide_n_workers(harpy_input, 100, 536, requested=16) == 1


@pytest.mark.basic
def test_decide_n_jobs_requested_clamped_to_cores(monkeypatch):
    """A requested N above the usable cores is throttled down: we never oversubscribe cores."""
    monkeypatch.setattr(_parallel, "usable_cores", lambda: 8)  # 8 cores only
    monkeypatch.setattr(_parallel, "available_ram_bytes", lambda: int(1e12))  # 1000GB of RAM!
    assert _parallel.decide_n_workers(_min_harpy_input(), 100, 536, requested=200) == 8


@pytest.mark.basic
def test_decide_n_jobs_ram_unknown_falls_back_to_serial(monkeypatch):
    """
    With no RAM info we cannot size the pool safely, so deciding automatically
    conservatively falls back to a single worker (an OOM-killed worker would take
    the whole pool down with it, so 'slow' is a better failure mode than 'dead').
    """
    monkeypatch.setattr(_parallel, "usable_cores", lambda: 8)  # 8 cores
    monkeypatch.setattr(_parallel, "available_ram_bytes", lambda: None)  # Unavailable RAM info
    assert _parallel.decide_n_workers(_min_harpy_input(), 100, 536, requested=0) == 1
    assert _parallel.decide_n_workers(_min_harpy_input(), 3, 536, requested=0) == 1


@pytest.mark.basic
def test_decide_n_jobs_ram_unknown_honours_explicit_request(monkeypatch):
    """
    An explicitly requested --n_jobs lifts the fallback above: the user knows their
    machine better than a detection which just failed. The cores and bunches caps
    still apply, though.
    """
    monkeypatch.setattr(_parallel, "usable_cores", lambda: 8)  # 8 cores
    monkeypatch.setattr(_parallel, "available_ram_bytes", lambda: None)  # Unavailable RAM info
    assert _parallel.decide_n_workers(_min_harpy_input(), 100, 536, requested=4) == 4  # requested
    assert _parallel.decide_n_workers(_min_harpy_input(), 100, 536, requested=200) == 8  # n_cores
    assert _parallel.decide_n_workers(_min_harpy_input(), 3, 536, requested=64) == 3  # n_bunches


@pytest.mark.basic
def test_decide_n_jobs_ram_unknown_warns(monkeypatch, caplog):
    """
    Both unknown-RAM paths warn: the automatic one so that an unexpected serial run
    can be diagnosed, and the explicit one so the user knows we did not validate it.
    """
    monkeypatch.setattr(_parallel, "usable_cores", lambda: 8)  # 8 cores
    monkeypatch.setattr(_parallel, "available_ram_bytes", lambda: None)  # Unavailable RAM info

    with caplog.at_level(logging.WARNING):
        _parallel.decide_n_workers(_min_harpy_input(), 100, 536, requested=0)
    assert "falling back to a single worker" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _parallel.decide_n_workers(_min_harpy_input(), 100, 536, requested=4)
    assert "explicitly requested n_jobs=4" in caplog.text


@pytest.mark.basic
def test_decide_n_jobs_never_below_one(monkeypatch):
    """Ensure with a very limited system we still attribute 1 worker."""
    monkeypatch.setattr(_parallel, "usable_cores", lambda: 1)  # 1 core!
    monkeypatch.setattr(_parallel, "available_ram_bytes", lambda: 1)  # 1 byte!
    assert _parallel.decide_n_workers(_min_harpy_input(), 0, 536, requested=0) == 1
    assert _parallel.decide_n_workers(_min_harpy_input(), 10, 536, requested=0) == 1
