"""
Unit tests for the harpy parallel orchestration strategy in omc3.harpy._parallel.
"""

import pytest
from generic_parser import DotDict

from omc3.harpy import _parallel


def _min_harpy_input(turn_bits=20, to_write=("lin",), clean=True) -> DotDict:
    """Just the ones we query from harpy input."""
    return DotDict(turn_bits=turn_bits, to_write=list(to_write), clean=clean)


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
