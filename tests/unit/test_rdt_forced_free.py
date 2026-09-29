from itertools import product
from math import comb, factorial

import numpy as np
import pandas as pd
import pytest
import tfs

from omc3.optics_measurements.rdt import (
    _apply_forced_to_free,
    _decompose_line,
    _delta_plus_correction,
    _forced_to_free_factor,
    _group_rdts_by_line,
    get_linearized_problem,
)
from omc3.optics_measurements.tune import TuneDict


def _free_tunes(qx: float, qy: float) -> TuneDict:
    """A TuneDict with no AC-dipole excitation: measured tune equals the free tune."""
    tunes = TuneDict()
    tunes["X"]["Q"] = tunes["X"]["QF"] = qx
    tunes["Y"]["Q"] = tunes["Y"]["QF"] = qy
    return tunes


def _driven_tunes(qx: float, qy: float, qdx: float, qdy: float) -> TuneDict:
    tunes = TuneDict()
    tunes["X"]["Q"], tunes["X"]["QF"] = qdx, qx
    tunes["Y"]["Q"], tunes["Y"]["QF"] = qdy, qy
    return tunes


@pytest.mark.basic
def test_group_rdts_by_line():
    # f3000 and f4100 both drive the -2Qx line
    groups = _group_rdts_by_line([(3, 0, 0, 0), (4, 1, 0, 0), (1, 0, 2, 0)], "X")
    assert set(groups) == {(-2, 0, 0), (0, -2, 0)}
    assert groups[(-2, 0, 0)] == [(3, 0, 0, 0), (4, 1, 0, 0)]
    assert groups[(0, -2, 0)] == [(1, 0, 2, 0)]


@pytest.mark.basic
@pytest.mark.parametrize("rdt,plane", [((3, 0, 0, 0), "X"), ((1, 0, 2, 0), "X"), ((0, 0, 3, 0), "Y")])
def test_forced_to_free_factor_is_identity_without_excitation(rdt, plane):
    from omc3.optics_measurements.rdt import _determine_line

    tunes = _free_tunes(0.28, 0.31)
    line = _determine_line(rdt, plane)
    factor = _forced_to_free_factor(line, plane, tunes)
    assert factor == pytest.approx(1.0 + 0.0j, abs=1e-12)


@pytest.mark.basic
def test_forced_to_free_factor_matches_sine_ratio_and_half_phase():
    """Cross-check the general (nu_x, nu_y) implementation against the single-plane
    sine-ratio-plus-half-phase formula (tracking-validated; see the companion paper),
    written out independently in terms of Delta = 2*pi*Theta rather than reusing the
    function's own Theta/sin(pi*Theta) parametrization.

    Note: a *direct ratio of the two RDT denominators* (1 - exp(2j*pi*Theta)) gives the
    opposite sign for the half-phase term -- this is a known, tracking-settled subtlety
    (see the reference paper, Sec. V.A), not a bug: this test intentionally checks against
    the correct (validated) formula, not the naive one.
    """
    from omc3.optics_measurements.rdt import _determine_line

    qx, qy = 0.28, 0.31
    qdx, qdy = 0.27, 0.31
    tunes = _driven_tunes(qx, qy, qdx, qdy)
    rdt = (3, 0, 0, 0)
    line = _determine_line(rdt, "X")  # (-2, 0, 0)
    nu_x, nu_y = line[0], line[1]

    delta_free = 2 * np.pi * ((3 - 0) * qx + (0 - 0) * qy)
    delta_forced = 2 * np.pi * (qx - nu_x * qdx - nu_y * qdy)
    expected = (
        np.sin(delta_forced / 2) / np.sin(delta_free / 2)
        * np.exp(1j * (delta_free - delta_forced) / 2)
    )

    factor = _forced_to_free_factor(line, "X", tunes)
    assert factor == pytest.approx(expected, rel=1e-9)


@pytest.mark.basic
def test_apply_forced_to_free_roundtrip():
    amp = np.array([2.0, 3.0])
    err_amp = np.array([0.1, 0.2])
    phase = np.array([0.1, 0.6])
    factor = 1.5 * np.exp(1j * np.pi / 3)  # |factor|=1.5, arg=60deg

    amp_free, err_amp_free, phase_free = _apply_forced_to_free(amp, err_amp, phase, factor)
    assert amp_free == pytest.approx(amp * 1.5)
    assert err_amp_free == pytest.approx(err_amp * 1.5)
    assert phase_free == pytest.approx((phase + (1.0 / 6.0)) % 1)

    # factor == 1 must be a no-op
    amp_id, err_id, phase_id = _apply_forced_to_free(amp, err_amp, phase, 1.0 + 0.0j)
    assert amp_id == pytest.approx(amp)
    assert err_id == pytest.approx(err_amp)
    assert phase_id == pytest.approx(phase)


@pytest.mark.basic
def test_decompose_line_recovers_synthetic_rdts():
    """Two multipole orders (f3000, f4100) sharing the -2Qx line: synthesize a noiseless
    complex line coefficient from known RDTs and check the decomposition recovers them,
    with a near-zero residual fraction."""
    rdt1, rdt2 = (3, 0, 0, 0), (4, 1, 0, 0)
    plane = "X"
    line = (-2, 0, 0)

    n_kicks = 4
    act_x = np.array([1e-4, 2e-4, 3e-4, 4e-4])
    act_y = np.full(n_kicks, 5e-5)
    invariants = {
        "X": np.column_stack([act_x, np.zeros(n_kicks)]),
        "Y": np.column_stack([act_y, np.zeros(n_kicks)]),
    }

    design = np.column_stack(
        [get_linearized_problem(invariants, plane, rdt1), get_linearized_problem(invariants, plane, rdt2)]
    )

    f1_true = np.array([10.0 + 2.0j, -5.0 + 1.0j])  # two BPMs
    f2_true = np.array([3.0 - 4.0j, 1.0 + 1.0j])

    complex_line = np.stack(
        [design @ np.array([f1_true[b], f2_true[b]]) for b in range(2)]
    )  # (n_bpms, n_kicks)
    line_amp = np.abs(complex_line)
    rdt_phases_per_file = (np.angle(complex_line) / (2 * np.pi)) % 1

    index = pd.Index(["BPM1", "BPM2"], name="NAME")
    result = _decompose_line(
        invariants, plane, line, [rdt1, rdt2], line_amp, rdt_phases_per_file, index, conversion=1.0 + 0.0j
    )

    for rdt, f_true in ((rdt1, f1_true), (rdt2, f2_true)):
        df = result[rdt]
        recovered = df["AMPDECOMP"].to_numpy() * np.exp(2j * np.pi * df["PHASEDECOMP"].to_numpy())
        assert recovered == pytest.approx(f_true, rel=1e-6)
        assert df["LINERESIDUAL"].to_numpy() == pytest.approx([0.0, 0.0], abs=1e-8)
        assert (df["NCONTRIBUTORS"] == 2).all()
        # free == forced here since conversion == 1
        assert df["AMPDECOMPFREE"].to_numpy() == pytest.approx(df["AMPDECOMP"].to_numpy())


def _reference_forced_line(
    target_freq: float,
    plane: str,
    qdx: float,
    qdy: float,
    dxm: complex, dxp: complex, dym: complex, dyp: complex,
    mux_src: np.ndarray, muy_src: np.ndarray,
    betx_src: np.ndarray, bety_src: np.ndarray, k4l_src: np.ndarray,
    qx_free: float, qy_free: float,
    nmax: int,
) -> complex:
    """
    Independent reference for the delta_+ cross-term correction (Eq. 33), a direct transcription
    of the tracking-validated notebook formula (`xsuite_lhc/full_forced_prediction.py` /
    `build_nb06.py`'s `forced_line`) for the SPECIAL case of an observation point sitting at the
    "lattice start" (mux_b=Qx_free, muy_b=Qy_free): there, the general (arbitrary-observation-point)
    formula implemented in `_delta_plus_correction` is analytically shown (companion paper) to
    reduce exactly to this simpler, already-validated one. Returns the pure cross term
    (C_full - C_pure in the notebook's naming), i.e. excluding the clean all-delta_- partition.
    """
    two_pi = 2 * np.pi
    mux_rad, muy_rad = mux_src * two_pi, muy_src * two_pi

    def alias(f):
        return (f + 0.5) % 1.0 - 0.5

    def h_arr(j, k, l, m):  # noqa: E741
        n = j + k + l + m
        if (l + m) % 2 or n != nmax:
            return np.zeros_like(betx_src)
        return (
            -(k4l_src * (1j) ** (l + m) * betx_src ** ((j + k) / 2) * bety_src ** ((l + m) / 2)
              / (factorial(j) * factorial(k) * factorial(l) * factorial(m) * 2 ** n))
        ).real

    target = round(alias(target_freq), 9)
    need = (lambda t: t[0] >= 1) if plane == "X" else (lambda t: t[2] >= 1)
    terms = [
        (j, k, l, m)  # noqa: E741
        for j, k, l, m in product(range(nmax + 1), repeat=4)  # noqa: E741
        if 3 <= j + k + l + m <= nmax and (l + m) % 2 == 0 and need((j, k, l, m))
    ]
    c_full = c_pure = 0j
    for (j, k, l, m) in terms:  # noqa: E741
        hw = h_arr(j, k, l, m)
        if not np.any(hw):
            continue
        if plane == "X":
            spatial = np.sum(hw * np.exp(1j * ((two_pi * qx_free - mux_rad) + (m - l) * muy_rad + (k - j + 1) * mux_rad)))
            ax_range, ay_range, coeff, q_r = range(j), range(l + 1), -2j * j, qx_free
        else:
            spatial = np.sum(hw * np.exp(1j * ((two_pi * qy_free - muy_rad) + (k - j) * mux_rad + (m - l + 1) * muy_rad)))
            ax_range, ay_range, coeff, q_r = range(j + 1), range(l), -2j * l, qy_free
        for ax_ in ax_range:
            for bx in range(k + 1):
                for ay in ay_range:
                    for by in range(m + 1):
                        if plane == "X":
                            nux = (2 * ax_ - (j - 1)) + (2 * bx - k)
                            nuy = (2 * ay - l) + (2 * by - m)
                            binom = comb(j - 1, ax_) * comb(k, bx) * comb(l, ay) * comb(m, by)
                            xp = np.conj(dxm) ** (j - 1 - ax_) * np.conj(dxp) ** ax_
                            yp = np.conj(dym) ** (l - ay) * np.conj(dyp) ** ay
                        else:
                            nux = (2 * ax_ - j) + (2 * bx - k)
                            nuy = (2 * ay - (l - 1)) + (2 * by - m)
                            binom = comb(j, ax_) * comb(k, bx) * comb(l - 1, ay) * comb(m, by)
                            xp = np.conj(dxm) ** (j - ax_) * np.conj(dxp) ** ax_
                            yp = np.conj(dym) ** (l - 1 - ay) * np.conj(dyp) ** ay
                        if round(alias(nux * qdx + nuy * qdy), 9) != target:
                            continue
                        mix = binom * xp * yp * dxm ** bx * dxp ** (k - bx) * dym ** by * dyp ** (m - by)
                        om = two_pi * (nux * qdx + nuy * qdy)
                        c = coeff * spatial * mix * np.exp(-1j * om) / (1 - np.exp(1j * (two_pi * q_r - om)))
                        c_full += c
                        if (ax_, bx, ay, by) == (0, k, 0, m):
                            c_pure += c
    return c_full - c_pure


@pytest.mark.basic
def test_delta_plus_correction_matches_reference_formula():
    """
    Cross-validates `_delta_plus_correction` (general observation point) against an independent
    transcription of the tracking-validated notebook formula (only valid at a single, fixed
    observation point -- the "lattice start", mux_b=Qx_free, muy_b=Qy_free -- where the two are
    analytically equal). A synthetic 4-source decapole model feeds f1004's own line (+4Qy, read
    on X): this exercises the actual multi-partition, multi-order sum, not just a trivial case.
    """
    qx_free, qy_free = 0.28, 0.313
    qdx, qdy = 0.272, 0.323
    line = (0, 4, 0)  # f1004's line
    plane = "X"
    nmax = 5

    rng = np.random.default_rng(42)
    n_src = 4
    mux_src = rng.uniform(0, qx_free, n_src)
    muy_src = rng.uniform(0, qy_free, n_src)
    betx_src = rng.uniform(20, 200, n_src)
    bety_src = rng.uniform(20, 200, n_src)
    k4l_src = rng.uniform(-300, 300, n_src)

    dxm, dxp = 1e-5 + 2e-6j, 3e-7 - 1e-7j
    dym, dyp = 2e-4 - 1e-5j, 4e-6 + 2e-6j

    obs_name = "OBS_AT_START"
    obs_beta = 100.0
    # a bare-bones TfsDataFrame (not a plain DataFrame): _delta_plus_correction reads the FULL,
    # unreduced per-turn tune from elements.headers["Q1"/"Q2"] to wrap MUX/MUY (which, in a real
    # model, accumulate over the *whole* tune, e.g. 0 to 62.28 for the LHC, not just its
    # fractional part) -- here Q_full == Qx_free/Qy_free (no integer part), so the synthetic
    # mux_src/muy_src generated within [0, qx_free)/[0, qy_free) below remain self-consistent.
    elements = tfs.TfsDataFrame(
        {
            "MUX": np.concatenate([mux_src, [qx_free]]),
            "MUY": np.concatenate([muy_src, [qy_free]]),
            "BETX": np.concatenate([betx_src, [obs_beta]]),
            "BETY": np.concatenate([bety_src, [obs_beta]]),
            "K4L": np.concatenate([k4l_src, [0.0]]),
        },
        index=[f"MCD.{i}" for i in range(n_src)] + [obs_name],
        headers={"Q1": qx_free, "Q2": qy_free},
    )

    class _FakeAccel:
        def __init__(self, elements):
            self.elements = elements

    class _FakeMeasInput(dict):
        def __getattr__(self, item):
            return self[item]

    meas_input = _FakeMeasInput(accelerator=_FakeAccel(elements))
    tunes = {"X": {"QF": qx_free, "Q": qdx}, "Y": {"QF": qy_free, "Q": qdy}}
    deltas = {
        "X": (obs_name, np.array([dxm]), np.array([dxp])),
        "Y": (obs_name, np.array([dym]), np.array([dyp])),
    }
    index = pd.Index([obs_name])

    a_cross = _delta_plus_correction(meas_input, plane, line, tunes, deltas, 1, index, nmax)
    # _delta_plus_correction returns the correction in harpy's ratio-space convention (divided by
    # the readout plane's own driven-mode complex amplitude, propagated from the reference BPM to
    # the observation point); since here the observation point IS the reference point, that
    # propagated amplitude is exactly dxm (plane "X"), so multiplying back by it undoes the
    # division and recovers the normalized (hat_x = x/sqrt(beta)) units the reference formula
    # works in.
    ours_normalized = a_cross[0, 0] * dxm

    target_freq = line[0] * qdx + line[1] * qdy
    expected = _reference_forced_line(
        target_freq, plane, qdx, qdy, dxm, dxp, dym, dyp,
        mux_src, muy_src, betx_src, bety_src, k4l_src, qx_free, qy_free, nmax,
    )

    assert ours_normalized == pytest.approx(expected, rel=1e-9)
