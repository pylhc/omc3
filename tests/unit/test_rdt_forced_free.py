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
    # _delta_plus_correction returns the correction in the units of the measured complex line:
    # the h^- line coefficient divided by the *amplitude* of the local driven tune line, with an
    # absolute phase (harpy normalizes amplitudes, not phases, by the tune line). Multiplying by
    # |dxm| recovers normalized (hat_x = x/sqrt(beta)) units. With no AC-dipole element in this
    # bare model it is assumed just upstream of the reference BPM, here the observation point at
    # the end of the turn: exactly the geometry of the reference formula.
    ours_normalized = a_cross[0, 0] * abs(dxm)

    target_freq = line[0] * qdx + line[1] * qdy
    expected = _reference_forced_line(
        target_freq, plane, qdx, qdy, dxm, dxp, dym, dyp,
        mux_src, muy_src, betx_src, bety_src, k4l_src, qx_free, qy_free, nmax,
    )

    assert ours_normalized == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------------------------
# Tracking-truth tests: a toy ring in normalized coordinates (beta = 1 everywhere), decapoles,
# AC dipoles in both planes MID-RING, BPMs on both sides of the AC dipole and of the data-turn
# start, full tunes with an integer part. Validates the AC-dipole bookkeeping of the forced-line
# model (delta_+ correction) and the location factor against the tracked spectrum itself.
# ---------------------------------------------------------------------------------------------
_QX, _QY = 2.28, 2.313           # full tunes
_QDX, _QDY = 2.265, 2.326        # drive tunes (Q_- = -0.015, +0.013), no line collisions
# (name, MUX, MUY, K4L) in beam order; K4L is the MAD-X decapole strength (European K5L)
_TOY = [
    ("BPM.A", 0.05, 0.07, 0.0), ("MCD.1", 0.31, 0.22, 900.0), ("BPM.REF", 0.52, 0.49, 0.0),
    ("MCD.2", 0.83, 0.95, -1400.0), ("BPM.B", 1.02, 1.10, 0.0), ("MKQA", 1.21, 1.26, 0.0),
    ("BPM.C", 1.37, 1.44, 0.0), ("MCD.3", 1.66, 1.58, 1100.0), ("BPM.D", 1.90, 1.83, 0.0),
    ("MCD.4", 2.05, 2.16, -700.0), ("BPM.E", 2.19, 2.25, 0.0),
]


def _naff(sig, f, tt):
    w = np.hanning(len(sig))
    w /= w.mean()
    return np.mean(sig * w * np.exp(-2j * np.pi * f * tt))


def _ls_line(sig, tt, f, big):
    """Complex amplitude of the e^{2 pi i f n} component, fitted jointly with the strong lines
    `big` so that their window leakage (which swamps weak secondary lines) is removed exactly."""
    freqs = [f] + [b for b in big if abs(((b - f) + 0.5) % 1 - 0.5) > 1e-6]
    basis = np.exp(2j * np.pi * np.outer(tt - tt[0], freqs))
    coef = np.linalg.lstsq(basis, sig, rcond=None)[0]
    return coef[0] * np.exp(2j * np.pi * f * tt[0])


def _refine(sig, f0, tt, span=2e-4):
    from scipy.optimize import minimize_scalar
    return minimize_scalar(lambda f: -abs(_naff(sig, f, tt)), bounds=(f0 - span, f0 + span),
                           method="bounded", options={"xatol": 1e-13}).x


@pytest.fixture(scope="module")
def toy_tracking():
    def rot(u, pu, mu):
        c, s = np.cos(2 * np.pi * mu), np.sin(2 * np.pi * mu)
        return c * u + s * pu, -s * u + c * pu

    theta_x, theta_y, n_ramp, n_flat = 1.4e-4, 1.6e-4, 3000, 4096
    x = px = y = py = 0.0
    bpms = [e[0] for e in _TOY if e[0].startswith("BPM")]
    rec = {b: [] for b in bpms}
    for n in range(n_ramp + n_flat):
        cx = cy = 0.0
        ramp = min(1.0, (n + 1) / n_ramp)
        for name, mux, muy, k4l in _TOY:
            x, px = rot(x, px, mux - cx)
            y, py = rot(y, py, muy - cy)
            cx, cy = mux, muy
            if k4l:
                kick = -(k4l / 24) * (x - 1j * y) ** 4
                px, py = px + kick.real, py + kick.imag
            elif name == "MKQA":
                px += ramp * theta_x * np.cos(2 * np.pi * _QDX * n + 0.3)
                py += ramp * theta_y * np.cos(2 * np.pi * _QDY * n + 1.1)
            elif n >= n_ramp:
                rec[name].append((x - 1j * px, y - 1j * py))
        x, px = rot(x, px, _QX - cx)
        y, py = rot(y, py, _QY - cy)
    tt = np.arange(n_ramp, n_ramp + n_flat)
    return {b: (np.array(v)[:, 0], np.array(v)[:, 1]) for b, v in rec.items()}, tt


def _toy_meas_input(reversed_model=False):
    # reversed_model: the model table ordered against the beam (as LHC Beam 2 models, in the
    # Beam 1 frame): MU_model = Q_full - MU_beam and beam_direction = -1. Must give identical results.
    sign, offx, offy = (-1, _QX, _QY) if reversed_model else (1, 0.0, 0.0)
    elements = tfs.TfsDataFrame(
        {"MUX": [offx + sign * e[1] for e in _TOY], "MUY": [offy + sign * e[2] for e in _TOY],
         "BETX": 1.0, "BETY": 1.0, "K4L": [e[3] for e in _TOY]},
        index=[e[0] for e in _TOY], headers={"Q1": _QX, "Q2": _QY},
    )

    class _Accel:
        def __init__(self):
            self.elements = elements
            self.beam_direction = -1 if reversed_model else 1

        def get_exciter_bpm(self, plane, bpms):
            return "BPM.C", "MKQA"

    class _MeasInput(dict):
        def __getattr__(self, item):
            return self[item]

    return _MeasInput(accelerator=_Accel(), rdt_magnet_order=5)


def _big_lines(tunes):
    qdx, qdy = tunes["X"]["Q"], tunes["Y"]["Q"]
    return [qdx, -qdx, qdy, -qdy, tunes["X"]["QF"], tunes["Y"]["QF"]]


def _toy_deltas_and_tunes(signals, tt):
    hx, hy = signals["BPM.REF"]
    fdx, fdy = _refine(hx, _QDX % 1, tt), _refine(hy, _QDY % 1, tt)
    tunes = _driven_tunes(_QX % 1, _QY % 1, fdx, fdy)
    big = _big_lines(tunes)
    deltas = {
        "X": ("BPM.REF", np.array([_ls_line(hx, tt, fdx, big)]), np.array([_ls_line(hx, tt, -fdx, big)])),
        "Y": ("BPM.REF", np.array([_ls_line(hy, tt, fdy, big)]), np.array([_ls_line(hy, tt, -fdy, big)])),
    }
    return deltas, tunes


@pytest.mark.basic
@pytest.mark.parametrize("line,plane", [((0, 4, 0), "X"), ((1, -3, 0), "Y"), ((0, -2, 0), "X")])
def test_forced_line_model_matches_tracking(toy_tracking, line, plane):
    """Complete first-order forced line (clean + delta_+ cross-terms) at every BPM, in the
    measured units (h^- line coefficient / |local tune line|, absolute phase), vs tracking."""
    from omc3.optics_measurements.rdt import _forced_line_model

    signals, tt = toy_tracking
    deltas, tunes = _toy_deltas_and_tunes(signals, tt)
    index = pd.Index(list(signals))
    clean, cross = _forced_line_model(_toy_meas_input(), plane, line, tunes, deltas, 1, index, 5)
    f_line = line[0] * tunes["X"]["Q"] + line[1] * tunes["Y"]["Q"]
    big = _big_lines(tunes)
    for i, bpm in enumerate(index):
        sig = signals[bpm][0 if plane == "X" else 1]
        measured = _ls_line(sig, tt, f_line, big) / abs(_ls_line(sig, tt, tunes[plane]["Q"], big))
        pred = clean[i, 0] + cross[i, 0]
        assert abs(cross[i, 0]) > 5e-3 * abs(pred)       # the delta_+ part is not negligible
        assert pred / measured == pytest.approx(1.0, abs=2e-3), bpm


@pytest.mark.basic
def test_location_factor_recovers_free_rdt_at_every_bpm(toy_tracking):
    """f1004 read on X: the forced RDT extracted at each BPM (delta_+ removed), converted with
    the plain factor divided by the location factor, equals the free RDT at that BPM."""
    from omc3.optics_measurements.rdt import (
        _forced_line_model, _forced_to_free_factor, _location_factor, _wrapped_phase_matrix,
    )

    signals, tt = toy_tracking
    deltas, tunes = _toy_deltas_and_tunes(signals, tt)
    meas_input = _toy_meas_input()
    index = pd.Index(list(signals))
    rdt, line = (1, 0, 0, 4), (0, 4, 0)
    _clean, cross = _forced_line_model(meas_input, "X", line, tunes, deltas, 1, index, 5)
    loc = _location_factor(meas_input, "X", line, rdt, tunes, index)
    conv = _forced_to_free_factor(line, "X", tunes)

    el = meas_input.accelerator.elements
    src = el[el["K4L"] != 0]
    h = src["K4L"].to_numpy() / (factorial(4) * 2 ** 5)       # h_1004 = -K i^4 / (1! 4! 2^5)
    f_free = (np.exp(2j * np.pi * (
        _wrapped_phase_matrix(src["MUX"].to_numpy(), el.loc[index, "MUX"].to_numpy(), _QX)
        - 4 * _wrapped_phase_matrix(src["MUY"].to_numpy(), el.loc[index, "MUY"].to_numpy(), _QY)
    )) @ (-h)) / (1 - np.exp(2j * np.pi * ((_QX % 1) - 4 * (_QY % 1))))

    plain_errors = []
    big = _big_lines(tunes)
    for i, bpm in enumerate(index):
        hx, hy = signals[bpm]
        ax = _ls_line(hx, tt, tunes["X"]["Q"], big)
        ay = _ls_line(hy, tt, tunes["Y"]["Q"], big)
        c_line = _ls_line(hx, tt, 4 * tunes["Y"]["Q"], big)
        f_forced = (c_line - cross[i, 0] * abs(ax)) / (-2j * ay ** 4)
        assert f_forced * conv / loc[i] == pytest.approx(f_free[i], rel=5e-3), bpm
        plain_errors.append(abs(f_forced * conv / f_free[i] - 1))
    assert max(plain_errors) > 0.05       # without the location factor the error is visible


@pytest.mark.basic
def test_forced_to_free_factor_error_matches_analytic_derivative():
    from omc3.optics_measurements.rdt import _forced_to_free_factor_error

    qx, qy, qdy = 0.28, 0.313, 0.323
    tunes = _driven_tunes(qx, qy, qx, qdy)
    line = (0, 4, 0)                        # f1004: near the Qx - 4Qy = -1 resonance
    sig_abs, sig_arg = _forced_to_free_factor_error(line, "X", tunes, {"rdt_free_tune_error": 1e-3})
    th_fo, th_fr = qx - 4 * qdy, qx - 4 * qy
    d_qx = np.pi / np.tan(np.pi * th_fo) - np.pi / np.tan(np.pi * th_fr)   # d ln|F| / d Qx_free
    d_qy = 4 * np.pi / np.tan(np.pi * th_fr)                             # d ln|F| / d Qy_free
    assert sig_abs == pytest.approx(1e-3 * np.hypot(d_qx, d_qy), rel=1e-4)
    assert sig_arg == pytest.approx(1e-3 * 4 * np.pi, rel=1e-4)   # only Theta_free depends on Qy
    assert _forced_to_free_factor_error(line, "X", _free_tunes(qx, qy), {"rdt_free_tune_error": 1e-3}) == (0.0, 0.0)


@pytest.mark.basic
def test_freq_header_uses_driven_tunes():
    from omc3.optics_measurements.rdt import add_freq_to_header

    tunes = _driven_tunes(0.28, 0.313, 0.272, 0.323)
    hdr = add_freq_to_header({"Q1": 0.28, "Q2": 0.313}, "X", (1, 0, 0, 4), tunes)
    assert hdr["FREQ"] == pytest.approx((4 * 0.323) % 1)     # 4 Q_Dy = 1.292 -> 0.292
    assert hdr["FREQFREE"] == pytest.approx(4 * 0.313 - 1)


@pytest.mark.basic
def test_forced_line_model_and_location_factor_follow_beam_direction(toy_tracking):
    """A model ordered against the beam (LHC Beam 2 convention) must give the same prediction."""
    from omc3.optics_measurements.rdt import _forced_line_model, _location_factor

    signals, tt = toy_tracking
    deltas, tunes = _toy_deltas_and_tunes(signals, tt)
    index = pd.Index(list(signals))
    fwd = _forced_line_model(_toy_meas_input(), "X", (0, 4, 0), tunes, deltas, 1, index, 5)
    rev = _forced_line_model(_toy_meas_input(True), "X", (0, 4, 0), tunes, deltas, 1, index, 5)
    for a, b in zip(fwd, rev):
        np.testing.assert_allclose(a, b, rtol=1e-9)
    loc_f = _location_factor(_toy_meas_input(), "X", (0, 4, 0), (1, 0, 0, 4), tunes, index)
    loc_r = _location_factor(_toy_meas_input(True), "X", (0, 4, 0), (1, 0, 0, 4), tunes, index)
    np.testing.assert_allclose(loc_f, loc_r, rtol=1e-9)
