import itertools
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tfs
import turn_by_turn as tbt

from omc3.definitions.constants import PLANES
from omc3.hole_in_one import hole_in_one_entrypoint

LIMITS = {"F1": 1e-6, "A1": 1.5e-3, "P1": 3e-4, "F2": 1.5e-4, "A2": 1.5e-1, "P2": 0.03}
NOISE = 3.2e-5
COUPLING = 0.01
NTURNS = 1024
NBPMS = 100
BASEAMP = 0.001
AMPZ, MUZ, TUNEZ = 0.01, 0.3, 0.008

HARPY_SETTINGS = {
    "clean": [True, False],
    "keep_exact_zeros": [False, True],
    "sing_val": [12],
    "peak_to_peak": [1e-8],
    "window": ["hann", "rectangle", "welch", "triangle", "hamming", "nuttal3", "nuttal4"],
    "max_peak": [0.02],
    "svd_dominance_limit": [0.925],
    "num_svd_iterations": [3],
    "tolerance": [0.01],
    "tune_clean_limit": [1e-5],
    "turn_bits": [18],
    "output_bits": [12],
}

HARPY_INPUT = list(itertools.product(*HARPY_SETTINGS.values()))


@pytest.mark.basic
def test_harpy(_test_file: Path, _model_file: Path):
    [
        clean,
        keep_exact_zeros,
        sing_val,
        peak_to_peak,
        window,
        max_peak,
        svd_dominance_limit,
        num_svd_iterations,
        tolerance,
        tune_clean_limit,
        turn_bits,
        output_bits,
    ] = HARPY_INPUT[0]

    model_df: pd.DataFrame = _get_model_dataframe()
    tfs.write(_model_file, model_df, save_index="NAME")
    _write_tbt_file(model_df, _test_file.parent)
    hole_in_one_entrypoint(
        harpy=True,
        clean=clean,
        keep_exact_zeros=keep_exact_zeros,
        sing_val=sing_val,
        peak_to_peak=peak_to_peak,
        window=window,
        max_peak=max_peak,
        svd_dominance_limit=svd_dominance_limit,
        num_svd_iterations=num_svd_iterations,
        tolerance=tolerance,
        tune_clean_limit=tune_clean_limit,
        turn_bits=turn_bits,
        output_bits=output_bits,
        autotunes="transverse",
        outputdir=_test_file.parent,
        files=[_test_file],
        model=_model_file,
        to_write=["lin"],
        unit="m",
    )

    linfiles = {"X": tfs.read(f"{_test_file}.linx"), "Y": tfs.read(f"{_test_file}.liny")}
    model_df = tfs.read(_model_file)
    _assert_spectra(linfiles, model_df)


@pytest.mark.basic
def test_harpy_without_model(_test_file, _model_file):
    model_df = _get_model_dataframe()
    tfs.write(_model_file, model_df, save_index="NAME")
    _write_tbt_file(model_df, _test_file.parent)
    hole_in_one_entrypoint(
        harpy=True,
        clean=True,
        autotunes="transverse",
        outputdir=_test_file.parent,
        files=[_test_file],
        to_write=["lin"],
        turn_bits=18,
        unit="m",
    )

    linfiles = {"X": tfs.read(f"{_test_file}.linx"), "Y": tfs.read(f"{_test_file}.liny")}
    model_df = tfs.read(_model_file)
    _assert_spectra(linfiles, model_df)


@pytest.mark.extended
@pytest.mark.parametrize(
    "clean, keep_exact_zeros, sing_val, peak_to_peak, window, max_peak,"
    "svd_dominance_limit, num_svd_iterations, tolerance, tune_clean_limit, turn_bits, output_bits",
    HARPY_INPUT,
)
def test_harpy_run(
    _test_file: Path,
    _model_file: Path,
    clean: bool,
    keep_exact_zeros: bool,
    sing_val: int,
    peak_to_peak: float,
    window: str,
    max_peak: float,
    svd_dominance_limit: float,
    num_svd_iterations: int,
    tolerance: float,
    tune_clean_limit: float,
    turn_bits: int,
    output_bits: int,
):
    model_df = _get_model_dataframe()
    tfs.write(_model_file, model_df, save_index="NAME")
    _write_tbt_file(model_df, _test_file.parent)
    hole_in_one_entrypoint(
        harpy=True,
        clean=clean,
        keep_exact_zeros=keep_exact_zeros,
        sing_val=sing_val,
        peak_to_peak=peak_to_peak,
        window=window,
        max_peak=max_peak,
        svd_dominance_limit=svd_dominance_limit,
        num_svd_iterations=num_svd_iterations,
        tolerance=tolerance,
        tune_clean_limit=tune_clean_limit,
        turn_bits=turn_bits,
        output_bits=output_bits,
        autotunes="transverse",
        outputdir=_test_file.parent,
        files=[_test_file],
        model=_model_file,
        to_write=["lin"],
        unit="m",
    )


@pytest.mark.extended
@pytest.mark.parametrize("nturns", [NTURNS, NTURNS - 1])
def test_freekick_harpy(_test_file: Path, _model_file: Path, nturns: int):
    model_df = _get_model_dataframe()
    tfs.write(_model_file, model_df, save_index="NAME")
    _write_tbt_file(model_df, _test_file.parent, nturns)
    hole_in_one_entrypoint(
        harpy=True,
        clean=True,
        autotunes="transverse",
        is_free_kick=True,
        outputdir=_test_file.parent,
        files=[_test_file],
        model=_model_file,
        to_write=["lin"],
        unit="m",
        turn_bits=18,
    )

    linfiles = {"X": tfs.read(f"{_test_file}.linx"), "Y": tfs.read(f"{_test_file}.liny")}
    model_df = tfs.read(_model_file)

    for plane in PLANES:
        # Main and secondary frequencies
        assert (
            _rms(_diff(linfiles[plane].loc[:, f"TUNE{plane}"], model_df.loc[:, f"TUNE{plane}"]))
            < LIMITS["F1"]
        )
        # Main and secondary amplitudes
        assert (
            _rms(_rel_diff(linfiles[plane].loc[:, f"AMP{plane}"], model_df.loc[:, f"AMP{plane}"]))
            < LIMITS["A1"]
        )
        # Main and secondary phases
        assert (
            _rms(_angle_diff(linfiles[plane].loc[:, f"MU{plane}"], model_df.loc[:, f"MU{plane}"]))
            < LIMITS["P1"]
        )


@pytest.mark.extended
def test_harpy_3d(_test_file: Path, _model_file: Path):
    model_df = _get_model_dataframe()
    tfs.write(_model_file, model_df, save_index="NAME")
    _write_tbt_file(model_df, _test_file.parent)
    hole_in_one_entrypoint(
        harpy=True,
        clean=True,
        autotunes="all",
        outputdir=_test_file.parent,
        files=[_test_file],
        model=_model_file,
        to_write=["lin"],
        turn_bits=18,
        unit="m",
    )

    linfiles = {"X": tfs.read(f"{_test_file}.linx"), "Y": tfs.read(f"{_test_file}.liny")}
    model_df = tfs.read(_model_file)
    _assert_spectra(linfiles, model_df)
    assert _rms(_diff(linfiles["X"].loc[:, "TUNEZ"], TUNEZ)) < LIMITS["F2"]
    assert (
        _rms(_rel_diff(linfiles["X"].loc[:, "AMPZ"] * linfiles["X"].loc[:, "AMPX"], AMPZ * BASEAMP))
        < LIMITS["A2"]
    )
    assert _rms(_angle_diff(linfiles["X"].loc[:, "MUZ"], MUZ)) < LIMITS["P2"]


@pytest.mark.basic
@pytest.mark.parametrize("nturns", [NTURNS, NTURNS - 1])
def test_harpy_aliased_resonance_line_phase(tmp_path: Path, nturns: int):
    """
    A resonance line outside [0, 0.5] (here the (0, 4) line in X, at 4 Q_y = 1.24, which the FFT finds
    at 0.24) must keep its phase relation to the tune lines for any number of turns: its phase minus
    4 x the vertical tune phase is independent of the reference turn. An odd number of turns used to
    shift it by half a turn (a sign flip of e.g. f1004).
    """
    rng = np.random.default_rng(20260930)
    qx, qy, line_amp = 0.28, 0.31, 0.1
    names = [f"BPM{i}" for i in range(20)]
    mux, muy, mu_line = (rng.random(len(names)) for _ in range(3))  # phases at turn 0
    turns = np.arange(nturns)
    wave = lambda mu, freq: BASEAMP * np.cos(2 * np.pi * (mu[:, None] + freq * turns[None, :]))
    data_x = wave(mux, qx) + line_amp * wave(mu_line, 4 * qy) + rng.normal(0, NOISE, (len(names), nturns))
    data_y = wave(muy, qy) + rng.normal(0, NOISE, (len(names), nturns))
    matrices = [tbt.TransverseData(X=pd.DataFrame(data_x, index=names), Y=pd.DataFrame(data_y, index=names))]
    tbt_data = tbt.TbtData(matrices=matrices, bunch_ids=[0], nturns=nturns, meta={"file": "test_file.sdds"})

    hole_in_one_entrypoint(
        harpy=True,
        clean=False,
        autotunes="transverse",
        resonances=5,
        outputdir=tmp_path,
        files=[tbt_data],
        tbt_datatype="tbt_data",
        to_write=["lin"],
        turn_bits=18,
        unit="m",
    )
    lin_x = tfs.read(tmp_path / "test_file.sdds.linx", index="NAME")
    lin_y = tfs.read(tmp_path / "test_file.sdds.liny", index="NAME")

    assert np.allclose(lin_x.loc[names, "FREQ04"], 4 * qy - 1, atol=LIMITS["F2"])
    measured = lin_x.loc[names, "PHASE04"] - 4 * lin_y.loc[names, "MUY"]
    expected = mu_line - 4 * muy
    assert _rms((measured - expected + 0.5) % 1 - 0.5) < LIMITS["P2"]


def _assert_spectra(linfiles: dict[str, tfs.TfsDataFrame], model_df: tfs.TfsDataFrame):
    for plane in PLANES:
        # main and secondary frequencies
        assert (
            _rms(_diff(linfiles[plane].loc[:, f"TUNE{plane}"], model_df.loc[:, f"TUNE{plane}"]))
            < LIMITS["F1"]
        )
        assert (
            _rms(
                _diff(
                    linfiles[plane].loc[:, f"FREQ{_couple(plane)}"],
                    model_df.loc[:, f"TUNE{_other(plane)}"],
                )
            )
            < LIMITS["F2"]
        )
        # main and secondary amplitudes
        assert (
            _rms(_rel_diff(linfiles[plane].loc[:, f"AMP{plane}"], model_df.loc[:, f"AMP{plane}"]))
            < LIMITS["A1"]
        )
        assert (
            _rms(
                _rel_diff(
                    linfiles[plane].loc[:, f"AMP{_couple(plane)}"]
                    * linfiles[plane].loc[:, f"AMP{plane}"],
                    COUPLING * model_df.loc[:, f"AMP{_other(plane)}"],
                )
            )
            < LIMITS["A2"]
        )
        # main and secondary phases
        assert (
            _rms(_angle_diff(linfiles[plane].loc[:, f"MU{plane}"], model_df.loc[:, f"MU{plane}"]))
            < LIMITS["P1"]
        )
        assert (
            _rms(
                _angle_diff(
                    linfiles[plane].loc[:, f"PHASE{_couple(plane)}"],
                    model_df.loc[:, f"MU{_other(plane)}"],
                )
            )
            < LIMITS["P2"]
        )


def _get_model_dataframe() -> pd.DataFrame:
    np.random.seed(1234567)
    return pd.DataFrame(
        data={
            "S": np.arange(NBPMS, dtype=float),
            "AMPX": (np.random.rand(NBPMS) + 1) * BASEAMP,
            "AMPY": (np.random.rand(NBPMS) + 1) * BASEAMP,
            "MUX": np.random.rand(NBPMS) - 0.5,
            "MUY": np.random.rand(NBPMS) - 0.5,
            "TUNEX": 0.25 + np.random.rand(1)[0] / 40,
            "TUNEY": 0.3 + np.random.rand(1)[0] / 40,
        },
        index=np.array(
            ["".join(random.choices(string.ascii_uppercase, k=7)) for _ in range(NBPMS)]
        ),
    )


def _write_tbt_file(model: tfs.TfsDataFrame, dir_path: Path, nturns: int = NTURNS) -> None:
    ints = np.arange(nturns) - nturns // 2  # phases in the model are at harpy's reference turn
    data_x = model.loc[:, "AMPX"].to_numpy()[:, None] * np.cos(
        2
        * np.pi
        * (
            model.loc[:, "MUX"].to_numpy()[:, None]
            + model.loc[:, "TUNEX"].to_numpy()[:, None] * ints[None, :]
        )
    )
    data_y = model.loc[:, "AMPY"].to_numpy()[:, None] * np.cos(
        2
        * np.pi
        * (
            model.loc[:, "MUY"].to_numpy()[:, None]
            + model.loc[:, "TUNEY"].to_numpy()[:, None] * ints[None, :]
        )
    )
    data_z = (
        AMPZ
        * BASEAMP
        * np.ones((NBPMS, 1))
        * np.cos(
            2 * np.pi * (MUZ * np.ones((NBPMS, 1)) + TUNEZ * np.ones((NBPMS, 1)) * ints[None, :])
        )
    )
    matrices = [
        tbt.TransverseData(
            X=pd.DataFrame(
                data=np.random.randn(model.index.size, nturns) * NOISE
                + data_x
                + COUPLING * data_y
                + data_z,
                index=model.index,
            ),
            Y=pd.DataFrame(
                data=np.random.randn(model.index.size, nturns) * NOISE + data_y + COUPLING * data_x,
                index=model.index,
            ),
        )
    ]
    tbt_data = tbt.TbtData(matrices=matrices, bunch_ids=[0], nturns=nturns)  # let date default
    tbt.write(dir_path / "test_file", tbt_data)


def _other(plane: str) -> str:
    return "X" if plane == "Y" else "Y"


def _couple(plane: str) -> str:
    return "10" if plane == "Y" else "01"


def _rms(a: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(a)))


def _diff(a: float, b: float) -> float:
    return a - b


def _rel_diff(a: float, b: float) -> float:
    return (a / b) - 1


def _angle_diff(a: float, b: float) -> float:
    ang = a - b
    return np.where(np.abs(ang) > 0.5, ang - np.sign(ang), ang)


@pytest.fixture()
def _test_file(tmp_path: Path) -> Path:
    return tmp_path / "test_file.sdds"


@pytest.fixture()
def _model_file(tmp_path: Path) -> Path:
    return tmp_path / "model.tfs"
