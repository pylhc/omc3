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
def test_freekick_harpy(_test_file: Path, _model_file: Path):
    model_df = _get_model_dataframe()
    tfs.write(_model_file, model_df, save_index="NAME")
    _write_tbt_file(model_df, _test_file.parent)
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
            _rms(
                _diff(
                    linfiles[plane].loc[:, f"TUNE{plane}"].to_numpy(),
                    model_df.loc[:, f"TUNE{plane}"].to_numpy(),
                )
            )
            < LIMITS["F1"]
        )
        # Main and secondary amplitudes
        assert (
            _rms(
                _rel_diff(
                    linfiles[plane].loc[:, f"AMP{plane}"].to_numpy(),
                    model_df.loc[:, f"AMP{plane}"].to_numpy(),
                )
            )
            < LIMITS["A1"]
        )
        # Main and secondary phases
        assert (
            _rms(
                _angle_diff(
                    linfiles[plane].loc[:, f"MU{plane}"].to_numpy(),
                    model_df.loc[:, f"MU{plane}"].to_numpy(),
                )
            )
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
    assert _rms(_diff(linfiles["X"].loc[:, "TUNEZ"].to_numpy(), TUNEZ)) < LIMITS["F2"]
    assert (
        _rms(
            _rel_diff(
                linfiles["X"].loc[:, "AMPZ"].to_numpy()
                * linfiles["X"].loc[:, "AMPX"].to_numpy(),
                AMPZ * BASEAMP,
            )
        )
        < LIMITS["A2"]
    )
    assert _rms(_angle_diff(linfiles["X"].loc[:, "MUZ"].to_numpy(), MUZ)) < LIMITS["P2"]


def _assert_spectra(linfiles: dict[str, tfs.TfsDataFrame], model_df: tfs.TfsDataFrame):
    for plane in PLANES:
        # main and secondary frequencies
        assert (
            _rms(
                _diff(
                    linfiles[plane].loc[:, f"TUNE{plane}"].to_numpy(),
                    model_df.loc[:, f"TUNE{plane}"].to_numpy(),
                )
            )
            < LIMITS["F1"]
        )
        assert (
            _rms(
                _diff(
                    linfiles[plane].loc[:, f"FREQ{_couple(plane)}"].to_numpy(),
                    model_df.loc[:, f"TUNE{_other(plane)}"].to_numpy(),
                )
            )
            < LIMITS["F2"]
        )
        # main and secondary amplitudes
        assert (
            _rms(
                _rel_diff(
                    linfiles[plane].loc[:, f"AMP{plane}"].to_numpy(),
                    model_df.loc[:, f"AMP{plane}"].to_numpy(),
                )
            )
            < LIMITS["A1"]
        )
        assert (
            _rms(
                _rel_diff(
                    linfiles[plane].loc[:, f"AMP{_couple(plane)}"].to_numpy()
                    * linfiles[plane].loc[:, f"AMP{plane}"].to_numpy(),
                    COUPLING * model_df.loc[:, f"AMP{_other(plane)}"].to_numpy(),
                )
            )
            < LIMITS["A2"]
        )
        # main and secondary phases
        assert (
            _rms(
                _angle_diff(
                    linfiles[plane].loc[:, f"MU{plane}"].to_numpy(),
                    model_df.loc[:, f"MU{plane}"].to_numpy(),
                )
            )
            < LIMITS["P1"]
        )
        assert (
            _rms(
                _angle_diff(
                    linfiles[plane].loc[:, f"PHASE{_couple(plane)}"].to_numpy(),
                    model_df.loc[:, f"MU{_other(plane)}"].to_numpy(),
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


def _write_tbt_file(model: tfs.TfsDataFrame, dir_path: Path) -> None:
    ints = np.arange(NTURNS) - NTURNS / 2
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
                data=np.random.randn(model.index.size, NTURNS) * NOISE
                + data_x
                + COUPLING * data_y
                + data_z,
                index=model.index,
            ),
            Y=pd.DataFrame(
                data=np.random.randn(model.index.size, NTURNS) * NOISE + data_y + COUPLING * data_x,
                index=model.index,
            ),
        )
    ]
    tbt_data = tbt.TbtData(matrices=matrices, bunch_ids=[0], nturns=NTURNS)  # let date default
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
