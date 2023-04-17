from pathlib import Path

import matplotlib
import pytest

from omc3.plotting.plot_amplitude_detuning import main as pltampdet

# Forcing non-interactive Agg backend so rendering is done similarly across platforms during tests
matplotlib.use("Agg")


@pytest.mark.basic
def test_ampdet_plot(tmp_path):
    fig = pltampdet(
        kicks=[str(get_input_dir() / "kick_ampdet_xy.tfs")],
        labels=["Beam 1 Vertical"],
        plane="Y",
        correct_acd=True,
        output=tmp_path / "ampdet.pdf",
    )
    assert len(fig) == 4
    assert len(list(tmp_path.glob("*.pdf"))) == 4


@pytest.mark.basic
def test_ampdet_plot3d(tmp_path):
    fig = pltampdet(
        kicks=[get_2d_input_dir() / "kick_ampdet_xy.tfs"],
        labels=["Beam 1"],
        plane="3D",
        correct_acd=True,
        output=tmp_path / "ampdet.pdf",
    )
    assert len(fig) == 4
    assert len(list(tmp_path.glob("*.pdf"))) == 4


@pytest.mark.basic
def test_ampdet_plot_2danalysis(tmp_path):
    fig = pltampdet(
        kicks=[get_2d_input_dir() / "kick_ampdet_xy.tfs"],
        labels=["Beam 1"],
        plane="XY",
        correct_acd=True,
        output=tmp_path / "ampdet.pdf",
    )
    assert len(fig) == 8
    assert len(list(tmp_path.glob("*.pdf"))) == 8


def get_input_dir():
    return Path(__file__).parent.parent / "inputs" / "amplitude_detuning"


def get_2d_input_dir():
    return Path(__file__).parent.parent / "inputs" / "amplitude_detuning_2d"
