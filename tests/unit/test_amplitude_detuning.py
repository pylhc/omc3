from pathlib import Path

import pytest

from omc3.amplitude_detuning_analysis import analyse_with_bbq_corrections
from omc3.plotting.plot_bbq import main as pltbbq


@pytest.mark.basic
def test_amplitude_detuning_outliers_filter(tmp_path):
    test_amplitude_detuning_full(tmp_path=tmp_path, method="outliers")


@pytest.mark.basic
def test_bbq_plot(tmp_path):
    fig = pltbbq(
        input=get_input_dir() / "bbq_ampdet.tfs", output=tmp_path / "bbq.pdf",
    )
    assert fig is not None
    assert len(list(tmp_path.glob("*.pdf"))) == 1


@pytest.mark.extended
@pytest.mark.parametrize("method", ["cut", "minmax"])
def test_amplitude_detuning_full(tmp_path, method):
    setup = dict(
        beam=1,
        kick=get_input_dir(),
        plane="Y",
        label="B1Vkicks",
        bbq_in=get_input_dir() / "bbq_ampdet.tfs",
        detuning_order=1,
        output=tmp_path,
        window_length=100 if method != "outliers" else 50,
        tunes=[0.2838, 0.3104],
        tune_cut=0.001,
        tunes_minmax=[0.2828, 0.2848, 0.3094, 0.3114],
        fine_window=50,
        fine_cut=4e-4,
        outlier_limit=1e-4,
        bbq_filtering_method=method,
    )
    kick_df, bbq_df = analyse_with_bbq_corrections(**setup)

    assert len(list(tmp_path.glob("*.tfs"))) == 2
    assert len([k for k, v in kick_df.headers.items() if k.startswith("ODR") and v != 0]) == 16


@pytest.mark.extended
def test_no_bbq_input(tmp_path):
    setup = dict(
        beam=1,
        kick=get_input_dir(),
        plane="Y",
        label="B1Vkicks",
        detuning_order=1,
        output=tmp_path,
    )
    kick_df, bbq_df = analyse_with_bbq_corrections(**setup)

    assert bbq_df is None
    assert len(list(tmp_path.glob("*.tfs"))) == 1
    assert len([k for k, v in kick_df.headers.items() if k.startswith("ODR") and v != 0]) == 8


def get_input_dir():
    return Path(__file__).parent.parent / "inputs" / "amplitude_detuning"
