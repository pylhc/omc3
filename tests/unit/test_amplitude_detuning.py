from pathlib import Path
import tempfile

import pytest

from omc3.amplitude_detuning_analysis import analyse_with_bbq_corrections
from omc3.plotting.plot_bbq import main as pltbbq
from omc3.plotting.plot_amplitude_detuning import main as pltampdet


class BasicTests:
    @staticmethod
    def test_amplitude_detuning_outliers_filter():
        ExtendedTests().test_amplitude_detuning_full(method='outliers')

    @staticmethod
    def test_bbq_plot():
        with tempfile.TemporaryDirectory() as out:
            fig = pltbbq(
                input=str(get_input_dir() / 'bbq_ampdet.tfs'),
                output=str(Path(out) / 'bbq.pdf'),
            )
            assert fig is not None
            assert len(list(Path(out).glob("*.pdf"))) == 1

    @staticmethod
    def test_ampdet_plot():
        with tempfile.TemporaryDirectory() as out:
            fig = pltampdet(
                kicks=[str(get_input_dir() / 'kick_ampdet_xy.tfs')],
                labels=['Beam 1 Vertical'],
                plane='Y',
                correct_acd=True,
                output=str(Path(out) / 'ampdet.pdf'),
            )
            assert len(fig) == 4
            assert len(list(Path(out).glob("*.pdf"))) == 4


class ExtendedTests:
    @staticmethod
    @pytest.mark.parametrize("method",  ['cut', 'minmax'])
    def test_amplitude_detuning_full(method):
        with tempfile.TemporaryDirectory() as out:
            setup = dict(
                beam=1,
                kick=str(get_input_dir()),
                plane="Y",
                label="B1Vkicks",
                bbq_in=str(get_input_dir() / "bbq_ampdet.tfs"),
                detuning_order=1,
                output=out,
                window_length=100 if method != 'outliers' else 50,
                tunes=[0.2838, 0.3104],
                tune_cut=0.001,
                tunes_minmax=[0.2828, 0.2848, 0.3094, 0.3114],
                fine_window=50,
                fine_cut=4e-4,
                outlier_limit=1e-4,
                bbq_filtering_method=method,
            )
            kick_df, bbq_df = analyse_with_bbq_corrections(**setup)

            assert len(list(Path(out).glob("*.tfs"))) == 2
            assert len([k for k, v in kick_df.headers.items() if k.startswith("ODR") and v != 0]) == 16


def get_input_dir():
    return Path(__file__).parent.parent / 'inputs' / 'amplitude_detuning'
