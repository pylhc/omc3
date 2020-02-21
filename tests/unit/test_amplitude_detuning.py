from pathlib import Path
import tempfile

import pytest

from omc3.amplitude_detuning_analysis import analyse_with_bbq_corrections


class BasicTests:
    @staticmethod
    def test_amplitude_detuning_outliers_filter():
        ExtendedTests().test_amplitude_detuning_full(method='outliers')


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
                timber_in=str(get_input_dir().joinpath("bbq_ampdet.tfs")),
                detuning_order=1,
                output=out,
                window_length=100 if method != 'outliers' else 50,
                tune_x=0.2838,
                tune_y=0.3104,
                tune_cut=0.001,
                tune_x_min=0.2828,
                tune_x_max=0.2848,
                tune_y_min=0.3094,
                tune_y_max=0.3114,
                fine_window=50,
                fine_cut=4e-4,
                outlier_limit=1e-4,
                bbq_filtering_method=method,
            )
            kick_df, bbq_df = analyse_with_bbq_corrections(**setup)

            assert len(list(Path(out).glob("*.tfs"))) == 2
            assert len([k for k, v in kick_df.headers.items() if k.startswith("ODR") and v != 0]) == 16


def get_input_dir():
    return Path(__file__).parent.parent.joinpath('inputs', 'amplitude_detuning')