from pathlib import Path
import tempfile
from omc3.amplitude_detuning_analysis import analyse_with_bbq_corrections


class BasicTest:
    @staticmethod
    def test_amplitude_detuning_fulltest():
        with tempfile.TemporaryDirectory() as out:
            setup = dict(
                beam=1,
                kick=str(get_input_dir()),
                plane="Y",
                label="B1Vkicks",
                timber_in=str(get_input_dir().joinpath("bbq_ampdet.tfs")),
                detuning_order=1,
                output=out,
                window_length=200,
                tune_x=0.2838,
                tune_y=0.3104,
                tune_cut=0.001,
                fine_window=100,
                fine_cut=0.0002,
            )
            kick_df, bbq_df = analyse_with_bbq_corrections(**setup)

            assert len(list(Path(out).glob("*.tfs"))) == 2
            assert len([k for k, v in kick_df.headers.items() if k.startswith("ODR") and v != 0]) == 16


def get_input_dir():
    return Path(__file__).parent.parent.joinpath('inputs', 'amplitude_detuning')