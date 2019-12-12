import os
import pytest
import numpy as np
import pandas as pd
from . import context
from plot_spectrum import main as plot_spectrum


import tempfile


def test_basic_functionality(file_path):
    with tempfile.TemporaryDirectory() as out_dir:
        plot_spectrum(
            files=[file_path],
            output_dir=out_dir,
            bpms=['BPM.10L1.B1'],
            # lines_tune=[],
            # lines_nattune=[],
            # lines_manual=[dict(x=0.32, label='mytune', color='k', loc="line top")],
            stem_plot=True,
            # rescale=True,
            stem_single_fig=True,
            waterfall_plot=True,
            # waterfall_line_width=1,
            # hide_bpm_labels=True,
            show_plots=True,
        )
        file_out = os.path.join(out_dir, os.path.basename(file_path))
        assert len(os.listdir(file_out)) == 2


@pytest.fixture
def file_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "inputs", 'spec_test.sdds'))

