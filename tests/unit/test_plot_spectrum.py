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
            bpms=['BPM.10L1.B1', 'BPM.10L2.B1', 'unknown_bpm'],
            lines_manual=[dict(x=0.3, label="myline")],
            lines_nattune=None,
            stem_plot=True,
            stem_single_fig=False,
            waterfall_plot=True,
            show_plots=False,
            manual_style={},  # just to call the update line
        )
        assert len(os.listdir(_get_output_dir(out_dir, file_path))) == 3


def test_single_stem_plot(file_path):
    with tempfile.TemporaryDirectory() as out_dir:
        plot_spectrum(
            files=[file_path],
            output_dir=out_dir,
            bpms=['BPM.10L1.B1', 'BPM.10L2.B1', 'unknown_bpm'],
            stem_plot=True,
            stem_single_fig=True,
        )
        assert len(os.listdir(_get_output_dir(out_dir, file_path))) == 1


def test_crash_no_plot_selected():
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as out_dir:
            plot_spectrum(
                files=['test'],
                output_dir=out_dir,
            )


def test_crash_too_low_amplimit():
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as out_dir:
            plot_spectrum(
                files=['test'],
                output_dir=out_dir,
                stem_plot=True,
                amp_limit=-1.,
            )


def test_crash_file_not_found_amplimit():
    with pytest.raises(FileNotFoundError):
        with tempfile.TemporaryDirectory() as out_dir:
            plot_spectrum(
                files=['test'],
                output_dir=out_dir,
                stem_plot=True,
            )


def _get_output_dir(out_dir, file_path):
    return os.path.join(out_dir, os.path.splitext(os.path.basename(file_path))[0])


@pytest.fixture
def file_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "inputs", 'spec_test.sdds'))

