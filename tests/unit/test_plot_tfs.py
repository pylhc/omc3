import inspect
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from omc3.plotting.plot_tfs import plot

INPUT = Path(__file__).parent.parent / "inputs" / "optics_measurement" / "example_output"
DEBUG = False  # switch to local output instead of temp


# Basic Tests are tested with plot_optics_measurements

# Usage Examples ---
@pytest.mark.extended
def test_simple_plot_manual_planes_same_file(tmp_output_dir):
    figs = plot(
        files=[str(INPUT / "beta_phase_{0}.tfs"), str(INPUT / "beta_amplitude_{0}.tfs")],
        same_figure="planes",
        same_axes=["files"],
        x_columns=["S"],
        y_columns=["BET{0}"],
        error_columns=None,
        planes=["X", "Y"],
        x_labels=["Location [m]"],
        file_labels=[r"$\beta$ from phase", r"$\beta$ from amplitude"],
        # column_labels=[r'$\beta_{0}$'],  # would have correct axes-labels but also bx in legend
        column_labels=[""],  # removes BETX BETY from legend-names
        y_labels=[
            [r"$\beta_x$", r"$\beta_y$"]
        ],  # manual axes labels (outer = figures, inner = axes)
        output=tmp_output_dir,
        show=False,
        single_legend=True,
        change_marker=True,
    )
    assert len(figs) == 1
    for fig in figs.values():
        assert len(fig.axes) == 2


@pytest.mark.extended
def test_simple_plot_manual_planes_two_files(tmp_output_dir):
    figs = plot(
        files=[str(INPUT / "beta_phase_{0}.tfs"), str(INPUT / "beta_amplitude_{0}.tfs")],
        same_axes=["files"],
        x_columns=["S"],
        y_columns=["BET{0}"],
        error_columns=None,
        planes=["X", "Y"],
        x_labels=["Location [m]"],
        file_labels=[r"$\beta$ from phase", r"$\beta$ from amplitude"],
        # column_labels=[r'$\beta_{0}$'],  # would have correct axes-labels but also bx in legend
        column_labels=[""],  # removes BETX BETY from legend-names
        y_labels=[
            [r"$\beta_x$"],
            [r"$\beta_y$"],
        ],  # manual axes labels (outer = figures, inner = axes)
        output=tmp_output_dir,
        show=False,
        single_legend=True,
        change_marker=True,
    )
    assert len(figs) == 2
    for fig in figs.values():
        assert len(fig.axes) == 1


# Simple Tests ---
@pytest.mark.extended
def test_simple_plot(tmp_output_dir):
    figs = simple_plot_tfs(output=tmp_output_dir)
    assert len(figs) == 2
    for fig in figs.values():
        assert len(fig.axes) == 1


@pytest.mark.extended
def test_simple_plot_same_figure(tmp_output_dir):
    figs = simple_plot_tfs(output=tmp_output_dir, same_figure="planes")
    assert len(figs) == 1
    for fig in figs.values():
        assert len(fig.axes) == 2


@pytest.mark.extended
def test_simple_plot_same_axes(tmp_output_dir):
    figs = simple_plot_tfs(output=tmp_output_dir, same_axes=["planes"])
    assert len(figs) == 1
    for fig in figs.values():
        assert len(fig.axes) == 1


# check for wrong input ---
@pytest.mark.extended
def test_errors_too_many_filelabels(tmp_output_dir):
    with pytest.raises(AttributeError):
        simple_plot_tfs(output=tmp_output_dir, file_labels=["label1", "label2"])


@pytest.mark.extended
def test_errors_too_many_xcolumns(tmp_output_dir):
    with pytest.raises(AttributeError):
        simple_plot_tfs(output=tmp_output_dir, x_columns=["A", "B"])


@pytest.mark.extended
def test_errors_too_many_errorcolumns(tmp_output_dir):
    with pytest.raises(AttributeError):
        simple_plot_tfs(output=tmp_output_dir, error_columns=["A", "B"])


@pytest.mark.extended
def test_errors_too_many_columnlabels(tmp_output_dir):
    with pytest.raises(AttributeError):
        simple_plot_tfs(output=tmp_output_dir, column_labels=["label1", "label2"])


@pytest.mark.extended
def test_errors_same_options_same(tmp_output_dir):
    with pytest.raises(AttributeError):
        simple_plot_tfs(output=tmp_output_dir, same_axes=["planes"], same_figure="planes")


# Main plot (can be also used as example) ---


def simple_plot_tfs(**kwargs):
    default_args = dict(
        files=[INPUT / "orbit_{0}.tfs"],
        x_columns=["S"],
        y_columns=["{0}"],
        error_columns=["ERR{0}"],
        planes=["X", "Y"],
        show=False,
        # same_axes='planes',
        # same_figure='planes',
        single_legend=True,
        change_marker=True,
    )
    default_args.update(kwargs)
    return plot(**default_args)
