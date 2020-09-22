import inspect
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import pytest

from omc3.plotting.plot_optics_measurements import plot

# Figure (capital F) does not handle the canvas for us so we need to force the backend to make sure
# tests know what to do on every platform
# Forcing non-interactive Agg backend so rendering is done similarly across platforms during tests
matplotlib.use("Agg")

INPUT = Path(__file__).parent.parent / 'inputs' / 'optics_measurement' / 'example_output'
DEBUG = False  # switch to local output instead of temp


@pytest.mark.basic
def test_phase():
    _default_test('phase')


@pytest.mark.basic
def test_phase_delta():
    _default_test('phase', delta=True)


@pytest.mark.basic
def test_rdt_0040():
    _default_test_rdt('f0040_y')


@pytest.mark.extended
def test_orbit():
    figs = _default_test('orbit')
    for fig in figs.values():
        assert len(fig.axes) == 2


@pytest.mark.extended
def test_orbit_delta():
    figs = _default_test('orbit', delta=True)
    for fig in figs.values():
        assert len(fig.axes) == 2


@pytest.mark.extended
def test_orbit_combine_planes():
    figs = _default_test('orbit', combine_by=['planes'])
    for fig in figs.values():
        assert len(fig.axes) == 1


@pytest.mark.extended
def test_orbit_share_xaxis():
    figs = _default_test('orbit', share_xaxis=True)
    for fig in figs.values():
        assert len(fig.axes) == 2
        sharedx = list(fig.axes[0].get_shared_x_axes())
        assert len(sharedx) == 1
        assert len(sharedx[0]) == 2


@pytest.mark.extended
def test_orbit_ip_positions_location():
    _default_test('orbit', x_axis='location', ip_positions='LHCB1')


@pytest.mark.extended
def test_orbit_ip_positions_location_manual():
    _default_test('orbit', x_axis='location',
                  ip_positions=INPUT.parent.parent / 'models' / '25cm_beam1' / 'twiss_elements.dat')


@pytest.mark.extended
def test_orbit_ip_positions_phase():
    with pytest.raises(NotImplementedError):  # remove once implemented
        _default_test('orbit', x_axis='phase-advance', ip_positions='LHCB1')


@pytest.mark.extended
def test_two_directories():
    with _output_dir() as out_dir:
        figs = plot(
            show=False,
            x_axis='phase-advance',
            ncol_legend=2,
            folders=[str(INPUT), str(INPUT.parent / 'example_copy')],
            output=str(out_dir),
            optics_parameters=['orbit', 'beta_phase'],
        )
        assert len(list(out_dir.glob('*.pdf'))) == 4
        assert len(figs) == 4


@pytest.mark.extended
def test_two_directories_combined():
    with _output_dir() as out_dir:
        figs = plot(
            show=False,
            x_axis='phase-advance',
            ncol_legend=2,
            folders=[str(INPUT), str(INPUT.parent / 'example_copy')],
            output=str(out_dir),
            optics_parameters=['orbit', 'beta_phase'],
            combine_by=['files']
        )
        assert len(list(out_dir.glob('*.pdf'))) == 2
        assert len(figs) == 2


@pytest.mark.extended
def test_beta_phase():
    _default_test('beta_phase')


@pytest.mark.extended
def test_beta_phase_delta():
    _default_test('beta_phase', delta=True)


@pytest.mark.extended
def test_beta_amplitude():
    _default_test('beta_amplitude')


@pytest.mark.extended
def test_beta_amplitude_delta():
    _default_test('beta_amplitude', delta=True)


@pytest.mark.extended
def test_total_phase():
    _default_test('total_phase')


@pytest.mark.extended
def test_total_phase_delta():
    _default_test('total_phase', delta=True)


@pytest.mark.extended
def test_rdt_1001():
    figs = _default_test_rdt('f1001_x')
    for fig in figs.values():
        assert len(fig.axes) == 2


@pytest.mark.extended
def test_rdt_1001_combine_planes():
    figs = _default_test_rdt('f1001_x', combine_by=['planes'])
    for fig in figs.values():
        assert len(fig.axes) == 1


@pytest.mark.extended
def test_rdt_0030():
    _default_test_rdt('f0030_y')


@pytest.mark.extended
def test_rdt_1002():
    _default_test_rdt('f1002_x')


@pytest.mark.extended
def test_rdt_1001_skip_for_xaxis_option():
    with pytest.raises(AssertionError):  # assertion fails as there are no plots
        _default_test_rdt('f1001_x', x_axis='phase-advance')


# Helper -----------------------------------------------------------------------


def _get_test_name():
    for s in inspect.stack():
        if s.function.startswith('test_'):
            return s.function
    raise AttributeError('Needs to be called downstream of a "test_" function')


@contextmanager
def _output_dir():
    if DEBUG:
        path = Path(f'temp_{_get_test_name()}')
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()
        yield path
    else:
        with tempfile.TemporaryDirectory() as dir_:
            yield Path(dir_)


def _default_test(*args, **kwargs):
    default_args = dict(
        show=False,
        x_axis='phase-advance',
        ncol_legend=2,
    )
    default_args.update(kwargs)
    with _output_dir() as out_dir:
        figs = plot(
            folders=[INPUT, ],
            output=out_dir,
            optics_parameters=list(args),
            **default_args
        )
        assert len(list(out_dir.glob('*.pdf'))) == len(args)
        assert len(figs) == len(args)
    return figs


def _default_test_rdt(*args, **kwargs):
    default_args = dict(
        show=False,
        x_axis='location',
        ncol_legend=2,
    )
    default_args.update(kwargs)
    with _output_dir() as out_dir:
        figs = plot(
            folders=[str(INPUT), ],
            output=str(out_dir),
            optics_parameters=list(args),
            **default_args
        )
        assert len(list(out_dir.glob('*.pdf'))) == 2*len(args)
        assert len(figs) == 2*len(args)
    return figs
