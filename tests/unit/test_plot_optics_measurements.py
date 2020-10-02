from pathlib import Path

import pytest

from omc3.plotting.plot_optics_measurements import plot

INPUT = Path(__file__).parent.parent / 'inputs' / 'optics_measurement' / 'example_output'
DEBUG = False  # switch to local output instead of temp


@pytest.mark.basic
def test_phase(tmp_output_dir):
    _default_test('phase', output_dir=tmp_output_dir)


@pytest.mark.basic
def test_phase_delta(tmp_output_dir):
    _default_test('phase', delta=True, output_dir=tmp_output_dir)


@pytest.mark.basic
def test_rdt_0040(tmp_output_dir):
    _default_test_rdt('f0040_y', output_dir=tmp_output_dir)


@pytest.mark.extended
def test_orbit(tmp_output_dir):
    figs = _default_test('orbit', output_dir=tmp_output_dir)
    for fig in figs.values():
        assert len(fig.axes) == 2


@pytest.mark.extended
def test_orbit_delta(tmp_output_dir):
    figs = _default_test('orbit', delta=True, output_dir=tmp_output_dir)
    for fig in figs.values():
        assert len(fig.axes) == 2


@pytest.mark.extended
def test_orbit_combine_planes(tmp_output_dir):
    figs = _default_test('orbit', combine_by=['planes'], output_dir=tmp_output_dir)
    for fig in figs.values():
        assert len(fig.axes) == 1


@pytest.mark.extended
def test_orbit_share_xaxis(tmp_output_dir):
    figs = _default_test('orbit', share_xaxis=True, output_dir=tmp_output_dir)
    for fig in figs.values():
        assert len(fig.axes) == 2
        sharedx = list(fig.axes[0].get_shared_x_axes())
        assert len(sharedx) == 1
        assert len(sharedx[0]) == 2


@pytest.mark.extended
def test_orbit_ip_positions_location(tmp_output_dir):
    _default_test('orbit', x_axis='location', ip_positions='LHCB1',
                  output_dir=tmp_output_dir)


@pytest.mark.extended
def test_orbit_ip_positions_location_manual(tmp_output_dir):
    _default_test('orbit', x_axis='location',
                  ip_positions=INPUT.parent.parent / 'models' / '25cm_beam1' / 'twiss_elements.dat',
                  output_dir=tmp_output_dir)


@pytest.mark.extended
def test_orbit_ip_positions_phase(tmp_output_dir):
    with pytest.raises(NotImplementedError):  # remove once implemented
        _default_test('orbit', x_axis='phase-advance', ip_positions='LHCB1', output_dir=tmp_output_dir)


@pytest.mark.extended
def test_two_directories(tmp_output_dir):
    figs = plot(
        show=False,
        x_axis='phase-advance',
        ncol_legend=2,
        folders=[str(INPUT), str(INPUT.parent / 'example_copy')],
        output=str(tmp_output_dir),
        optics_parameters=['orbit', 'beta_phase'],
    )
    assert len(list(tmp_output_dir.glob('*.pdf'))) == 4
    assert len(figs) == 4


@pytest.mark.extended
def test_two_directories_combined(tmp_output_dir):
    figs = plot(
        show=False,
        x_axis='phase-advance',
        ncol_legend=2,
        folders=[str(INPUT), str(INPUT.parent / 'example_copy')],
        output=str(tmp_output_dir),
        optics_parameters=['orbit', 'beta_phase'],
        combine_by=['files']
    )
    assert len(list(tmp_output_dir.glob('*.pdf'))) == 2
    assert len(figs) == 2


@pytest.mark.extended
def test_beta_phase(tmp_output_dir):
    _default_test('beta_phase', output_dir=tmp_output_dir)


@pytest.mark.extended
def test_beta_phase_delta(tmp_output_dir):
    _default_test('beta_phase', delta=True, output_dir=tmp_output_dir)


@pytest.mark.extended
def test_beta_amplitude(tmp_output_dir):
    _default_test('beta_amplitude', output_dir=tmp_output_dir)


@pytest.mark.extended
def test_beta_amplitude_delta(tmp_output_dir):
    _default_test('beta_amplitude', delta=True, output_dir=tmp_output_dir)


@pytest.mark.extended
def test_total_phase(tmp_output_dir):
    _default_test('total_phase', output_dir=tmp_output_dir)


@pytest.mark.extended
def test_total_phase_delta(tmp_output_dir):
    _default_test('total_phase', delta=True, output_dir=tmp_output_dir)


@pytest.mark.extended
def test_rdt_1001(tmp_output_dir):
    figs = _default_test_rdt('f1001_x', output_dir=tmp_output_dir)
    for fig in figs.values():
        assert len(fig.axes) == 2


@pytest.mark.extended
def test_rdt_1001_combine_planes(tmp_output_dir):
    figs = _default_test_rdt('f1001_x', combine_by=['planes'], output_dir=tmp_output_dir)
    for fig in figs.values():
        assert len(fig.axes) == 1


@pytest.mark.extended
def test_rdt_0030(tmp_output_dir):
    _default_test_rdt('f0030_y', output_dir=tmp_output_dir)


@pytest.mark.extended
def test_rdt_1002(tmp_output_dir):
    _default_test_rdt('f1002_x', output_dir=tmp_output_dir)


@pytest.mark.extended
def test_rdt_1001_skip_for_xaxis_option(tmp_output_dir):
    with pytest.raises(AssertionError):  # assertion fails as there are no plots
        _default_test_rdt('f1001_x', x_axis='phase-advance', output_dir=tmp_output_dir)


# Helper -----------------------------------------------------------------------


def _default_test(*args, **kwargs):
    out_dir = kwargs.pop('output_dir')
    default_args = dict(
        show=False,
        x_axis='phase-advance',
        ncol_legend=2,
    )
    default_args.update(kwargs)
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
    out_dir = kwargs.pop('output_dir')
    default_args = dict(
        show=False,
        x_axis='location',
        ncol_legend=2,
    )
    default_args.update(kwargs)
    figs = plot(
        folders=[str(INPUT), ],
        output=str(out_dir),
        optics_parameters=list(args),
        **default_args
    )
    assert len(list(out_dir.glob('*.pdf'))) == 2*len(args)
    assert len(figs) == 2*len(args)
    return figs
