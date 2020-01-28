import tempfile
from glob import glob
from shutil import copy
from os import listdir
from os.path import join, abspath, basename, dirname, splitext

import pytest
import tfs
from matplotlib.figure import Figure

from omc3.plot_spectrum import main as plot_spectrum
from omc3.plotting.spectrum_utils import get_unique_filenames


def test_unique_filenames():
    def _test_list(list_of_paths):
        paths, names = zip(*get_unique_filenames(list_of_paths))
        for item in zip(list_of_paths, paths):
            assert item[0] == item[1]
        assert len(set(names)) == len(names)
        assert len(names) == len(list_of_paths)
        return names
    names = _test_list([join('mozart', 'wolfgang'), join('petri', 'wolfgang'), join('frisch', 'max')])
    assert "mozart_wolfgang" in names
    assert "frisch_max" in names

    names = _test_list([join('mozart', 'wolfgang'), join('petri', 'heil'), join('frisch', 'max')])
    assert "wolfgang" in names
    assert "max" in names


def test_basic_functionality(file_path, bpms):
    with tempfile.TemporaryDirectory() as out_dir:
        stem, waterfall = plot_spectrum(
            plot_type=['stem', 'waterfall'],
            files=[file_path],
            output_dir=out_dir,
            bpms=bpms + ['unknown_bpm'],
            lines_manual=[dict(x=0.3, label="myline")],
            lines_nattunes=None,
            show_plots=False,
            manual_style={},  # just to call the update line
        )
        _, filename = list(get_unique_filenames([file_path]))[0]
        bpm_ids = (f"{filename}_{bpm}" for bpm in bpms)
        assert len(listdir(_get_output_dir(out_dir, file_path))) == 3
        assert len(waterfall) == 1
        assert (filename in waterfall) and (waterfall[filename] is not None)
        assert len(stem) == len(bpms)
        assert all((bpm in stem) and (stem[bpm] is not None)
                   for bpm in bpm_ids)


def test_combined_bpms_stem_plot(file_path, bpms):
    with tempfile.TemporaryDirectory() as out_dir:
        stem, waterfall = plot_spectrum(
            files=[file_path],
            output_dir=out_dir,
            bpms=bpms + ['unknown_bpm'],
            lines_manual=[{'x': 0.44, 'loc': "top"}],
            combine_by=['bpms'],
        )
        _, filename = list(get_unique_filenames([file_path]))[0]
        assert len(listdir(_get_output_dir(out_dir, file_path))) == 1
        assert len(waterfall) == 0
        assert len(stem) == 1
        assert (filename in stem) and isinstance(stem[filename], Figure)


def test_no_tunes_in_files_plot(file_path, bpms):
    with tempfile.TemporaryDirectory() as out_dir:
        for f in glob(f'{file_path}*'):
            copy(f, out_dir)
        file_path = join(out_dir, basename(file_path))
        for p in ('x', 'y'):
            fname = f'{file_path}.lin{p}'
            df = tfs.read(fname)
            tfs.write(fname,
                      df.drop(columns=[f'TUNE{p.upper()}',
                                       f'NATTUNE{p.upper()}']))
        plot_spectrum(
            files=[file_path],
            bpms=bpms,
            combine_by=['files', 'bpms'],
        )


def test_crash_too_low_amplimit():
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as out_dir:
            plot_spectrum(
                files=['test'],
                output_dir=out_dir,
                amp_limit=-1.,
            )


def test_crash_file_not_found_amplimit():
    with pytest.raises(FileNotFoundError):
        with tempfile.TemporaryDirectory() as out_dir:
            plot_spectrum(
                files=['test'],
                output_dir=out_dir,
            )


def _get_output_dir(out_dir, file_path):
    return join(out_dir, splitext(basename(file_path))[0])


@pytest.fixture
def file_path():
    return abspath(join(dirname(__file__), "..", "inputs", 'spec_test.sdds'))


@pytest.fixture
def bpms():
    return ['BPM.10L1.B1', 'BPM.10L2.B1']
