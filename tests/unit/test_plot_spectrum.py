from pathlib import Path
from shutil import copy

import matplotlib
import pytest
import tfs
from matplotlib.figure import Figure

from omc3.plotting.plot_spectrum import main as plot_spectrum
from omc3.plotting.spectrum.utils import PLANES, get_unique_filenames

INPUT_DIR = Path(__file__).parent.parent / "inputs"
# Forcing non-interactive Agg backend so rendering is done similarly across platforms during tests
matplotlib.use("Agg")


@pytest.mark.basic
def test_unique_filenames():
    def _test_list(list_of_paths):
        paths, names = zip(*get_unique_filenames(list_of_paths))
        names = ["_".join(n) for n in names]
        for item in zip(list_of_paths, paths):
            assert item[0] == item[1]
        assert len(set(names)) == len(names)
        assert len(names) == len(list_of_paths)
        return names

    names = _test_list(
        [Path("mozart", "wolfgang"), Path("petri", "wolfgang"), Path("frisch", "max")]
    )
    assert "mozart_wolfgang" in names
    assert "frisch_max" in names

    names = _test_list([Path("mozart", "wolfgang"), Path("petri", "heil"), Path("frisch", "max")])
    assert "wolfgang" in names
    assert "max" in names


@pytest.mark.basic
def test_basic_functionality(tmp_output_dir, file_path, bpms):
    stem, waterfall = plot_spectrum(
        plot_type=["stem", "waterfall"],
        files=[file_path],
        output_dir=str(tmp_output_dir),
        bpms=bpms + ["unknown_bpm"],
        lines_manual=[dict(x=0.3, label="myline")],
        lines_nattunes=None,
        show_plots=False,
        manual_style={},  # just to call the update line
    )
    _, filename = list(get_unique_filenames([file_path]))[0]
    filename = "_".join(filename)
    bpm_ids = (f"{filename}_{bpm}" for bpm in bpms)
    assert len(list(_get_output_dir(tmp_output_dir, file_path).iterdir())) == 3
    assert len(waterfall) == 1
    assert (filename in waterfall) and (waterfall[filename] is not None)
    assert len(stem) == len(bpms)
    assert all((bpm in stem) and (stem[bpm] is not None) for bpm in bpm_ids)


@pytest.mark.basic
def test_combined_bpms_stem_plot(tmp_output_dir, file_path, bpms):
    stem, waterfall = plot_spectrum(
        files=[file_path],
        output_dir=str(tmp_output_dir),
        bpms=bpms + ["unknown_bpm"],
        lines_manual=[{"x": 0.44, "loc": "top"}],
        combine_by=["bpms"],
    )
    _, filename = list(get_unique_filenames([file_path]))[0]
    filename = "_".join(filename)
    assert len(list(_get_output_dir(tmp_output_dir, file_path).iterdir())) == 1
    assert len(waterfall) == 0
    assert len(stem) == 1
    assert (filename in stem) and isinstance(stem[filename], Figure)


@pytest.mark.basic
def test_no_tunes_in_files_plot(tmp_output_dir, file_path, bpms):
    from glob import glob

    for f in glob(f"{file_path}*"):
        print(f)
    for f in INPUT_DIR.glob(f"{file_path.name}*"):
        copy(f, tmp_output_dir)
    file_path = tmp_output_dir / file_path.name
    for p in PLANES:
        fname = file_path.with_suffix(f"{file_path.suffix}.lin{p.lower()}")
        df = tfs.read(fname)
        tfs.write(fname, df.drop(columns=[f"TUNE{p.upper()}", f"NATTUNE{p.upper()}"]))
    plot_spectrum(
        files=[file_path], bpms=bpms, combine_by=["files", "bpms"],
    )


@pytest.mark.basic
def test_crash_too_low_amplimit(tmp_output_dir):
    with pytest.raises(ValueError):
        plot_spectrum(
            files=["test"], output_dir=str(tmp_output_dir), amp_limit=-1.0,
        )


@pytest.mark.basic
def test_crash_file_not_found_amplimit(tmp_output_dir):
    with pytest.raises(FileNotFoundError):
        plot_spectrum(
            files=["test"], output_dir=str(tmp_output_dir),
        )


def _get_output_dir(tmp_output_dir, file_path):
    return tmp_output_dir / file_path.with_suffix("").name


@pytest.fixture
def file_path():
    return INPUT_DIR / "spec_test.sdds"


@pytest.fixture
def bpms():
    return ["BPM.10L1.B1", "BPM.10L2.B1"]
