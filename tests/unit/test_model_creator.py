import os
import shutil
from pathlib import Path

import pytest
from omc3.model.accelerators.accelerator import AcceleratorDefinitionError, AccExcitationMode
from omc3.model.constants import TWISS_AC_DAT, TWISS_ADT_DAT, TWISS_DAT, TWISS_ELEMENTS_DAT
from omc3.model.manager import get_accelerator
from omc3.model.model_creators.lhc_model_creator import LhcBestKnowledgeCreator, LhcModelCreator
from omc3.model_creator import create_instance_and_model

INPUTS = Path(__file__).parent.parent / "inputs"
COMP_MODEL = INPUTS / "models" / "25cm_beam1"
CODEBASE_PATH = Path(__file__).parent.parent.parent / "omc3"
PS_MODEL = CODEBASE_PATH / "model" / "accelerators" / "ps" / "2018" / "strength"


@pytest.mark.basic
def test_booster_creation_nominal(tmp_path):
    accel_opt = dict(
        accel="psbooster",
        ring=1,
        nat_tunes=[4.21, 4.27],
        drv_tunes=[0.205, 0.274],
        driven_excitation="acd",
        dpp=0.0,
        energy=0.16,
        modifiers=None,
    )
    accel = create_instance_and_model(
        type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["ring"])


@pytest.mark.basic
def test_ps_creation_nominal(tmp_path):
    accel_opt = dict(
        accel="ps",
        nat_tunes=[6.32, 6.29],
        drv_tunes=[0.325, 0.284],
        driven_excitation="acd",
        dpp=0.0,
        energy=1.4,
        modifiers=[PS_MODEL / "elements.str", PS_MODEL / "PS_LE_LHC_low_chroma.str"],
    )
    accel = create_instance_and_model(
        type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=[])


@pytest.mark.basic
def test_lhc_creation_nominal_driven(tmp_path):
    accel_opt = dict(
        accel="lhc",
        year="2018",
        ats=True,
        beam=1,
        nat_tunes=[0.31, 0.32],
        drv_tunes=[0.298, 0.335],
        driven_excitation="acd",
        dpp=0.0,
        energy=6.5,
        modifiers=[COMP_MODEL / "opticsfile.24_ctpps2"],
    )
    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["beam", "year"])


@pytest.mark.basic
def test_lhc_creation_nominal_free(tmp_path):
    accel_opt = dict(
        accel="lhc",
        year="2018",
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6.5,
        modifiers=[COMP_MODEL / "opticsfile.24_ctpps2"],
    )
    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["beam", "year"])


@pytest.mark.basic
def test_lhc_creation_best_knowledge(tmp_path):
    (tmp_path / LhcBestKnowledgeCreator.EXTRACTED_MQTS_FILENAME).write_text("\n")
    (tmp_path / LhcBestKnowledgeCreator.CORRECTIONS_FILENAME).write_text("\n")
    accel_opt = dict(
        accel="lhc",
        year="2018",
        ats=True,
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6.5,
        modifiers=[COMP_MODEL / "opticsfile.24_ctpps2"],
    )
    accel = create_instance_and_model(
        outputdir=tmp_path, type="best_knowledge", logfile=tmp_path / "madx_log.txt", **accel_opt
    )


@pytest.mark.basic
def test_lhc_creation_relative_modifier_path(tmp_path):
    accel_opt = dict(
        accel="lhc",
        year="2018",
        ats=True,
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6.5,
        modifiers=[Path("opticsfile.24_ctpps2")],
    )
    shutil.copy(COMP_MODEL / "opticsfile.24_ctpps2", tmp_path / "opticsfile.24_ctpps2")

    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["beam", "year"])


@pytest.mark.basic
def test_lhc_creation_modifier_nonexistent(tmp_path):
    accel_opt = dict(
        accel="lhc",
        year="2018",
        ats=True,
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6.5,
        modifiers=[COMP_MODEL / "opticsfile.non_existent"],
    )
    with pytest.raises(AcceleratorDefinitionError) as creation_error:
        create_instance_and_model(
            outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
        )
    assert "opticsfile.non_existent" in str(creation_error.value)


@pytest.mark.basic
@pytest.mark.timeout(60)  # madx might get stuck (seen on macos)
def test_lhc_creation_relative_modeldir_path(request, tmp_path):
    os.chdir(tmp_path)  # switch cwd to tmp_path
    model_dir_relpath = Path("test_model")
    model_dir_relpath.mkdir()

    optics_file_relpath = Path("opticsfile.24_ctpps2")
    shutil.copy(COMP_MODEL / optics_file_relpath, model_dir_relpath / optics_file_relpath)

    accel_opt = dict(
        accel="lhc",
        year="2018",
        ats=True,
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6.5,
        modifiers=[optics_file_relpath],
    )

    # sometimes create_instance_and_model seems to run but does not create twiss-files ...
    accel = create_instance_and_model(
        outputdir=model_dir_relpath, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )

    # ... which is then caught here:
    check_accel_from_dir_vs_options(
        model_dir_relpath, accel_opt, accel, required_keys=["beam", "year"]
    )
    os.chdir(request.config.invocation_dir)  # return to original cwd


@pytest.mark.basic
def test_lhc_creation_nominal_driven_check_output(model_25cm_beam1):
    accel = get_accelerator(**model_25cm_beam1)
    LhcModelCreator.check_run_output(accel)

    for dat_file in (TWISS_AC_DAT, TWISS_DAT, TWISS_ELEMENTS_DAT, TWISS_ADT_DAT):
        file_path: Path = accel.model_dir / dat_file
        file_path_moved: Path = file_path.with_suffix(f"{file_path.suffix}0")
        if dat_file == TWISS_ADT_DAT:
            accel.excitation = AccExcitationMode.ADT  # Test ACD before
        else:
            shutil.move(file_path, file_path_moved)

        # Run test
        with pytest.raises(FileNotFoundError) as creation_error:
            LhcModelCreator.check_run_output(accel)
        assert str(dat_file) in str(creation_error.value)

        if file_path_moved.exists():
            shutil.move(file_path_moved, file_path)


# Helper -----------------------------------------------------------------------


def check_accel_from_dir_vs_options(model_dir, accel_options, accel_from_opt, required_keys):
    # creation via model_from_dir tests that all files are in place:
    accel_from_dir = get_accelerator(
        accel=accel_options["accel"],
        model_dir=model_dir,
        **{k: accel_options[k] for k in required_keys},
    )

    _check_arrays(accel_from_opt.nat_tunes, accel_from_dir.nat_tunes, eps=1e-4, tunes=True)
    _check_arrays(accel_from_opt.drv_tunes, accel_from_dir.drv_tunes, eps=1e-4, tunes=True)
    _check_arrays(accel_from_opt.modifiers, accel_from_dir.modifiers)
    assert accel_from_opt.excitation == accel_from_dir.excitation
    assert accel_from_opt.model_dir == accel_from_dir.model_dir

    # TODO: Energy not set in model ? (jdilly, 2021)
    # assert abs(accel_from_opt.energy - accel_from_dir.energy) < 1e-2


def _check_arrays(a_array, b_array, eps=None, tunes=False):
    if a_array is None:
        a_array = []

    if b_array is None:
        b_array = []

    if len(a_array) != len(b_array):
        raise AssertionError("Not the same amounts given.")

    for a, b in zip(a_array, b_array):
        if eps is None:
            assert a == b
        elif tunes:
            assert abs((a % 1) - (b % 1)) <= eps
        else:
            assert abs(a - b) <= eps
