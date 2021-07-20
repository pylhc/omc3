import shutil
from pathlib import Path

import pytest

from omc3.model.accelerators.accelerator import AcceleratorDefinitionError
from omc3.model.manager import get_accelerator
from omc3.model.model_creators.lhc_model_creator import LhcBestKnowledgeCreator
from omc3.model_creator import create_instance_and_model

INPUTS = Path(__file__).parent.parent / "inputs"
COMP_MODEL = INPUTS / "models" / "25cm_beam1"
PS_MODEL = Path(__file__).parent.parent.parent / "omc3" / "model" / "accelerators" / "ps" / "2018" / "strength"


@pytest.mark.basic
def test_booster_creation_nominal(tmp_path):
    accel_opt = dict(
        accel="psbooster",
        ring=1,
        nat_tunes=[4.21, 4.27],
        drv_tunes=[0.205, 0.274],
        driven_excitation="acd",
        dpp=0.0, energy=0.16,
        modifiers=None,
    )
    accel = create_instance_and_model(
        type="nominal",
        outputdir=tmp_path,
        logfile=tmp_path / "madx_log.txt",
        **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel,
                                    required_keys=['ring'])


@pytest.mark.basic
def test_ps_creation_nominal(tmp_path):
    accel_opt = dict(
        accel="ps",
        nat_tunes=[6.32, 6.29],
        drv_tunes=[0.325, 0.284],
        driven_excitation="acd",
        dpp=0.0, energy=1.4,
        modifiers=[PS_MODEL / "elements.str", PS_MODEL / "PS_LE_LHC_low_chroma.str"],
    )
    accel = create_instance_and_model(
        type="nominal",
        outputdir=tmp_path,
        logfile=tmp_path / "madx_log.txt",
        **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel,
                                    required_keys=[])


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
        outputdir=tmp_path,
        type="nominal",
        logfile=tmp_path / "madx_log.txt",
        **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel,
                                    required_keys=['beam', 'year'])


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
        outputdir=tmp_path,
        type="nominal",
        logfile=tmp_path / "madx_log.txt",
        **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel,
                                    required_keys=['beam', 'year'])


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
        outputdir=tmp_path,
        type="best_knowledge",
        logfile=tmp_path / "madx_log.txt",
        **accel_opt
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
        outputdir=tmp_path,
        type="nominal",
        logfile=tmp_path / "madx_log.txt",
        **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel,
                                    required_keys=['beam', 'year'])


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
    with pytest.raises(AcceleratorDefinitionError) as e:
        create_instance_and_model(
            outputdir=tmp_path,
            type="nominal",
            logfile=tmp_path / "madx_log.txt",
            **accel_opt
        )
    assert "opticsfile.non_existent" in str(e.value)


# Helper -----------------------------------------------------------------------

def check_accel_from_dir_vs_options(model_dir, accel_options, accel_from_opt, required_keys):
    accel_from_dir = get_accelerator(
        accel=accel_options['accel'],
        model_dir=model_dir,
        **{k: accel_options[k] for k in required_keys}
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
            assert abs((a%1) - (b%1)) <= eps
        else:
            assert abs(a - b) <= eps

