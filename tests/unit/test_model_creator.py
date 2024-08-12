import os
import shutil
import copy
from pathlib import Path

import pytest
from omc3.model.accelerators.accelerator import AcceleratorDefinitionError, AccExcitationMode
from omc3.model.constants import TWISS_AC_DAT, TWISS_ADT_DAT, TWISS_DAT, TWISS_ELEMENTS_DAT, PATHFETCHER
from omc3.model.manager import get_accelerator
from omc3.model.model_creators.lhc_model_creator import LhcBestKnowledgeCreator, LhcModelCreator
from omc3.model_creator import create_instance_and_model
from omc3.model.model_creators.lhc_model_creator import LhcModelCreator

INPUTS = Path(__file__).parent.parent / "inputs"
LHC_30CM_MODIFIERS = [Path("R2023a_A30cmC30cmA10mL200cm.madx")]
HIGH_BETA_MODIFIERS = [Path("R2018h_A90mC90mA10mL10m.madx")]
UNAVAILABLE_FETCHER = "unavailable_fetcher"

# ---- creation tests ------------------------------------------------------------------------------

@pytest.mark.basic
def test_booster_creation_nominal_driven(tmp_path, acc_models_psb_2021):
    accel_opt = dict(
        accel="psbooster",
        ring=1,
        nat_tunes=[0.28, 0.45],
        drv_tunes=[0.205, 0.274],
        driven_excitation="acd",
        dpp=0.0,
        energy=0.16,
        modifiers=None,
        fetch=PATHFETCHER,
        path=acc_models_psb_2021,
        scenario="lhc",
        cycle_point="0_injection",
        str_file="psb_inj_lhc.str",
    )
    accel = create_instance_and_model(
        type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["ring"])

    # now check a few error cases

    accel_opt_duplicate = accel_opt.copy()
    accel_opt_duplicate["scenario"] = None
    with pytest.raises(AttributeError):
        create_instance_and_model(
            type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt_duplicate
        )

    accel_opt_duplicate = accel_opt.copy()
    accel_opt_duplicate["str_file"] = None
    with pytest.raises(AttributeError):
        create_instance_and_model(
            type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt_duplicate
        )


@pytest.mark.basic
def test_booster_creation_nominal_free(tmp_path, acc_models_psb_2021):
    accel_opt = dict(
        accel="psbooster",
        ring=1,
        nat_tunes=[0.28, 0.45],
        dpp=0.0,
        energy=0.16,
        modifiers=None,
        fetch=PATHFETCHER,
        path=acc_models_psb_2021,
        scenario="lhc",
        cycle_point="0_injection",
        str_file="psb_inj_lhc.str",
    )
    accel = create_instance_and_model(
        type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["ring"])

# # ps tune matching fails for 2018 optics
# # The magnets used for the different tune matching methods in > 2018 were installed in LS2. 
# # TODO: check with PS expert a) if model creation <= 2018 is desired and b) how it worked
#
# @pytest.mark.basic
# def test_ps_creation_nominal_driven_2018(tmp_path):
#     accel_opt = dict(
#         accel="ps",
#         nat_tunes=[0.21, 0.323], # from madx_job file in acc_models
#         drv_tunes=[0.215, 0.318],
#         driven_excitation="acd",
#         dpp=0.0,
#         energy=1.4,
#         year="2018",
#         fetch=PATHFETCHER,
#         path=MODEL_CREATOR_INPUT / "ps_2018",
#         scenario="lhc_proton",
#         cycle_point="0_injection",
#         str_file="ps_inj_lhc.str",
#         tune_method="f8l",
#     )
#     accel = create_instance_and_model(
#         type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt
#     )
#     check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["year"])
#
#
@pytest.mark.basic
def test_ps_creation_nominal_free_2018(tmp_path, acc_models_ps_2021):
    accel_opt = dict(
        accel="ps",
        nat_tunes=[0.21, 0.323], # from madx_job file in acc_models
        dpp=0.0,
        energy=1.4,
        year="2018",
        fetch=PATHFETCHER,
        path=acc_models_ps_2021,
        scenario="lhc",
        cycle_point="2_flat_top",
        str_file="ps_ft_lhc.str",
        tune_method="pfw",
    )
    accel = create_instance_and_model(
        type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["year"])

    # the PS acc-models repo doesn't provide `.beam` files, that could be used to extract the
    # energy settings for each scenario automatically. So we rely on te user to specify this
    accel_opt_duplicate = accel_opt.copy()
    accel_opt_duplicate["energy"] = None

    with pytest.raises(RuntimeError) as excinfo:
        create_instance_and_model(
            type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt_duplicate
        )
    assert "Please provide the --energy ENERGY flag" in str(excinfo.value)


@pytest.mark.basic
def test_lhc_creation_nominal_driven(tmp_path, acc_models_lhc_2023):
    accel_opt = dict(
        accel="lhc",
        year="2023",
        beam=1,
        nat_tunes=[0.31, 0.32],
        drv_tunes=[0.298, 0.335],
        driven_excitation="acd",
        dpp=0.0,
        energy=6800.0,
        fetch=PATHFETCHER,
        path=acc_models_lhc_2023,
        modifiers=LHC_30CM_MODIFIERS,
    )
    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["beam", "year"])

    # checks that should fail

    with pytest.raises(AcceleratorDefinitionError) as excinfo:
        accel_duplicate = copy.deepcopy(accel)
        accel_duplicate.model_dir = None
        LhcModelCreator.check_accelerator_instance(accel_duplicate)
    assert "model directory (outputdir option) was not given" in str(excinfo.value)

    with pytest.raises(AcceleratorDefinitionError) as excinfo:
        accel_duplicate = copy.deepcopy(accel)
        accel_duplicate.modifiers = None
        LhcModelCreator.check_accelerator_instance(accel_duplicate)
    assert "no modifiers could be found" in str(excinfo.value)

    with pytest.raises(AttributeError):
        create_instance_and_model(
            type="nominal", outputdir=None, logfile=tmp_path / "madx_log.txt", **accel_opt
        )


@pytest.mark.basic
def test_lhc_creation_nominal_free_high_beta(tmp_path, acc_models_lhc_2018):
    accel_opt = dict(
        accel="lhc",
        year="2018",
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6500.0,
        fetch=PATHFETCHER,
        path=acc_models_lhc_2018,
        modifiers=HIGH_BETA_MODIFIERS
    )
    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["beam", "year"])


@pytest.mark.basic
def test_lhc_creation_nominal_free(tmp_path, acc_models_lhc_2023):
    accel_opt = dict(
        accel="lhc",
        year="2023",
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6800.0,
        fetch=PATHFETCHER,
        path=acc_models_lhc_2023,
        modifiers=LHC_30CM_MODIFIERS
    )
    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["beam", "year"])


@pytest.mark.basic
def test_lhc_creation_best_knowledge(tmp_path, acc_models_lhc_2023):
    (tmp_path / LhcBestKnowledgeCreator.EXTRACTED_MQTS_FILENAME).write_text("\n")
    
    corrections = tmp_path / "other_corrections.madx"
    corrections_str = "! just a comment to test the corrections file is actually loaded in madx. whfifhkdskjfshkdhfswojeorijr"
    corrections.write_text(f"{corrections_str}\n")

    logfile = tmp_path / "madx_log.txt"

    accel_opt = dict(
        accel="lhc",
        year="2023",
        ats=True,
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6800.0,
        fetch=PATHFETCHER,
        path=acc_models_lhc_2023,
        modifiers=LHC_30CM_MODIFIERS + [corrections]
    )

    # like from the GUI, dump best knowledge on top of nominal
    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=logfile, **accel_opt
    )

    accel_opt["b2_errors"] = str(INPUTS / "models/error_tables/MB2022_6500.0GeV_0133cm")

    accel = create_instance_and_model(
        outputdir=tmp_path, type="best_knowledge", logfile=logfile, **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["beam", "year"], best_knowledge=True)
    assert corrections_str in logfile.read_text()


@pytest.mark.basic
def test_lhc_creation_relative_modifier_path(tmp_path, acc_models_lhc_2022):
    accel_opt = dict(
        accel="lhc",
        year="2022",
        ats=True,
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6800.0,
        fetch=PATHFETCHER,
        path=acc_models_lhc_2022,
        modifiers=LHC_30CM_MODIFIERS
    )
    #shutil.copy(MODEL_CREATOR_INPUT / "lhc_2022/operation/optics" / "R2022a_A30cmC30cmA10mL200cm.madx", tmp_path / "R2022a_A30cmC30cmA10mL200cm.madx")

    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, required_keys=["beam", "year"])


@pytest.mark.basic
def test_lhc_creation_modifier_nonexistent(tmp_path, acc_models_lhc_2018):
    NONEXISTENT = Path("opticsfile.non_existent")
    accel_opt = dict(
        accel="lhc",
        year="2018",
        ats=True,
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6800.0,
        fetch=PATHFETCHER,
        path=acc_models_lhc_2018,
        modifiers=[NONEXISTENT]
    )
    with pytest.raises(FileNotFoundError) as creation_error:
        create_instance_and_model(
            outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
        )
    assert NONEXISTENT.name in str(creation_error.value)


@pytest.mark.basic
@pytest.mark.timeout(60)  # madx might get stuck (seen on macos)
def test_lhc_creation_relative_modeldir_path(request, tmp_path, acc_models_lhc_2022):
    os.chdir(tmp_path)  # switch cwd to tmp_path
    model_dir_relpath = Path("test_model")
    model_dir_relpath.mkdir()

    optics_file_relpath = Path("R2022a_A30cmC30cmA10mL200cm.madx")
    shutil.copy(acc_models_lhc_2022 / "operation/optics" / optics_file_relpath, model_dir_relpath / optics_file_relpath.name)

    accel_opt = dict(
        accel="lhc",
        year="2022",
        ats=True,
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6800.0,
        fetch=PATHFETCHER,
        path=acc_models_lhc_2022,
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

# ---- cli tests -----------------------------------------------------------------------------------

@pytest.mark.basic
def test_lhc_creator_cli(tmp_path, acc_models_lhc_2023, capsys):

    accel_opt = dict(
        accel="lhc",
        year="2023",
        ats=True,
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6800.0,
        fetch=PATHFETCHER,
        path=acc_models_lhc_2023,
        list_choices=True,
    )
    create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )

    output = capsys.readouterr().out
    modifiers = output.split("\n")
    modifiers = [m for m in modifiers if len(m) > 0]  # remove empty lines

    # let's check that we got modifiers (we must, since `acc_models_lhc_2023` is pointing to a valid
    # acc-models directory)
    assert len(modifiers) > 0

    # furthermore, all of the returned filenames must be `.madx` files
    for m in modifiers:
        assert m.endswith(".madx")

@pytest.mark.basic
def test_booster_creator_cli(tmp_path, acc_models_psb_2021, capsys):
    accel_opt = dict(
        accel="psbooster",
        ring=1,
        nat_tunes=[0.28, 0.45],
        drv_tunes=[0.205, 0.274],
        driven_excitation="acd",
        dpp=0.0,
        energy=0.16,
        modifiers=None,
        fetch=PATHFETCHER,
        path=acc_models_psb_2021,
        list_choices=True,
    )

    create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )

    output = capsys.readouterr().out
    scenarios = output.split("\n")
    scenarios = [c for c in scenarios if len(c) > 0]  # remove empty lines
    scenarios.sort()

    assert scenarios == [
        "ad",
        "east",
        "isolde",
        "lhc",
        "sftpro",
        "tof"
    ]

    accel_opt["scenario"] = "lhc"

    create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )

    output = capsys.readouterr().out
    cycle_points = output.split("\n")
    cycle_points = [c for c in cycle_points if len(c) > 0]  # remove empty lines
    cycle_points.sort()

    assert cycle_points == [
        "0_injection",
        "1_flat_bottom",
        "2_flat_top",
        "3_extraction",
    ]

@pytest.mark.basic
def test_ps_creation_cli(tmp_path, acc_models_ps_2021, capsys):
    accel_opt = dict(
        accel="ps",
        nat_tunes=[0.21, 0.323], # from madx_job file in acc_models
        dpp=0.0,
        energy=1.4,
        year="2018",
        fetch=PATHFETCHER,
        path=acc_models_ps_2021,
        scenario="lhc",
        tune_method="pfw",
        list_choices=True,
    )
    create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )

    output = capsys.readouterr().out
    cycle_points = output.split("\n")
    cycle_points = [c for c in cycle_points if len(c) > 0]  # remove empty lines
    cycle_points.sort()

    assert cycle_points == [
        "0_injection",
        "1_flat_bottom",
        "2_flat_top",
        "3_extraction",
        "4_flat_bottom_wo_QDN90",
    ]

# ---- helper --------------------------------------------------------------------------------------

def check_accel_from_dir_vs_options(model_dir, accel_options, accel_from_opt, required_keys, best_knowledge=False):
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

    if best_knowledge:
        assert accel_from_dir.model_best_knowledge is not None

        beta_model = accel_from_dir.model["BETX"]
        beta_bk = accel_from_dir.model_best_knowledge["BETX"]

        _check_arrays(beta_model, beta_bk, eps=1e-4, is_close=False)

    # TODO: Energy not set in model ? (jdilly, 2021)
    # assert abs(accel_from_opt.energy - accel_from_dir.energy) < 1e-2


def _check_arrays(a_array, b_array, eps=None, tunes=False, is_close=True):
    if a_array is None:
        a_array = []

    if b_array is None:
        b_array = []

    if len(a_array) != len(b_array):
        raise AssertionError("Not the same amounts given.")

    for a, b in zip(a_array, b_array):
        if eps is None:
            assert (a == b) == is_close
        elif tunes:
            assert (abs((a % 1) - (b % 1)) <= eps) == is_close
        else:
            assert (abs(a - b) <= eps) == is_close
