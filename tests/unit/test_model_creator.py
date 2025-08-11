import copy
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import tfs
from generic_parser import DotDict

from omc3.model.accelerators.accelerator import (
    Accelerator,
    AcceleratorDefinitionError,
    AccExcitationMode,
)
from omc3.model.constants import (
    ACC_MODELS_PREFIX,
    JOB_MODEL_MADX_NOMINAL,
    OPTICS_SUBDIR,
    TWISS_AC_DAT,
    TWISS_ADT_DAT,
    TWISS_DAT,
    TWISS_ELEMENTS_DAT,
    Fetcher,
)
from omc3.model.manager import get_accelerator
from omc3.model.model_creators.lhc_model_creator import (
    LhcBestKnowledgeCreator,
    LhcModelCreator,
)
from omc3.model_creator import create_instance_and_model
from omc3.optics_measurements.constants import NAME
from tests.conftest import assert_frame_equal

if TYPE_CHECKING:
    from omc3.model.accelerators.lhc import Lhc

INPUTS = Path(__file__).parent.parent / "inputs"
LHC_2025_30CM_MODIFIERS = [Path("R2025aRP_A30cmC30cmA10mL200cm_Flat.madx")]
LHC_2018_HIGH_BETA_MODIFIERS = [Path("R2018h_A90mC90mA10mL10m.madx")]
UNAVAILABLE_FETCHER = "unavailable_fetcher"

# ---- creation tests ------------------------------------------------------------------------------

@pytest.mark.basic
def test_booster_creation_nominal_driven(tmp_path, acc_models_psb_2021):
    accel_opt = {
        "accel": "psbooster",
        "ring": 1,
        "nat_tunes": [0.28, 0.45],
        "drv_tunes": [0.205, 0.274],
        "driven_excitation": "acd",
        "dpp": 0.0,
        "energy": 0.16,
        "modifiers": None,
        "fetch": Fetcher.PATH,
        "path": acc_models_psb_2021,
        "scenario": "lhc",
        "cycle_point": "0_injection",
        "str_file": "psb_inj_lhc.str",
    }
    accel = create_instance_and_model(
        type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel)

    # now check a few error cases

    accel_opt_duplicate = accel_opt.copy()
    accel_opt_duplicate["scenario"] = None
    with pytest.raises(AcceleratorDefinitionError) as e:
        create_instance_and_model(
            type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt_duplicate
        )
    assert "flag --scenario" in str(e.value)
    assert "Selected: 'None'" in str(e.value)

    accel_opt_duplicate = accel_opt.copy()
    accel_opt_duplicate["str_file"] = None
    with pytest.raises(AcceleratorDefinitionError) as e:
        create_instance_and_model(
            type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt_duplicate
        )
    assert "flag --str_file" in str(e.value)
    assert "Selected: 'None'" in str(e.value)


@pytest.mark.basic
def test_booster_creation_nominal_free(tmp_path, acc_models_psb_2021):
    accel_opt = {
        "accel": "psbooster",
        "ring": 1,
        "nat_tunes": [0.28, 0.45],
        "dpp": 0.0,
        "energy": 0.16,
        "modifiers": None,
        "fetch": Fetcher.PATH,
        "path": acc_models_psb_2021,
        "scenario": "lhc",
        "cycle_point": "0_injection",
        "str_file": "psb_inj_lhc.str",
    }
    accel = create_instance_and_model(
        type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel)

# ps tune matching fails for 2018 optics
# The magnets used for the different tune matching methods in > 2018 were installed in LS2.
# TODO: check with PS expert a) if model creation <= 2018 is desired and b) how it worked
# @pytest.mark.basic
# def test_ps_creation_nominal_driven_2018(tmp_path):
#     accel_opt = {
#         "accel": "ps",
#         "nat_tunes": [0.21, 0.323], # from madx_job file in acc_models
#         "drv_tunes": [0.215, 0.318],
#         "driven_excitation": "acd",
#         "dpp": 0.0,
#         "energy": 1.4,
#         "year": "2018",
#         "fetch": Fetcher.PATH,
#         "path": MODEL_CREATOR_INPUT / "ps_2018",
#         "scenario": "lhc_proton",
#         "cycle_point": "0_injection",
#         "str_file": "ps_inj_lhc.str",
#         "tune_method": "f8l",
#     }
#     accel = create_instance_and_model(
#         type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt
#     )
#     check_accel_from_dir_vs_options(tmp_path, accel_opt, accel)


@pytest.mark.basic
def test_ps_creation_nominal_free_2018(tmp_path, acc_models_ps_2021):
    accel_opt = {
        "accel": "ps",
        "nat_tunes": [0.21, 0.323], # from madx_job file in acc_models
        "dpp": 0.0,
        "energy": 1.4,
        "year": "2018",
        "fetch": Fetcher.PATH,
        "path": acc_models_ps_2021,
        "scenario": "lhc",
        "cycle_point": "2_flat_top",
        "str_file": "ps_ft_lhc.str",
        "tune_method": "pfw",
    }
    accel = create_instance_and_model(
        type="nominal", outputdir=tmp_path, logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel)

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
def test_lhc_creation_nominal_driven(tmp_path, acc_models_lhc_2025):
    accel_opt = {
        "accel": "lhc",
        "year": "2025",
        "beam": 1,
        "nat_tunes": [0.31, 0.32],
        "drv_tunes": [0.298, 0.335],
        "driven_excitation": "acd",
        "dpp": 0.0,
        "energy": 6800.0,
        "fetch": Fetcher.PATH,
        "path": acc_models_lhc_2025,
        "modifiers": LHC_2025_30CM_MODIFIERS,
    }
    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel)

    # quick check for DOROS BPMs
    for twiss_name in (TWISS_DAT, TWISS_ELEMENTS_DAT):
        df_twiss = tfs.read(tmp_path / twiss_name, index=NAME)
        assert any(df_twiss.index.str.match(r"BPM.+_DOROS$"))

    # checks that should fail

    with pytest.raises(AcceleratorDefinitionError) as excinfo:
        accel_duplicate = copy.deepcopy(accel)
        accel_duplicate.model_dir = None
        LhcModelCreator(accel_duplicate).check_accelerator_instance()
    assert "model directory (outputdir option) was not given" in str(excinfo.value)

    with pytest.raises(AcceleratorDefinitionError) as excinfo:
        accel_duplicate = copy.deepcopy(accel)
        accel_duplicate.modifiers = None
        LhcModelCreator(accel_duplicate).check_accelerator_instance()
    assert "no modifiers could be found" in str(excinfo.value).lower()

    with pytest.raises(AttributeError) as excinfo:
        create_instance_and_model(
            type="nominal", outputdir=None, logfile=tmp_path / "madx_log.txt", **accel_opt
        )
    assert "Missing flag `outputdir`" in str(excinfo.value)


@pytest.mark.basic
def test_lhc_creation_nominal_free_high_beta(tmp_path, acc_models_lhc_2018):
    accel_opt = {
        "accel": "lhc",
        "year": "2018",
        "beam": 1,
        "nat_tunes": [0.31, 0.32],
        "dpp": 0.0,
        "energy": 6500.0,
        "fetch": Fetcher.PATH,
        "path": acc_models_lhc_2018,
        "modifiers": LHC_2018_HIGH_BETA_MODIFIERS
    }
    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel)


@pytest.mark.basic
def test_lhc_creation_nominal_free(tmp_path, acc_models_lhc_2025):
    accel_opt = {
        "accel": "lhc",
        "year": "2025",
        "beam": 1,
        "nat_tunes": [0.31, 0.32],
        "dpp": 0.0,
        "energy": 6800.0,
        "fetch": Fetcher.PATH,
        "path": acc_models_lhc_2025,
        "modifiers": LHC_2025_30CM_MODIFIERS
    }
    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel)


@pytest.mark.basic
def test_lhc_creation_nominal_2016(tmp_path):
    accel_opt = {
        "accel": "lhc",
        "year": "2016",
        "beam": 1,
        "nat_tunes": [0.31, 0.32],
        "dpp": 0.0,
        "energy": 6500.0,
        "modifiers": [INPUTS / "models" / "modifiers_2016" / "opt_400_10000_400_3000.madx"]
    }
    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel)


@pytest.mark.basic
def test_lhc_creation_best_knowledge(tmp_path, acc_models_lhc_2025):
    (tmp_path / LhcBestKnowledgeCreator.EXTRACTED_MQTS_FILENAME).write_text("\n")

    corrections = tmp_path / "other_corrections.madx"
    corrections_str = "! just a comment to test the corrections file is actually loaded in madx. whfifhkdskjfshkdhfswojeorijr"
    corrections.write_text(f"{corrections_str}\n")

    logfile = tmp_path / "madx_log.txt"

    accel_opt = {
        "accel": "lhc",
        "year": "2025",
        "ats": True,
        "beam": 1,
        "nat_tunes": [0.31, 0.32],
        "dpp": 0.0,
        "energy": 6800.0,
        "fetch": Fetcher.PATH,
        "path": acc_models_lhc_2025,
        "modifiers": LHC_2025_30CM_MODIFIERS + [corrections]
    }

    # like from the GUI, dump best knowledge on top of nominal
    accel_nominal: Lhc = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=logfile, **accel_opt
    )

    accel_opt["b2_errors"] = str(INPUTS / "models/error_tables/MB2022_6500.0GeV_0133cm")

    accel: Lhc = create_instance_and_model(
        outputdir=tmp_path, type="best_knowledge", logfile=logfile, **accel_opt
    )

    assert accel.model is None  # should not have been created in the opt
    accel.model = accel_nominal.model  # but is present in the tmp_dir, so add here to compare

    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel, best_knowledge=True)
    assert corrections_str in logfile.read_text()


@pytest.mark.basic
def test_lhc_creation_absolute_modifier_path(tmp_path: Path, acc_models_lhc_2022: Path):
    rel_path = OPTICS_SUBDIR / "R2022a_A30cmC30cmA10mL200cm.madx"
    accel_opt = {
        "accel": "lhc",
        "year": "2022",
        "ats": True,
        "beam": 1,
        "nat_tunes": [0.31, 0.32],
        "dpp": 0.0,
        "energy": 6800.0,
        "fetch": Fetcher.PATH,
        "path": acc_models_lhc_2022,
        "modifiers": [(acc_models_lhc_2022 / rel_path).absolute()]
    }
    accel = create_instance_and_model(
        outputdir=tmp_path, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )
    absolute_path = tmp_path / f"{ACC_MODELS_PREFIX}-{accel.NAME}" / rel_path  # replaced in model creation
    madx_string = f"call, file = '{absolute_path!s}"
    assert madx_string in (tmp_path / JOB_MODEL_MADX_NOMINAL).read_text()
    assert madx_string in (tmp_path / "madx_log.txt").read_text()
    check_accel_from_dir_vs_options(tmp_path, accel_opt, accel)


@pytest.mark.basic
def test_lhc_creation_modifier_nonexistent(tmp_path, acc_models_lhc_2018):
    non_existent = Path("opticsfile.non_existent")
    accel_opt = {
        "accel": "lhc",
        "year": "2018",
        "ats": True,
        "beam": 1,
        "nat_tunes": [0.31, 0.32],
        "dpp": 0.0,
        "energy": 6800.0,
        "fetch": Fetcher.PATH,
        "path": acc_models_lhc_2018,
        "modifiers": [non_existent]
    }
    with pytest.raises(FileNotFoundError) as creation_error:
        create_instance_and_model(
            outputdir=tmp_path,
            type="nominal",
            logfile=tmp_path / "madx_log.txt",
            **accel_opt,
        )
    assert non_existent.name in str(creation_error.value)


@pytest.mark.basic
@pytest.mark.timeout(60)  # madx might get stuck (seen on macos)
def test_lhc_creation_relative_modeldir_path(request, tmp_path, acc_models_lhc_2022):
    os.chdir(tmp_path)  # switch cwd to tmp_path
    model_dir_relpath = Path("test_model")
    model_dir_relpath.mkdir()

    optics_file_relpath = Path("R2022a_A30cmC30cmA10mL200cm.madx")
    shutil.copy(acc_models_lhc_2022 / "operation/optics" / optics_file_relpath, model_dir_relpath / optics_file_relpath.name)

    accel_opt = {
        "accel": "lhc",
        "year": "2022",
        "ats": True,
        "beam": 1,
        "nat_tunes": [0.31, 0.32],
        "dpp": 0.0,
        "energy": 6800.0,
        "fetch": Fetcher.PATH,
        "path": acc_models_lhc_2022,
        "modifiers": [optics_file_relpath],
    }

    # sometimes create_instance_and_model seems to run but does not create twiss-files ...
    accel = create_instance_and_model(
        outputdir=model_dir_relpath, type="nominal", logfile=tmp_path / "madx_log.txt", **accel_opt
    )

    # ... which is then caught here:
    check_accel_from_dir_vs_options(model_dir_relpath.absolute(), accel_opt, accel)
    os.chdir(request.config.invocation_dir)  # return to original cwd


@pytest.mark.basic
def test_lhc_creation_nominal_driven_check_output(model_25cm_beam1):
    """ Checks if the post_run() method succeeds on an already existing given model (dir),
    and then checks that it failes when removing individual files from that model. """
    accel = get_accelerator(**model_25cm_beam1)
    LhcModelCreator(accel).post_run()

    for dat_file in (TWISS_AC_DAT, TWISS_DAT, TWISS_ELEMENTS_DAT, TWISS_ADT_DAT):
        file_path: Path = accel.model_dir / dat_file
        file_path_moved: Path = file_path.with_suffix(f"{file_path.suffix}0")
        if dat_file == TWISS_ADT_DAT:
            accel.excitation = AccExcitationMode.ADT  # Test ACD before
        else:
            shutil.move(file_path, file_path_moved)

        # Run test
        with pytest.raises(FileNotFoundError) as creation_error:
            LhcModelCreator(accel).post_run()
        assert str(dat_file) in str(creation_error.value)

        if file_path_moved.exists():
            shutil.move(file_path_moved, file_path)

# ---- cli tests -----------------------------------------------------------------------------------

@pytest.mark.basic
def test_lhc_creator_cli(tmp_path, acc_models_lhc_2025, capsys):

    accel_opt = {
        "accel": "lhc",
        "year": "2025",
        "ats": True,
        "beam": 1,
        "nat_tunes": [0.31, 0.32],
        "dpp": 0.0,
        "energy": 6800.0,
        "fetch": Fetcher.PATH,
        "path": acc_models_lhc_2025,
        "list_choices": True,
    }
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
    accel_opt = {
        "accel": "psbooster",
        "ring": 1,
        "nat_tunes": [0.28, 0.45],
        "drv_tunes": [0.205, 0.274],
        "driven_excitation": "acd",
        "dpp": 0.0,
        "energy": 0.16,
        "modifiers": None,
        "fetch": Fetcher.PATH,
        "path": acc_models_psb_2021,
        "list_choices": True,
    }

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
    accel_opt = {
        "accel": "ps",
        "nat_tunes": [0.21, 0.323], # from madx_job file in acc_models
        "dpp": 0.0,
        "energy": 1.4,
        "year": "2018",
        "fetch": Fetcher.PATH,
        "path": acc_models_ps_2021,
        "scenario": "lhc",
        "tune_method": "pfw",
        "list_choices": True,
    }
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

def check_accel_from_dir_vs_options(
    model_dir: Path,
    accel_options: DotDict,
    accel_from_opt: Accelerator,
    best_knowledge=False
    ):
    # creation via model_from_dir tests that all files are in place:
    accel_from_dir: Accelerator = get_accelerator(
        accel=accel_options["accel"],
        model_dir=model_dir,
        **_get_required_accelerator_parameters(accel_from_opt),
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

        assert_frame_equal(accel_from_opt.model_best_knowledge, accel_from_dir.model_best_knowledge)

    if accel_from_dir.model is not None:
        assert_frame_equal(accel_from_opt.model, accel_from_dir.model)

    if accel_from_opt.excitation != AccExcitationMode.FREE:
        assert_frame_equal(accel_from_opt.model_driven, accel_from_dir.model_driven)

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


def _get_required_accelerator_parameters(accel_inst: Accelerator) -> dict:
    """Return the required parameters with the values from  the accelerator instance."""
    parameters_required = {}
    parameters_accel = accel_inst.__class__.get_parameters()
    for name, param in parameters_accel.items():
        if param.get("required", False):
            parameters_required[name] = getattr(accel_inst, name)
    return parameters_required
