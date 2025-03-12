""" 
SPS Tests
---------

Here are tests for the SPS Accelerator, testing that different units of the 
analysis works also for this machine.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
import shutil

import pytest
import tfs

from omc3.definitions.optics import OpticsMeasurement
from omc3.global_correction import OPTICS_PARAMS_CHOICES
from omc3.hole_in_one import (
    LINFILES_SUBFOLDER,
    hole_in_one_entrypoint,
)
from omc3.model import manager
from omc3.model.accelerators.accelerator import AcceleratorDefinitionError
from omc3.model.accelerators.sps import Sps
from omc3.model.constants import (
    JOB_MODEL_MADX_NOMINAL,
    STRENGTHS_SUBDIR,
    TWISS_AC_DAT,
    TWISS_DAT,
    TWISS_ELEMENTS_DAT,
    Fetcher,
)
from omc3.model.model_creators.manager import CreatorType
from omc3.model.model_creators.sps_model_creator import (
    SpsModelCreator,
    SpsSegmentCreator,
)
from omc3.model_creator import create_instance_and_model
from omc3.optics_measurements import phase
from omc3.optics_measurements.constants import (
    NAME,
)
from omc3.response_creator import ResponseCreatorType, create_response_entrypoint as create_response
from omc3.segment_by_segment.constants import logfile
from omc3.segment_by_segment.propagables import get_all_propagables
from omc3.segment_by_segment.segments import Segment
from tests.accuracy.test_sbs import (
    assert_file_exists_and_nonempty,
    assert_twiss_contains_segment,
)
from tests.conftest import INPUTS
from tests.unit.test_hole_in_one import (
    _check_all_harpy_files,
    _check_linear_optics_files,
    _check_nonlinear_optics_files,
)
from tests.unit.test_model_creator import check_accel_from_dir_vs_options

SPS_DIR = INPUTS / "sps_data"
SPS_MODEL_DIR = SPS_DIR / "model_Q20_noacd"
Q20_STRENGTHS_FILE = "lhc_q20.str"

class TestModelCrationSPS:
    @pytest.mark.basic
    @pytest.mark.parametrize("use_acdipole", [True, False], ids=["acdipole", "no_acdipole"])
    def test_nominal_driven(self, tmp_path: Path, acc_models_sps_2025: Path, use_acdipole: bool):
        """ Tests the creation of a nominal model with ACDipole for the SPS. """
        accel_opt = dict(
            accel="sps",
            year="2025",
            nat_tunes=[20.13, 20.18],
            drv_tunes=[0.26, 0.282],
            driven_excitation="acd" if use_acdipole else None,
            fetch=Fetcher.PATH,
            path=acc_models_sps_2025,
            modifiers=[Q20_STRENGTHS_FILE],
        )
        accel = create_instance_and_model(
            outputdir=tmp_path, type=CreatorType.NOMINAL, logfile=tmp_path / "madx_log.txt", **accel_opt
        )

        if not use_acdipole:
            accel.drv_tunes = None  # cannot read that form model-dir when TWISS-AC is not created
        check_accel_from_dir_vs_options(tmp_path, accel_opt, accel)

        # quick check for BPMs
        twisses = [TWISS_DAT, TWISS_ELEMENTS_DAT] + ([TWISS_AC_DAT] if use_acdipole else [])
        for twiss_name in twisses:
            df_twiss = tfs.read(tmp_path / twiss_name, index=NAME)
            assert any(df_twiss.index.str.match(r"BPV"))
            assert any(df_twiss.index.str.match(r"BPH"))

        job_content = (tmp_path / JOB_MODEL_MADX_NOMINAL).read_text()
        assert Q20_STRENGTHS_FILE in job_content
        assert "hacmap" in job_content
        assert re.search(fr"use_acd\s*=\s*{use_acdipole:d}", job_content)

        if not use_acdipole:
            assert not (tmp_path / TWISS_AC_DAT).is_file()
            return

        # checks that should fail
        with pytest.raises(AcceleratorDefinitionError) as excinfo:
            accel_duplicate = copy.deepcopy(accel)
            accel_duplicate.modifiers = None
            SpsModelCreator(accel_duplicate).check_accelerator_instance()
        assert "no modifiers could be found" in str(excinfo.value).lower()

        with pytest.raises(AttributeError) as excinfo:
            create_instance_and_model(
                type="nominal", outputdir=None, logfile=tmp_path / "madx_log.txt", **accel_opt
            )
        assert "Missing flag `outputdir`" in str(excinfo.value)


    @pytest.mark.basic
    def test_nominal_free(self, tmp_path, acc_models_sps_2025):
        """ Tests the creation of a nominal model without ACDipole for the SPS. """
        accel_opt = dict(
            accel="sps",
            year="2025",
            nat_tunes=[20.13, 20.18],
            fetch=Fetcher.PATH,
            path=acc_models_sps_2025,
            modifiers=[Q20_STRENGTHS_FILE],
        )
        accel = create_instance_and_model(
            outputdir=tmp_path, type=CreatorType.NOMINAL, logfile=tmp_path / "madx_log.txt", **accel_opt
        )
        check_accel_from_dir_vs_options(tmp_path, accel_opt, accel)

        assert not (tmp_path / TWISS_AC_DAT).is_file()
        job_content = (tmp_path / JOB_MODEL_MADX_NOMINAL).read_text()
        assert Q20_STRENGTHS_FILE in job_content
        assert "hacmap" not in job_content

    @pytest.mark.extended
    def test_segment_creation(self, 
        tmp_path: Path, 
        acc_models_sps_2025: Path, 
        ): 
        """ Tests the creation of the Segment Models via SpsSegmentCreator. 
        Everything else about Segment-by-Segment should be the same as for LHC.
        """
        # Preparation ----------------------------------------------------------
        accel_opt = dict(
            accel="sps",
            nat_tunes=[20.13, 20.18],
            modifiers=[acc_models_sps_2025 / STRENGTHS_SUBDIR / Q20_STRENGTHS_FILE],
        )

        correction_path = create_error_file(tmp_path)

        iplabel = "SomeSegment"
        segment = Segment(
            name=iplabel,
            start="BPV.52108",
            end="BPH.53608",
        )
        measurement = OpticsMeasurement(SPS_DIR / "fake_measurement_Q20")

        propagables = [propg(segment, measurement) for propg in get_all_propagables()]
        measureables = [measbl for measbl in propagables if measbl]     
        
        accel_inst: Sps = manager.get_accelerator(accel_opt)
        accel_inst.model_dir = tmp_path  # if set in accel_opt, it tries to load from model_dir, but this is the output dir for the segment-models
        accel_inst.acc_model_path = acc_models_sps_2025
        
        segment_creator = SpsSegmentCreator(
            segment=segment, 
            measurables=measureables,
            logfile=tmp_path / logfile.format(segment.name),
            accel=accel_inst,
            corrections=correction_path,
        )

        # Actual Run -----------------------------------------------------------
        segment_creator.full_run()

        # Test the output ------------------------------------------------------ 
        assert len(list(tmp_path.glob(f"*{Path(TWISS_DAT).suffix}"))) == 4  # 2 segment, 2 segment corrected

        assert_file_exists_and_nonempty(tmp_path / segment_creator.measurement_madx)
        
        # created in madx (should also have been checked in the post_run() method)
        assert_twiss_contains_segment(tmp_path / segment_creator.twiss_forward, segment.start, segment.end)
        assert_twiss_contains_segment(tmp_path / segment_creator.twiss_backward, segment.end, segment.start)

        assert_file_exists_and_nonempty(tmp_path / segment_creator.corrections_madx)
        
        # created in madx (should also have been checked in the post_run() method)
        assert_twiss_contains_segment(tmp_path / segment_creator.twiss_forward_corrected, segment.start, segment.end)
        assert_twiss_contains_segment(tmp_path / segment_creator.twiss_backward_corrected, segment.end, segment.start)

    @pytest.mark.extended
    def test_response_creation(self, tmp_path: Path, acc_models_sps_2025: Path):
        """ Tests the creation of the Response via ResponseCreator. """
        tmp_model = tmp_path / "model"
        acc_models_link = tmp_model / "acc-models-sps"
        shutil.copytree(SPS_MODEL_DIR, tmp_model)
        acc_models_link.symlink_to(acc_models_sps_2025)

        tcsm_bump = ["kmdh51207", "kmdh51407", "kmdh52207"]  # update when json changes
        variables = ["tcsm_bump", "-kmdh52007", "klqsa", "kqf", "kqd"]
        fullresponse_path = tmp_path / "fullresponse.h5"
        new_response = create_response(
            accel="sps",
            model_dir=tmp_model,
            modifiers=[acc_models_link / STRENGTHS_SUBDIR / Q20_STRENGTHS_FILE],
            #
            creator=ResponseCreatorType.MADX,
            delta_k=1e-6,
            variable_categories=variables,
            outfile_path=fullresponse_path,
        )

        optics_params = list(OPTICS_PARAMS_CHOICES[2:]) + ["MUX", "MUY"]   # in choices its PHASE as this is the measurement column
        assert len(optics_params) 
        assert all(var in new_response for var in optics_params)

        bpms = tfs.read(tmp_model / TWISS_DAT, index=NAME).index

        for param in optics_params:
            assert all(bump_var in new_response[param].columns for bump_var in tcsm_bump)
            assert "kmdh52007" not in new_response[param].columns
            assert "klqsa" in new_response[param].columns
            assert "kqf" in new_response[param].columns
            assert "kqd" in new_response[param].columns
            if param == "Q":
                assert all(tune in new_response[param].index for tune in ("Q1", "Q2"))
            else:
                assert all(bpm in new_response[param].index for bpm in bpms)

        assert_file_exists_and_nonempty(fullresponse_path)
        

class TestAnalysisSPS:
    @pytest.mark.extended
    def test_hole_in_one(self, tmp_path, caplog):
        """
        This test runs harpy and optics analysis in one for SPS data.
        This data is representative for single-plane BPMs
        and BPMs with NaNs which caused some errors prior to v0.21.0 .
        """
        rdt_order = 3
        output = tmp_path / "output"
        files = [SPS_DIR / "sps_200turns.sdds"]
        nan_bpms = ["BPH.31808", "BPH.32008", "BPV.31708", "BPV.31908"]

        hole_in_one_entrypoint(
            harpy=True,
            optics=True,  # do not need to run optics, but good to test if it works
            clean=True,
            tbt_datatype="sps",
            compensation=phase.CompensationMode.NONE,
            output_bits=8,
            turn_bits=12,  # lower and rdt calculation fails for coupling
            resonances=rdt_order+1,
            # nattunes = [0.13, 0.18, 0.0],
            autotunes="transverse",
            outputdir=output,
            files=files,
            model=SPS_MODEL_DIR / TWISS_ELEMENTS_DAT,
            to_write=["lin", "spectra",],
            window="hann",
            coupling_method=2,
            nonlinear=['rdt',],
            rdt_magnet_order=rdt_order,
            unit="mm",
            accel="generic",
            model_dir=SPS_MODEL_DIR,
            three_bpm_method=True,  # n-bpm method needs error-def file
        )

        for sdds_file in files:
            _check_all_harpy_files(output / LINFILES_SUBFOLDER, sdds_file)

        _check_linear_optics_files(output, off_momentum=False)
        _check_nonlinear_optics_files(output, "rdt", order=rdt_order)
        
        assert "NaN BPMs detected." in caplog.text
        for bpm in nan_bpms:
            assert bpm in caplog.text


def create_error_file(path: Path):
    out_path = path / "my_errors.madx"
    out_path.write_text(
        "seqedit, sequence=sps;\n"
        "    flatten;\n"
        "    fakemagnetf: multipole, l=0, knl = {0, 5e-4, 1e-3};\n"
        "    fakemagnetd: multipole, l=0, knl = {0, 5e-4, 1e-3};\n"
        "    install, element=fakemagnetf, at=QF.52410->L/2, from=QF.52410;\n"
        "    install, element=fakemagnetd, at=QD.52510->L/2, from=QD.52510;\n"
        "endedit;\n"
    )
    return out_path