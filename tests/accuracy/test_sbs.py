import shutil
from pathlib import Path

import pytest

from omc3.definitions.optics import OpticsMeasurement
from omc3.model import manager
from omc3.model.constants import Fetcher
from omc3.model.model_creators.lhc_model_creator import LhcSegmentCreator
from omc3.sbs_propagation import segment_by_segment
from omc3.segment_by_segment.constants import corrections_madx, logfile
from omc3.segment_by_segment.definitions import PropagableColumns
from omc3.segment_by_segment.propagables import Phase, get_all_propagables
from omc3.segment_by_segment.segments import Segment, SegmentDiffs
from omc3.utils import logging_tools
from tests.conftest import INPUTS

LOG = logging_tools.get_logger(__name__)

SBS_DIR = INPUTS / "sbs"
MAX_DIFF = 1e-10


class TestSbSLHC:
    __test__ = False  # ignore for now

    @pytest.mark.basic
    def test_lhc_segment_creation(self, model_25cm_beam1, tmp_path): #TODO get measurements for Beam 2 
        """ Tests only the creation of the Segment Models via LhcSegmentCreator. 
        A lot of this is actually done in the sbs_propagation as well, but 
        if things fail in the madx model creation, this is a good place to start looking.
        """
        beam = model_25cm_beam1.beam
        accel_opt = dict(
            accel="lhc",
            year="2018",
            beam=beam,
            nat_tunes=[0.31, 0.32],
            dpp=0.0,
            energy=6500,
            modifiers=[model_25cm_beam1.model_dir / "opticsfile.24_ctpps2"],
        )
        iplabel = "IP1"
        _write_correction_file(tmp_path, iplabel)

        segment = Segment(
            name=iplabel,
            start=f"BPM.12L1.B{beam:d}",
            end=f"BPM.12R1.B{beam:d}",
        )
        measurement = OpticsMeasurement(SBS_DIR / "measurements")

        propagables = [propg(segment, measurement) for propg in get_all_propagables()]
        measureables = [measbl for measbl in propagables if measbl]     
        
        
        accel_inst = manager.get_accelerator(accel_opt)
        accel_inst.model_dir = tmp_path  # if set in accel_opt, it tries to load from model_dir

        segment_creator = LhcSegmentCreator(
            segment=segment, 
            measurables=measureables,
            logfile=tmp_path / logfile.format(segment.name),
            accel=accel_inst,
        )
        
        segment_creator.full_run()

        assert len(list(tmp_path.glob("*.dat"))) == 4
        
        files_to_check = [
            # created in preparation (madx would fail without)
            segment_creator.measurement_madx,
            # created in madx (should also have been checked in the post_run() method)
            segment_creator.twiss_forward, 
            segment_creator.twiss_forward_corrected,
            segment_creator.twiss_backward,
            segment_creator.twiss_backward_corrected, 
        ]
        for file_ in files_to_check:
            assert_file_exists_and_nonempty(tmp_path / file_)

    @pytest.mark.basic
    @pytest.mark.parametrize("load_model", [True, False], ids=("load_model", "create_model"))
    @pytest.mark.parametrize("with_correction", [True, False], ids=("with_correction", "no_correction"))
    def test_lhc_propagation_sbs(self, tmp_path, model_inj_beams, load_model: bool, with_correction: bool, acc_models_lhc_2018):
        """Runs the segment creation as well as the parameter propagation.
        TODO: make test with creating the model on the fly 
        TODO: make test with loading model with and without output dir 
        TODO: With and without correction
        TODO: Find measurements for beam 2
        (works only within cern network unless afs is mocked)
        """
        # Preparation ---
        model_dir: Path = tmp_path / "my_model"

        beam = model_inj_beams.beam
        if beam == 2:
            pytest.skip("Tests for Beam 2 not yet implemented")
            # return # TODO find measurement

        accel_opt = dict(
            accel="lhc",
            year="2018",
            beam=beam,
            nat_tunes=[0.31, 0.32],
            dpp=0.0,
            energy=6500,
            modifiers=[model_inj_beams.model_dir / "opticsfile.1"],
        )

        if load_model:
            output_dir = None
            accel_opt["model_dir"] = model_dir
            accel_opt["nat_tunes"] = None
            shutil.copytree(model_inj_beams.model_dir, model_dir)  # creates model_dir
        else:
            output_dir = model_dir
            accel_opt["fetch"] = Fetcher.PATH
            accel_opt["path"] = acc_models_lhc_2018 

        segments = [
            Segment("IP1", f"BPM.12L1.B{beam:d}", f"BPM.12R1.B{beam:d}"),
            Segment("IP5", f"BPM.12L5.B{beam:d}", f"BPM.12R5.B{beam:d}"),
        ]
        
        if with_correction:
            model_dir.mkdir(exist_ok=True, parents=True)
            for segment in segments:
                _write_correction_file(model_dir, segment.name)

        # Run segment creation ---
        sbs_res = segment_by_segment(
            measurement_dir=SBS_DIR / "measurements",
            segments=[s.to_input_string() for s in segments],
            output_dir=output_dir, 
            **accel_opt,
        )

        # Tests ---
        columns_x: PropagableColumns = Phase.columns.planed("X")
        columns_y: PropagableColumns = Phase.columns.planed("Y")
        column_types = ["column", "forward", "backward"]
        if with_correction:
            column_types += ["forward_corrected", "backward_corrected"]
        
        column_list_x = [getattr(columns_x, c) for c in column_types]
        column_list_y = [getattr(columns_y, c) for c in column_types]

        # Quick check, that the columns-object gives unique names
        assert len(set(column_list_x)) == len(column_list_x)
        assert len(set(column_list_y)) == len(column_list_y)

        for segment in segments:
            sbs_created: SegmentDiffs = sbs_res[segment.name]

            # Assert Files Exist ----
            files_to_check = [
                sbs_created.get_path("phase_x"),
                sbs_created.get_path("phase_y"),
            ]
            for file_ in files_to_check:
                assert_file_exists_and_nonempty(tmp_path / file_)


            # Assert Columns Exist ---
            sbs_x = sbs_created.phase_x
            sbs_y = sbs_created.phase_y

            for col_x, col_y in zip(column_list_x, column_list_y):
                assert col_x in sbs_x.columns
                assert col_y in sbs_y.columns


            # TODO: Get BBS reference values (columns need to be renamed)
            # sbs_ref = SegmentDiffs(SBS_DIR / "ref_files", segment.name)

            # sbs_x_ref = sbs_ref.phase_x
            # sbs_y_ref = sbs_ref.phase_y

            # # First absolute value and then the largest difference
            # diff_max_x = (sbs_x - sbs_x_ref).abs().max().max()
            # diff_max_y = (sbs_y - sbs_y_ref).abs().max().max()

            # assert diff_max_x < MAX_DIFF
            # assert diff_max_y < MAX_DIFF


def _write_correction_file(path: Path, label: str):
    corr_file = path / corrections_madx.format(label)
    corr_file.write_text("ktqx2.r1 = ktqx2.r1 + 1e-5;")


def assert_file_exists_and_nonempty(path: Path):
    assert path.exists()
    assert path.stat().st_size
