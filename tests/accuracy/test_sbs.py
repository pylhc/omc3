from pathlib import Path
import shutil

import pytest

from omc3.definitions.optics import OpticsMeasurement
from omc3.model import manager
from omc3.model.model_creators.lhc_model_creator import LhcSegmentCreator
from omc3.segment_by_segment.propagables import get_all_propagables
from omc3.segment_by_segment.segments import Segment, SegmentDiffs
from omc3.segment_by_segment.constants import corrections_madx, logfile

from omc3.sbs_propagation import segment_by_segment
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)

INPUTS = Path(__file__).parent.parent / 'inputs'
SBS_DIR = INPUTS / "sbs"
MAX_DIFF = 1e-10


class TestSbSLHC:
    @pytest.mark.parametrize("beam", [1, ])  #TODO get measurements for Beam 2 
    @pytest.mark.basic
    def test_lhc_segment_creation(self, tmp_path, beam):
        """ Tests only the creation of the Segment Models via LhcSegmentCreator. 
        A lot of this is actually done in the sbs_propagation as well, but 
        if things fail in the madx model creation, this is a good place to start looking.
        """
        accel_opt = dict(
            accel="lhc",
            year="2018",
            beam=beam,
            nat_tunes=[0.31, 0.32],
            dpp=0.0,
            energy=6.5,
            modifiers=[get_model_path(beam) / "opticsfile.24_ctpps2"],
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
        accel_inst.model_dir = tmp_path  # if in accel_opt, tries to load from model_dir

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

    def test_lhc_propagation_sbs(self, tmp_path, model_inj_beams):
        """Runs the segment creation as well as the parameter propagation.
        TODO: make test with creating the model on the fly 
        TODO: make test with loading model with and without output dir 
        TODO: With and without correction
        TODO: Find measurements for beam 2
        (works only within cern network unless afs is mocked)
        """
        beam = model_inj_beams.beam
        if beam == 2:
            return # TODO find measurement

        accel_opt = dict(
            accel="lhc",
            year="2018",
            beam=beam,
            # model_dir=model_inj_beams.model_dir, 
            nat_tunes=[0.31, 0.32],
            dpp=0.0,
            energy=6.5,
            modifiers=[get_model_path(beam) / "opticsfile.24_ctpps2"],
        )

        segments = [
            Segment("IP1", f"BPM.12L1.B{beam:d}", f"BPM.12R1.B{beam:d}"),
            Segment("IP5", f"BPM.12L5.B{beam:d}", f"BPM.12R5.B{beam:d}"),
        ]


        sbs_res = segment_by_segment(
            measurement_dir=SBS_DIR / "measurements",
            segments=[s.to_input_string() for s in segments],
            output_dir=tmp_path, 
            **accel_opt,
        )

        for segment in segments:
            sbs_created: SegmentDiffs = sbs_res[segment.name]
            assert sbs_created.get_path("phase_x").exists()
            assert sbs_created.get_path("phase_y").exists()

            sbs_x = sbs_created.phase_x
            sbs_y = sbs_created.phase_y

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


def get_model_path(beam: int) -> Path:
    return INPUTS / "models" / f"25cm_beam{beam}" 