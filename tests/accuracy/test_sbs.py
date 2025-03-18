""" 
Test Segment-by-Segment
-----------------------
"""
from pathlib import Path

import numpy as np
import pytest
import tfs
from generic_parser import DotDict

from omc3.definitions.optics import OpticsMeasurement
from omc3.model import manager
from omc3.model.accelerators.lhc import Lhc
from omc3.model.constants import OPTICS_SUBDIR, TWISS_DAT, TWISS_ELEMENTS_DAT, Fetcher
from omc3.model.model_creators.lhc_model_creator import LhcSegmentCreator
from omc3.optics_measurements.constants import NAME
from omc3.sbs_propagation import segment_by_segment
from omc3.segment_by_segment.constants import logfile
from omc3.segment_by_segment.propagables import (
    ALL_PROPAGABLES,
    AlphaPhase,
    BetaPhase,
    Phase,
    PropagableColumns,
)
from omc3.segment_by_segment.segments import Segment, SegmentDiffs
from omc3.utils import logging_tools
from tests.conftest import INPUTS

LOG = logging_tools.get_logger(__name__)

INPUT_SBS: Path = INPUTS / "segment_by_segment"
INPUT_MODELS: Path = INPUTS / "models"

MAX_DIFF = 1e-10

YEAR = "2025"
OPTICS_30CM_FLAT = "R2025aRP_A30cmC30cmA10mL200cm_Flat.madx"  

class TestSbSLHC:

    @pytest.mark.basic
    @pytest.mark.parametrize("with_correction", [True, False], ids=("with_correction", "no_correction"))
    def test_lhc_segment_creation(self, 
        tmp_path: Path, 
        model_30cm_flat_beams: DotDict, 
        acc_models_lhc_2025: Path, 
        with_correction: bool
        ): 
        """ Tests only the creation of the Segment Models via LhcSegmentCreator. 
        A lot of this is actually done in the sbs_propagation as well, but 
        if things fail in the madx model creation, this is a good place to start looking.
        """
        # Preparation ----------------------------------------------------------
        twiss_elements = tfs.read(model_30cm_flat_beams.model_dir / TWISS_ELEMENTS_DAT, index=NAME)
        beam = model_30cm_flat_beams.beam
        accel_opt = dict(
            accel="lhc",
            year=YEAR,
            beam=beam,
            nat_tunes=[0.31, 0.32],
            dpp=0.0,
            energy=6500,
            modifiers=[acc_models_lhc_2025 / OPTICS_SUBDIR / OPTICS_30CM_FLAT],
        )

        correction_path = None
        if with_correction:
            correction_path = create_error_file(tmp_path)

        iplabel = "IP1"
        segment = Segment(
            name=iplabel,
            start=f"BPM.12L1.B{beam:d}",
            end=f"BPM.12R1.B{beam:d}",
        )
        measurement = OpticsMeasurement(INPUT_SBS / f"measurement_b{beam}")

        propagables = [propg(segment, measurement, twiss_elements) for propg in ALL_PROPAGABLES]
        measureables = [measbl for measbl in propagables if measbl]   # TODO  
        
        accel_inst: Lhc = manager.get_accelerator(accel_opt)
        accel_inst.model_dir = tmp_path  # if set in accel_opt, it tries to load from model_dir, but this is the output dir for the segment-models
        accel_inst.acc_model_path = acc_models_lhc_2025
        
        segment_creator = LhcSegmentCreator(
            segment=segment, 
            measurables=measureables,
            logfile=tmp_path / logfile.format(segment.name),
            accel=accel_inst,
            corrections=correction_path,
        )

        # Actual Run -----------------------------------------------------------
        segment_creator.full_run()

        # Test the output ------------------------------------------------------ 
        assert len(list(tmp_path.glob(f"*{Path(TWISS_DAT).suffix}"))) == 2 + 2 * with_correction  # 2 segment, 2 segment corrected
        
        assert_file_exists_and_nonempty(tmp_path / segment_creator.measurement_madx)
        
        # created in madx (should also have been checked in the post_run() method)
        assert_twiss_contains_segment(tmp_path / segment_creator.twiss_forward, segment.start, segment.end)
        assert_twiss_contains_segment(tmp_path / segment_creator.twiss_backward, segment.end, segment.start)

        if with_correction:
            assert_file_exists_and_nonempty(tmp_path / segment_creator.corrections_madx)
            
            # created in madx (should also have been checked in the post_run() method)
            assert_twiss_contains_segment(tmp_path / segment_creator.twiss_forward_corrected, segment.start, segment.end)
            assert_twiss_contains_segment(tmp_path / segment_creator.twiss_backward_corrected, segment.end, segment.start)
        
    
    @pytest.mark.basic
    @pytest.mark.parametrize("load_model", [True, False], ids=("load_model", "create_model"))
    @pytest.mark.parametrize("with_correction", [True, False], ids=("with_correction", "no_correction"))
    def test_lhc_propagation_sbs(self, 
        tmp_path: Path, 
        model_30cm_flat_beams: DotDict, 
        acc_models_lhc_2025: Path, 
        with_correction: bool,
        load_model: bool, 
        ):
        """Tests the propagation of the measurement via the sbs_propagation function.
        In this test we check that the calculated differences between the propagated models
        and measurements, which are written out in the sbs_*.tfs files, are indeed present and 
        correct.
        """
        # Preparation ---
        if not load_model and not with_correction:
            pytest.skip("Redundant.")  # Already running enough tests and this would test no new code

        sbs_dir: Path = tmp_path / "my_sbs"

        beam = model_30cm_flat_beams.beam
        accel_opt = dict(
            accel="lhc",
            year=YEAR,
            beam=beam,
            nat_tunes=[0.31, 0.32],
            dpp=0.0,
            energy=6800,
            fetch=Fetcher.PATH,
            path=acc_models_lhc_2025,
            modifiers=[OPTICS_30CM_FLAT],
        )

        correction_path = None
        if with_correction:
            correction_path = create_error_file(tmp_path)

        if load_model:
            accel_opt["model_dir"] = model_30cm_flat_beams.model_dir
            accel_opt["nat_tunes"] = None

        segments = [
            Segment("IP1", f"BPM.12L1.B{beam:d}", f"BPM.12R1.B{beam:d}"),
            Segment("IP5", f"BPM.12L5.B{beam:d}", f"BPM.12R5.B{beam:d}"),
        ]
        
        # Run segment creation ---
        sbs_res = segment_by_segment(
            measurement_dir=INPUT_SBS / f"measurement_b{beam}",
            segments=[s.to_input_string() for s in segments],
            output_dir=sbs_dir, 
            corrections=correction_path,
            **accel_opt,
        )

        # Tests ----------------------------------------------------------------
        eps = 1e-8
        diff_max_mapping = {  # some very crude estimates
            Phase: 1e-2,
            BetaPhase: 9e-2,
            AlphaPhase: 1e-1,
        }
        file_name_mapping = {
            Phase: "phase",
            BetaPhase: "beta_phase",
            AlphaPhase: "alpha_phase",
        }
        column_types = ["column", "forward", "backward"]
        if with_correction:
            column_types += ["forward_correction", "backward_correction"]

        for propagable in ALL_PROPAGABLES:
            diff_max: float = diff_max_mapping[propagable]
            file_name: str = file_name_mapping[propagable]

            for plane in "xy":
                columns: PropagableColumns = propagable.columns.planed(plane.upper())  

                # Quick cheks for existing columns ---------------------
                
                column_list = [getattr(columns, c) for c in column_types]

                # Assert the columns-object gives unique names
                assert len(set(column_list)) == len(column_list)

                # In-Depth check per segments ---------------------------
                if propagable is AlphaPhase:
                    meas_df = tfs.read(INPUT_SBS / f"measurement_b{beam}" / f"{file_name_mapping[BetaPhase]}_{plane}.tfs", index=NAME)
                else:
                    meas_df = tfs.read(INPUT_SBS / f"measurement_b{beam}" / f"{file_name}_{plane}.tfs", index=NAME)

                for segment in segments:
                    sbs_created: SegmentDiffs = sbs_res[segment.name]

                    # Assert Files Exist ----
                    assert_file_exists_and_nonempty(tmp_path / sbs_created.get_path(f"{file_name}_{plane}"))

                    # Assert Columns Exist ---
                    sbs_df = getattr(sbs_created, f"{file_name}_{plane}")

                    assert_propagated_measurement_contains_segment(sbs_df, segment.start, segment.end)
                    assert len(sbs_df.index) == len(meas_df.loc[segment.start:segment.end].index)  # works as long as the segment is not looping around

                    for col in column_list:
                        assert col in sbs_df.columns

                    # Assert there is a difference between the propagated models and the measurements
                    if propagable is Phase:
                        assert sum(sbs_df[columns.column] == 0) == 1  # the first entry should be set to 0
                    assert sbs_df[columns.error_column].all()

                    # forward ---
                    assert sbs_df[columns.forward].abs().min() == 0 # at least the first entry should show no difference
                    assert sbs_df[columns.forward].abs().max() > diff_max
                    assert sbs_df[columns.error_forward].all()
                    
                    # backward ---
                    assert sbs_df[columns.backward].abs().min() == 0 # at least the last entry should show no difference
                    assert sbs_df[columns.backward].abs().max() > diff_max  # but some should
                    assert sbs_df[columns.error_backward].all()

                    if with_correction:
                        # forward ---
                        if propagable is Phase:
                            assert np.allclose(sbs_df[columns.forward_correction], sbs_df[columns.forward], atol=eps)
                            assert np.allclose(sbs_df[columns.forward_expected], 0, atol=eps)
                        else:  # check relative expectation 
                            assert np.allclose(
                                sbs_df[columns.forward_correction] / sbs_df[columns.column], 
                                sbs_df[columns.forward] / sbs_df[columns.column], 
                                atol=eps
                            )
                            assert np.allclose(sbs_df[columns.forward_expected] / sbs_df[columns.column], 0, atol=3*eps, rtol=eps)
                        assert sbs_df[columns.error_forward_correction].iloc[1:].all()

                        # backward ---
                        if propagable is Phase:
                            assert np.allclose( sbs_df[columns.backward_correction], sbs_df[columns.backward], atol=eps)
                            assert np.allclose(sbs_df[columns.backward_expected], 0, atol=eps)
                        else:  # check relative expectation
                            assert np.allclose(
                                sbs_df[columns.backward_correction] / sbs_df[columns.column], 
                                sbs_df[columns.backward] / sbs_df[columns.column], 
                                atol=eps
                            )
                            assert np.allclose(sbs_df[columns.backward_expected] / sbs_df[columns.column], 0, atol=3*eps, rtol=eps)
                        assert sbs_df[columns.error_backward_correction].iloc[:-1].all()


# Auxiliary Functions ----------------------------------------------------------

def create_error_file(path: Path):
    out_path = path / "my_errors.madx"
    out_path.write_text(
        "ktqx2.r1 = ktqx2.r1 + 1e-5;\n"
        # "kqsx3.r1 = kqsx3.r1 - 1e-5;\n" # introduces coupling, only useful when coupling propagation implemented
        "ktqx1.r5 = ktqx1.r5 - 2.2e-5;\n"
    )
    return out_path


def assert_file_exists_and_nonempty(path: Path):
    assert path.exists()
    assert path.stat().st_size


def assert_twiss_contains_segment(df: Path, start: str, end: str):
    assert_file_exists_and_nonempty(df)
    df = tfs.read(df, index=NAME)
    assert df.index[1] == start  # first entry contains START maker
    assert df.index[-2] == end   # last entry contains END maker


def assert_propagated_measurement_contains_segment(df: Path, start: str, end: str):
    assert df.index[0] == start
    assert df.index[-1] == end 