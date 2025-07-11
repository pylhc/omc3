"""
Test Segment-by-Segment
-----------------------
"""
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import pytest
import tfs
from generic_parser import DotDict

from omc3.definitions.optics import OpticsMeasurement
from omc3.model import manager
from omc3.model.constants import OPTICS_SUBDIR, TWISS_DAT, TWISS_ELEMENTS_DAT, Fetcher
from omc3.model.model_creators.lhc_model_creator import LhcSegmentCreator
from omc3.optics_measurements.constants import AMPLITUDE, IMAG, NAME, PHASE, REAL
from omc3.sbs_propagation import segment_by_segment
from omc3.segment_by_segment.constants import logfile
from omc3.segment_by_segment.propagables import (
    ALL_PROPAGABLES,
    F1001,
    F1010,
    AlphaPhase,
    BetaPhase,
    Dispersion,
    Phase,
    PropagableColumns,
)
from omc3.segment_by_segment.segments import Segment, SegmentDiffs
from omc3.utils import logging_tools
from tests.conftest import INPUTS

if TYPE_CHECKING:
    from omc3.model.accelerators.lhc import Lhc

LOG = logging_tools.get_logger(__name__)

INPUT_SBS: Path = INPUTS / "segment_by_segment"
INPUT_MODELS: Path = INPUTS / "models"

MAX_DIFF = 1e-10

YEAR = "2025"
OPTICS_30CM_FLAT = "R2025aRP_A30cmC30cmA10mL200cm_Flat.madx"

class TestCfg(NamedTuple):
    __test__ = False  # avoid PytestCollectionWarning

    diff_max: float
    eps_fwd: float
    eps_bwd: float
    file_name: str

class TestSbSLHC:
    # some constants
    config_map = {  # eps_bwd are high! I think because of the coupling issue, see https://github.com/pylhc/omc3/issues/498
        Phase: TestCfg(1e-2, 1e-8, 5e-4, "phase"),
        BetaPhase: TestCfg(9e-2, 1e-10, 5e-4, "beta_phase"),
        AlphaPhase: TestCfg(1e-1, 1e-8, 8e-2, "alpha_phase"),
        Dispersion: TestCfg(1e-2, None, None, "dispersion"),  # not really working at the moment, see https://github.com/pylhc/omc3/issues/498
        F1001: TestCfg(5e-4, 5e-7, None, "f1001"),
        F1010: TestCfg(5e-4, 5e-6, None, "f1010"),
    }
    eps_rdt_min = 3e-9  # RDTs are not set directly in MAD-X, but converted to R-Matrix. So no value is exact.


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
        assert all(propagable.in_measurement(measurement) for propagable in propagables)  # check if all input files are present for this test!

        accel_inst: Lhc = manager.get_accelerator(accel_opt)
        accel_inst.model_dir = tmp_path  # if set in accel_opt, it tries to load from model_dir, but this is the output dir for the segment-models
        accel_inst.acc_model_path = acc_models_lhc_2025

        segment_creator = LhcSegmentCreator(
            segment=segment,
            measurables=propagables,
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
        column_types = ["column", "forward", "backward"]
        if with_correction:
            column_types += [
                "forward_correction", "backward_correction",
                "forward_expected", "backward_expected"
            ]

        for propagable in ALL_PROPAGABLES:
            if propagable is Dispersion:
                continue  # TODO: not working, see https://github.com/pylhc/omc3/issues/498

            cfg: TestCfg = self.config_map[propagable]
            planes = [REAL, IMAG, AMPLITUDE, PHASE] if propagable.is_rdt() else "xy"

            for plane in planes:
                full_file_name: str = cfg.file_name if propagable.is_rdt() else f"{cfg.file_name}_{plane}"
                columns: PropagableColumns = propagable.columns.planed(plane.upper())

                # print(f"\nTesting {propagable.__name__} {plane}: {columns.column}")  # for debugging

                # Quick cheks for existing columns ---------------------

                column_list = [getattr(columns, c) for c in column_types]

                # Assert the columns-object gives unique names
                assert len(set(column_list)) == len(column_list)

                # In-Depth check per segments ---------------------------
                if propagable is AlphaPhase:
                    # alpha is in the beta-measurement file
                    beta_file = self.config_map[BetaPhase].file_name
                    meas_df = tfs.read(
                        INPUT_SBS / f"measurement_b{beam}" / f"{beta_file}_{plane}.tfs",
                        index=NAME
                    )
                else:
                    meas_df = tfs.read(
                        INPUT_SBS / f"measurement_b{beam}" / f"{full_file_name}.tfs",
                        index=NAME
                    )

                for segment in segments:
                    sbs_created: SegmentDiffs = sbs_res[segment.name]

                    # Assert Files Exist ----
                    assert_file_exists_and_nonempty(tmp_path / sbs_created.get_path(full_file_name))

                    # Assert Columns Exist ---
                    sbs_df: tfs.TfsDataFrame = getattr(sbs_created, full_file_name)

                    assert_propagated_measurement_contains_segment(sbs_df, segment.start, segment.end)
                    assert len(sbs_df.index) == len(meas_df.loc[segment.start:segment.end].index)  # works as long as the segment is not looping around

                    for col in column_list:
                        assert col in sbs_df.columns

                    # Assert there is a difference between the propagated models and the measurements
                    if propagable is Phase:
                        assert sum(sbs_df[columns.column] == 0) == 1  # the first entry should be set to 0
                    assert sbs_df[columns.error_column].all()

                    # Forward / Backward ---
                    forward = (columns.forward, columns.error_forward)
                    backward = (columns.backward, columns.error_backward)
                    for col, err_col in (forward, backward):
                        if propagable.is_rdt():
                            assert sbs_df[col].abs().min() < self.eps_rdt_min  # for RDTs the init value is calculated,
                        else:
                            assert sbs_df[col].abs().min() == 0  # at least the first entry should show no difference

                        assert sbs_df[col].abs().max() > cfg.diff_max
                        assert sbs_df[err_col].all()

                    if with_correction:  # -------------------------------------
                        forward = (cfg.eps_fwd, columns.forward, columns.forward_correction, columns.forward_expected)
                        backward = (cfg.eps_bwd, columns.backward, columns.backward_correction, columns.backward_expected)

                        for idx, (eps, col, correction, expected) in enumerate((forward, backward)):
                            if idx and propagable.is_rdt():
                                # TODO: check backward coupling, see https://github.com/pylhc/omc3/issues/498
                                continue

                            if propagable in [Phase, F1001, F1010]:  # check absolute difference
                                assert_all_close(sbs_df, correction, col, atol=eps)
                                assert_all_close(sbs_df, expected, 0, atol=eps)
                            else:  # check relative difference
                                assert_all_close(sbs_df, correction, col, rel=columns.column, atol=eps)
                                assert_all_close(sbs_df, expected, 0, rel=columns.column, atol=3*eps, rtol=eps)

                        assert sbs_df[columns.error_forward_correction].iloc[1:].all()
                        assert sbs_df[columns.error_backward_correction].iloc[:-1].all()


# Auxiliary Functions ----------------------------------------------------------

def create_error_file(path: Path):
    out_path = path / "my_errors.madx"
    out_path.write_text(
        "ktqx2.r1 = ktqx2.r1 + 1e-5;\n"
        "kqsx3.r1 = kqsx3.r1 - 1e-5;\n" # introduces coupling ! (open bump)
        "ktqx1.r5 = ktqx1.r5 - 2.2e-5;\n"
        "kqsx3.l5 = kqsx3.l5 + 1e-5;\n" # introduces coupling ! (close bump)
    )
    return out_path

def assert_all_close(
    df: pd.DataFrame,
    a: str,
    b: str | float,
    rtol: float = 1e-5,  # default in numpy
    atol: float = 1e-8,  # default in numpy
    rel: str | None = None,
    idx_slice: slice | None = None
    ):
    """ Assert the the values of the a and b columns in df are close.
    If "rel" is given, divide both columns by this column.
    If b is a float, it will be used as the value to compare with.
    """
    if idx_slice is not None:
        df = df.loc[idx_slice]

    a_data = df[a]
    if isinstance(b, str):
        b_data = df[b]
    else:
        b_data = b

    if rel is not None:
        a_data = a_data / df[rel]
        b_data = b_data / df[rel]

    np.testing.assert_allclose(a_data, b_data, atol=atol, rtol=rtol)


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