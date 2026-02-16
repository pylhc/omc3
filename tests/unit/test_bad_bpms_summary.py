import logging

import pytest
import tfs
from tfs.testing import assert_tfs_frame_equal

from omc3.scripts.bad_bpms_summary import (
    HARPY,
    IFOREST,
    NAME,
    SOURCE,
    bad_bpms_summary,
    merge_reasons,
)
from tests.conftest import INPUTS


@pytest.mark.extended
def test_bad_bpms_summary(tmp_path, caplog):

    outfile = tmp_path / "bad_bpms_summary.tfs"
    with caplog.at_level(logging.INFO):
        df_eval = bad_bpms_summary(
            root=INPUTS,
            outfile=outfile,
            dates=["bad_bpms"],
            accel_glob="LHCB1",
            print_percentage=50,
        )

    assert df_eval is not None
    assert "Unknown reason" not in caplog.text

    # Test Data has been written
    df_eval = merge_reasons(df_eval).reset_index(drop=True)
    df_eval["REASONS"] = df_eval["REASONS"].astype("str")
    assert_tfs_frame_equal(df_eval, tfs.read(outfile))

    # Test some random BPMs
    not_in_model = ["BPMSI.A4R6.B1", ]
    for bpm in not_in_model:
        assert bpm not in df_eval[NAME].tolist()

    iforest_bpms = ["BPM.27R8.B1", "BPMS.2R1.B1"]
    eval_iforest_bpms = df_eval.loc[df_eval[SOURCE] == IFOREST, NAME].tolist()
    for bpm in iforest_bpms:
        assert bpm in eval_iforest_bpms
        assert bpm in caplog.text

    harpy_bpms = ["BPMSE.4L6.B1", "BPM.31L5.B1"]
    eval_harpy_bpms = df_eval.loc[df_eval[SOURCE] == HARPY, NAME].tolist()
    for bpm in harpy_bpms:
        assert bpm in eval_harpy_bpms
        assert bpm in caplog.text
