import pytest
import tfs 

from tests.conftest import INPUTS, assert_tfsdataframe_equal
from omc3.scripts.bad_bpms_summary import NAME, SOURCE, bad_bpms_summary, IFOREST, HARPY
import logging


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

    # Test Data has been written
    assert df_eval is not None
    assert_tfsdataframe_equal(df_eval, tfs.read(outfile))

    # Test some random BPMs
    not_in_model = ["BPMSI.A4R6.B1", ]
    for bpm in not_in_model:
        assert bpm not in df_eval[NAME].tolist()

    iforest_bpms = ["BPM.27R8.B1", "BPMS.2R1.B1"]
    df_iforest = df_eval[df_eval[SOURCE] == IFOREST]
    for bpm in iforest_bpms:
        assert bpm in df_iforest[NAME].tolist()
        assert bpm in caplog.text

    harpy_bpms = ["BPMSE.4L6.B1", "BPM.31L5.B1"]
    df_harpy = df_eval[df_eval[SOURCE] == HARPY]
    for bpm in harpy_bpms:
        assert bpm in df_harpy[NAME].tolist()
        assert bpm in caplog.text
