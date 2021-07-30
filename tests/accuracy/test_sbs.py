import itertools
from pathlib import Path

import numpy as np
import pytest
import tfs

from omc3.utils.iotools import create_dirs
from omc3.utils import logging_tools
from omc3.utils import stats
from omc3.model_creator import create_instance_and_model


LOG = logging_tools.get_logger(__name__)

INPUTS = Path(__file__).parent.parent / 'inputs'
SBS_DIR = INPUTS / "sbs"
MAX_DIFF = 1e-10

@pytest.mark.basic
def test_lhc_creation_sbs(tmp_path):
    accel_opt = dict(
        accel="lhc",
        year="2018",
        beam=1,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6.5,
        modifiers=[SBS_DIR / Path("opticsfile.22")]
    )
    iplabel = "IP1"
    _write_correction_file(tmp_path, iplabel)

    create_instance_and_model(
        outputdir = tmp_path,
        type  = "segment",
        label = iplabel,
        start = "BPM.12L1.B1", 
        end   = "BPM.12R1.B1",
        measuredir = SBS_DIR / Path("measurements"),
        **accel_opt
    )
    sbs_x = tfs.read(tmp_path / "sbsphasex_IP1.out", index="NAME")
    sbs_y = tfs.read(tmp_path / "sbsphasey_IP1.out", index="NAME")
    
    ref_dir = SBS_DIR / Path('ref_files')
    sbs_x_ref = tfs.read(ref_dir / "sbsphasex_IP1.out", index="NAME")
    sbs_y_ref = tfs.read(ref_dir / "sbsphasey_IP1.out", index="NAME")
    
    #First absolute value and then the largest difference
    diff_max_x = (sbs_x-sbs_x_ref).abs().max().max() 
    diff_max_y = (sbs_y-sbs_y_ref).abs().max().max()
    
    assert diff_max_x < MAX_DIFF
    assert diff_max_y < MAX_DIFF
    


def _write_correction_file(tmp_path, label):
    create_dirs(tmp_path)
    corr_file = Path("corrections_" + label + ".madx")
    corr_file = tmp_path / corr_file
    f = open(corr_file, "w")
    f.write("ktqx2.r1 = ktqx2.r1 + 1e-5;")
    f.close()