"""
Unit tests for everything Segment-by-Segment related.
"""
from pathlib import Path

from generic_parser import DotDict
import tfs
from omc3.model.constants import Fetcher
from omc3.optics_measurements.constants import NAME, PHASE
from omc3.sbs_propagation import segment_by_segment
from omc3.segment_by_segment.constants import BACKWARD, FORWARD
from tests.conftest import INPUTS

INPUT_SBS: Path = INPUTS / "segment_by_segment"
INPUT_MODELS: Path = INPUTS / "models"

YEAR: int = 2025
OPTICS_30CM_FLAT = "R2025aRP_A30cmC30cmA10mL200cm_Flat.madx"

def test_sbs_no_phase_jump_on_wraparound(tmp_path: Path, model_30cm_flat_beams: DotDict, acc_models_lhc_2025: dict):
    """ Test that no phase jump occurs on phase wraparound in SBS propagation.
    The possible phase jump is the tune, so around 3e-1."""
    beam = model_30cm_flat_beams.beam
    ips = [1, 2, 5, 8]  # problematic IPs are Beam 1 IP2, and Beam 2 IP8 (as IPs are at the beginning/end of twiss)
    segment_by_segment(
        measurement_dir=INPUT_SBS / f"measurement_b{beam}",
        output_dir=tmp_path,
        segments=[f'IP{ip},BPM.12L{ip}.B{beam},BPM.12R{ip}.B{beam}' for ip in ips],
        accel="lhc",
        model_dir=model_30cm_flat_beams.model_dir,
        beam=beam,
        year=str(YEAR),
        fetch=Fetcher.PATH,
        path=acc_models_lhc_2025,
        modifiers=[OPTICS_30CM_FLAT],  # override modifiers from madx jobfile (which are the same but different path)
    )

    for ip in ips:
        for plane in ['x', 'y']:
            phase = tfs.read(tmp_path / f"sbs_phase_{plane}_IP{ip}.tfs", index=NAME)
            for prop in [FORWARD, BACKWARD]:
                max_val = phase.loc[:, f"{prop}{PHASE}{plane.upper()}"].abs().max()
                assert max_val < 1.5e-1, f"Phase jump detected in SbS {prop} propagation Beam {beam} IP {ip} {plane.upper()}!"
