from pathlib import Path
import pytest
import tfs

from omc3.kmod_importer import AVERAGE_DIR, import_kmod_results
from omc3.optics_measurements.constants import EXT
from tests.conftest import assert_tfsdataframe_equal, ids_str
from tests.unit.test_kmod_averaging import (
    get_all_tfs_filenames as _get_averaged_filenames,
)
from tests.unit.test_kmod_averaging import (
    get_betastar_values,
    get_measurement_dir,
    get_reference_dir,
)
from tests.unit.test_kmod_import import (
    _get_betastar_reference_path,
    _get_bpm_reference_path,
    get_model_path,
)
from tests.unit.test_kmod_lumi_imbalance import REFERENCE_DIR
from tests.unit.test_kmod_lumi_imbalance import (
    _get_effbetas_filename as _get_lumi_filename,
)


# Tests ---

@pytest.mark.basic
@pytest.mark.parametrize('beam', [1, 2], ids=ids_str("b{}"))
@pytest.mark.parametrize('ips', ["1", "15", "28"], ids=ids_str("ip{}"))
def test_full_kmod_import(tmp_path: Path, beam: int, ips: str):
    ips = [int(ip) for ip in ips]

    # We have only 1 for IP2 and IP8, but 2 files for IP1 and IP5
    n_files = 1 if (2 in ips) else 2

    # Run the import ---
    import_kmod_results(
        meas_paths=[get_measurement_dir(ip=ip, i_meas=i) for ip in ips for i in range(1, n_files+1)],
        beam=beam,
        model=get_model_path(beam), 
        output_dir=tmp_path,
    )

    # OUTPUT CHECKS --------------------------------------------
    # Check the basics, if anything looks weird ---
    assert len(list(tmp_path.glob(f"*{EXT}"))) == 4  # beta_kmod x/y, betastar x/y 
    average_dir = tmp_path / AVERAGE_DIR

    assert average_dir.is_dir()
    assert len(list(average_dir.glob("*.pdf"))) == 3 * len(ips)  # beta, beat and waist per IP  
    assert len(list(average_dir.glob(f"*{EXT}"))) == 3 * len(ips) + (len(ips) == 2) # AV_BPM: N_BEAM*N_IP, AV_BETASTAR: N_IPs, Effective: 1 (only when both)

    # Check the content ---
    # averages --
    for ip in ips:
        # As IP2 and IP8 do not have the same betastar:
        betas = get_betastar_values(beam=beam, ip=ip)
        for out_name in _get_averaged_filenames(ip, betas=betas):
            out_file = tfs.read(average_dir / out_name)
            ref_file = tfs.read(get_reference_dir(ip, n_files=n_files) / out_name)
            assert_tfsdataframe_equal(out_file, ref_file, check_like=True)

        
    # Look at luminosity if we have IP1 and IP5 only.
    if len(ips) > 1 and (2 not in ips):
        # lumi --
        betas = get_betastar_values(beam=beam, ip=1)
        eff_betas = tfs.read(average_dir / _get_lumi_filename(betas))
        eff_betas_ref = tfs.read(REFERENCE_DIR / _get_lumi_filename(betas))
        assert_tfsdataframe_equal(eff_betas_ref, eff_betas, check_like=True)

        # import (reference created with IP1 and IP5) --
        for plane in "xy":
            for ref_path in (_get_bpm_reference_path(beam, plane), _get_betastar_reference_path(beam, plane)):
                ref_file = tfs.read(ref_path)
                out_file = tfs.read(tmp_path / ref_path.name)
                assert_tfsdataframe_equal(ref_file, out_file, check_like=True)


# /accpy/bin/python 
# -m omc3.kmod_importer 
# --meas_paths /user/slops/data/LHC_DATA/OP_DATA/Betabeat/2024-03-22/kmod/60cm/ip1_1 /user/slops/data/LHC_DATA/OP_DATA/Betabeat/2024-03-22/kmod/60cm/ip1_2 /user/slops/data/LHC_DATA/OP_DATA/Betabeat/2024-03-22/kmod/60cm/ip1_3 /user/slops/data/LHC_DATA/OP_DATA/Betabeat/2024-03-22/kmod/60cm/ip2 /user/slops/data/LHC_DATA/OP_DATA/Betabeat/2024-03-22/kmod/60cm/ip5_1 /user/slops/data/LHC_DATA/OP_DATA/Betabeat/2024-03-22/kmod/60cm/ip5_2 /user/slops/data/LHC_DATA/OP_DATA/Betabeat/2024-03-22/kmod/60cm/ip8 
# --output_dir /afs/cern.ch/user/j/jmgray/private/Freq_fake_data/2025-03-31/LHCB1/Results/17-37-22_ANALYSIS 
# --model /afs/cern.ch/user/j/jmgray/private/Freq_fake_data/2025-03-31/LHCB1/Models/B1_60cm_on_mom_kmod_test/twiss_elements.dat 
# --beam 1