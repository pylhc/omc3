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

N_EFFECTIVE_FILES = {  # number of effective beta files given number of IPs
        1: 0,
        2: 1,
        3: 3,
}

# Tests ---

@pytest.mark.basic
@pytest.mark.parametrize('beam', [1, 2], ids=ids_str("b{}"))
@pytest.mark.parametrize('ips', ["1", "15", "28", "158"], ids=ids_str("ip{}"))
def test_full_kmod_import(tmp_path: Path, beam: int, ips: str):
    ips = [int(ip) for ip in ips]

    # We have only 1 for IP2 and IP8, but 2 files for IP1 and IP5
    n_files = 1 if (8 in ips) else 2

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
    assert len(list(average_dir.glob(f"*{EXT}"))) == 3 * len(ips) + N_EFFECTIVE_FILES[len(ips)] # AV_BPM: N_BEAM*N_IP, AV_BETASTAR: N_IPs, Effective: see map

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
    if len(ips) == 2 and (8 not in ips):
        # lumi --
        betas = get_betastar_values(beam=beam, ip=1)
        lumi_filename = _get_lumi_filename(betas, ip_a=ips[0], ip_b=ips[1])
        eff_betas = tfs.read(average_dir / lumi_filename)
        eff_betas_ref = tfs.read(REFERENCE_DIR / lumi_filename)
        assert_tfsdataframe_equal(eff_betas_ref, eff_betas, check_like=True)

        # import (reference created with IP1 and IP5) --
        for plane in "xy":
            for ref_path in (_get_bpm_reference_path(beam, plane), _get_betastar_reference_path(beam, plane)):
                ref_file = tfs.read(ref_path)
                out_file = tfs.read(tmp_path / ref_path.name)
                assert_tfsdataframe_equal(ref_file, out_file, check_like=True)

    # Check all the combinations are created
    if ips == "158":  # just some quick tests
        # lumi --
        for ip_combinations in [(1, 5), (1, 8), (5, 8)]:
            betas = get_betastar_values(beam=beam, ip=1)
            lumi_filename = _get_lumi_filename(betas, ip_a=ips[0], ip_b=ips[1])
            assert (average_dir / lumi_filename).exists()

