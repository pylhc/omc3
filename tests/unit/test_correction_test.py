import matplotlib
import pytest

import tfs
from omc3.correction.constants import NAME, ERROR, DIFF, S, VALUE, MDL, PHASE_ADV, MODEL_MATCHED_FILENAME
from omc3.correction.model_appenders import add_coupling_to_model
from omc3.model.constants import TWISS_DAT
from omc3.optics_measurements.constants import EXT
from omc3.scripts.correction_test import correction_test_entrypoint, NOMINAL_MEASUREMENT
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement

from tests.accuracy.test_global_correction import get_skew_params, get_normal_params, _create_fake_measurement


@pytest.mark.basic
@pytest.mark.parametrize('orientation', ('skew', 'normal'))
def test_lhc_corrections(tmp_path, model_inj_beams, orientation):
    """ Checks that correction_test_entrypoint runs and that all the output
    data is there. Very simple test. """
# @pytest.mark.parametrize('orientation', ('normal',))
# def test_lhc_corrections(tmp_path, model_inj_beam1, orientation):
#     model_inj_beams = model_inj_beam1
    beam = model_inj_beams.beam
    correction_params = get_skew_params(beam) if orientation == 'skew' else get_normal_params(beam)
    _create_fake_measurement(tmp_path, model_inj_beams.model_dir, correction_params.twiss)

    output_dir = tmp_path / "Corrections"

    correction_test_entrypoint(
        # show=True,  # debugging
        meas_dir=tmp_path,
        output_dir=output_dir,
        corrections=[correction_params.correction_filename],
        # accelerator class params:
        **model_inj_beams
    )

    assert (output_dir / MODEL_MATCHED_FILENAME).is_file()
    assert (output_dir / NOMINAL_MEASUREMENT).is_dir()

    for directory in [output_dir, output_dir/NOMINAL_MEASUREMENT]:
        tfs_files = list(directory.glob(f"*{EXT}"))
        twiss_files = list(directory.glob(f"twiss*{EXT}"))

        assert (len(tfs_files) - len(twiss_files)) == 9
        for tfs_file in tfs_files:
            assert tfs_file.stat().st_size

            if tfs_file in twiss_files:
                continue

            df = tfs.read(tfs_file)
            assert all(c in df.columns for c in (S, NAME, DIFF, ERROR, VALUE))
            assert df.columns.str.match(f"{PHASE_ADV}.{MDL}").any()

    pdf_files = list(output_dir.glob(f"*.{matplotlib.rcParams['savefig.format']}"))
    assert len(pdf_files) == 7
    for pdf_file in pdf_files:
        assert pdf_file.stat().st_size


def _create_fake_measurement(tmp_path, model_path, twiss_path):
    model_df = tfs.read(model_path / TWISS_DAT, index=NAME)
    model_df = add_coupling_to_model(model_df)

    twiss_df = tfs.read(twiss_path, index=NAME)
    twiss_df = add_coupling_to_model(twiss_df)

    # create fake measurement data
    fake_measurement(
        model=model_df,
        twiss=twiss_df,
        randomize=[],
        outputdir=tmp_path,
    )

