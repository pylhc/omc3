import pytest

import tfs
from omc3.correction.constants import NAME
from omc3.correction.model_appenders import add_coupling_to_model
from omc3.scripts.correction_test import correction_test_entrypoint, MATCHED_MODEL_NAME, NOMINAL_MEASUREMENT
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement

from tests.accuracy.test_global_correction import get_skew_params, get_normal_params, _create_fake_measurement


@pytest.mark.basic
@pytest.mark.parametrize('orientation', ('skew', 'normal'))
def test_lhc_corrections(tmp_path, model_inj_beams, orientation):
    """ Checks that correction_test_entrypoint runs and that all the output
    data is there. Very simple test. """
    beam = model_inj_beams.beam
    correction_params = get_skew_params(beam) if orientation == 'skew' else get_normal_params(beam)
    _create_fake_measurement(tmp_path, model_inj_beams.model_dir, correction_params.twiss)

    output_dir = tmp_path / "Corrections"

    correction_test_entrypoint(
        meas_dir=tmp_path,
        output_dir=output_dir,
        corrections=[correction_params.correction_filename],
        # accelerator class params:
        **model_inj_beams
    )

    assert (output_dir / MATCHED_MODEL_NAME).is_file()
    assert (output_dir / NOMINAL_MEASUREMENT).is_dir()

    for idx, directory in enumerate([output_dir/NOMINAL_MEASUREMENT, output_dir]):
        tfs_files = list(directory.glob("*.tfs"))
        assert len(tfs_files) == 9 + idx  # +1 for model tfs-file
        for tfs_file in tfs_files:
            assert tfs_file.stat().st_size



def _create_fake_measurement(tmp_path, model_path, twiss_path):
    model_df = tfs.read(model_path / "twiss.dat", index=NAME)
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

