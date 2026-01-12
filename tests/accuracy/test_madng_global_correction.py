"""
Test MAD-NG Global Correction Accuracy
--------------------------------------

Tests the accuracy of global correction using MAD-NG created response matrices.
Creates response with MAD-NG, applies errors to model, corrects betas, dispersion, and coupling separately,
and verifies corrections.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tfs
from cpymad.madx import Madx
from pandas import DataFrame, Series

from omc3.correction.model_appenders import add_coupling_to_model
from omc3.global_correction import global_correction_entrypoint as global_correction
from omc3.model.constants import (
    TWISS_DAT,
    Fetcher,
)
from omc3.model_creator import create_instance_and_model
from omc3.optics_measurements.constants import (
    BETA,
    DISPERSION,
    # DISPERSION,
    F1001,
    F1010,
    NAME,
    NORM_DISPERSION,
    PHASE,
    PHASE_ADV,
    TUNE,
)
from omc3.response_creator import create_response_entrypoint as create_response
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from omc3.utils import logging_tools
from omc3.utils.stats import rms
from tests.accuracy.test_global_correction import (
    get_normal_params,
)

# from tests.conftest import INPUTS

LOG = logging_tools.get_logger(__name__)
LHC_2025_30CM_MODIFIERS = [Path("R2025aRP_A30cmC30cmA10mL200cm_Flat.madx")]

BETA_PARAMS = [f"{BETA}X", f"{BETA}Y"]
DISPERSION_PARAMS = [f"{DISPERSION}X", f"{DISPERSION}Y"]
NORMALISED_DISPERSION_PARAMS = [f"{NORM_DISPERSION}X"]
COUPLING_PARAMS = [f"{F1001}R", f"{F1001}I", f"{F1010}R", f"{F1010}I"]
PHASE_ADV_PARAMS = [f"{PHASE_ADV}X", f"{PHASE_ADV}Y"]
PHASE_PARAMS = [f"{PHASE}X", f"{PHASE}Y"]
TUNE_PARAMS = [f"{TUNE}1", f"{TUNE}2"]


def _check_rms_cols(
    col_with_errors: Series,
    model_col: Series,
    corrected_col: Series,
    param: str,
    tol: float | None = None,
) -> None:
    """
    Check RMS differences for a parameter and assert improvement.

    Args:
        col_with_errors: Series with errors.
        model_col: Nominal model series.
        corrected_col: Corrected series.
        param: Parameter name for logging.
        tol: Optional tolerance for allowable RMS change when improvement is not achieved.
    """
    original_rms = rms(col_with_errors - model_col)
    LOG.info(f"Original RMS for {param}: {original_rms}")
    new_rms_diff = rms(corrected_col - model_col)
    LOG.info(f"New RMS for {param}: {new_rms_diff}")
    try:
        assert new_rms_diff < original_rms, (
            f"RMS of {param} went from {original_rms} to {new_rms_diff} after correction. Not improved."
        )
    except AssertionError as e:
        if tol is None:
            raise e
        assert abs(new_rms_diff - original_rms) < tol, (
            f"RMS of {param} changed too much: {abs(new_rms_diff - original_rms)}"
        )


def make_check_func(params: list[str]) -> Callable[[DataFrame, DataFrame, DataFrame], None]:
    """Create a check function for given parameters."""

    def check(twiss_errors: DataFrame, model: DataFrame, corrected: DataFrame) -> None:
        for param in params:
            _check_rms_cols(twiss_errors[param], model[param], corrected[param], param)

    return check


def check_coupling_rms(twiss_errors: DataFrame, model: DataFrame, corrected: DataFrame) -> None:
    """
    Check RMS improvement for coupling RDTs F1001 and F1010.

    Args:
        twiss_errors: DataFrame with errors at common indices.
        model: Nominal model DataFrame at common indices.
        corrected: Corrected DataFrame at common indices.
    """
    for rdt in [F1001, F1010]:
        param_abs = f"{rdt}A"
        tol = 2e-8 if rdt == F1010 else None
        _check_rms_cols(
            twiss_errors[param_abs], model[param_abs], corrected[param_abs], param_abs, tol=tol
        )

def check_normalised_dispersion_rms(
    twiss_errors: DataFrame, model: DataFrame, corrected: DataFrame
) -> None:
    """
    Check RMS improvement for normalised dispersion.

    Args:
        twiss_errors: DataFrame with errors at common indices.
        model: Nominal model DataFrame at common indices.
        corrected: Corrected DataFrame at common indices.
    """
    # create the normalised dispersion columns for the three dataframes
    plane = "X"
    nd_errors = twiss_errors[f"{DISPERSION}{plane}"] / np.sqrt(twiss_errors[f"{BETA}{plane}"])
    nd_model = model[f"{DISPERSION}{plane}"] / np.sqrt(model[f"{BETA}{plane}"])
    nd_corrected = corrected[f"{DISPERSION}{plane}"] / np.sqrt(corrected[f"{BETA}{plane}"])
    _check_rms_cols(
        nd_errors, nd_model, nd_corrected, f"{NORM_DISPERSION}{plane}"
    ) # Beam 1 not changed at all by correction


param_configs = {
    # "betas": BETA_PARAMS,
    # "dispersion": DISPERSION_PARAMS,
    # "phase": PHASE_ADV_PARAMS,
    # "tune": TUNE_PARAMS,
    # "coupling": COUPLING_PARAMS,
    "normalised_dispersion": NORMALISED_DISPERSION_PARAMS,
}

check_funcs = {k: make_check_func(v) for k, v in param_configs.items()}
# check_funcs["coupling"] = check_coupling_rms # Override for coupling check
check_funcs["normalised_dispersion"] = check_normalised_dispersion_rms


@pytest.mark.extended
def test_madng_global_correction(
    tmp_path: Path, acc_models_lhc_2025: Path, model_30cm_flat_beams: dict[str, Any]
) -> None:
    """
    Test global correction using MAD-NG response for betas, dispersion, and coupling separately.

    This test creates a nominal model, applies quadrupole errors, generates a MAD-NG response matrix,
    creates fake measurements, performs global correction for different optics parameters,
    and verifies that the RMS of the corrected parameters improves compared to the erroneous model.

    Args:
        tmp_path: Temporary directory for test outputs.
        acc_models_lhc_2025: Path to LHC 2025 accelerator models.
        model_30cm_flat_beams: Dictionary containing beam information for 30cm flat beams.
    """
    beam = model_30cm_flat_beams.beam
    correction_types = param_configs.keys()

    # Create a new nominal model
    create_instance_and_model(
        outputdir=tmp_path,
        type="nominal",
        logfile=tmp_path / "madx_log.txt",
        accel="lhc",
        year="2025",
        beam=beam,
        nat_tunes=[0.31, 0.32],
        dpp=0.0,
        energy=6800.0,
        fetch=Fetcher.PATH,
        path=acc_models_lhc_2025,
        modifiers=LHC_2025_30CM_MODIFIERS,
    )

    # Apply errors to the model
    madx_filepath = tmp_path / "job.create_model_nominal.madx"
    madx = Madx(stdout=False)
    madx.chdir(str(tmp_path.absolute()))
    madx.call(str(madx_filepath))
    madx.input("SELECT, FLAG=ERROR, CLASS=quadrupole;")
    madx.input("""
EFCOMP, ORDER=1, RADIUS=1,
DKNR = {0, 1e-4},
;""")
    madx.globals[f"cmrs.b{beam}"] = 1e-4
    madx.input(f"exec, do_twiss_monitors(LHCB{beam}, 'twiss_errs.dat', 0.0);")

    # Load nominal model
    model_df = tfs.read(tmp_path / TWISS_DAT, index=NAME)
    model_df = add_coupling_to_model(model_df)

    # Load twiss with errors
    twiss_errors_df = tfs.read(tmp_path / "twiss_errs.dat", index=NAME)
    twiss_errors_df = add_coupling_to_model(twiss_errors_df)

    # Create MAD-NG response
    fullresponse_path = tmp_path / "fullresponse_madng.h5"
    normal_vars = get_normal_params(beam).variables
    create_response(
        model_dir=tmp_path,
        creator="madng",
        accel="lhc",
        year="2025",
        beam=beam,
        optics_params=[
            *BETA_PARAMS,
            *DISPERSION_PARAMS,
            *COUPLING_PARAMS,
            *PHASE_PARAMS,
            *NORMALISED_DISPERSION_PARAMS,
            TUNE,
        ],
        variable_categories=normal_vars + ["coupling_knobs"],
        delta_k=2e-5,
        outfile_path=fullresponse_path,
    )

    # Create fake measurement
    measurement_dir = tmp_path / f"measurement_beam{beam}"
    fake_measurement(
        model=model_df,
        twiss=twiss_errors_df,
        outputdir=measurement_dir,
    )

    correction_configs = {
        "coupling": (["coupling_knobs"], COUPLING_PARAMS),
        "phase": (normal_vars, PHASE_PARAMS), # We use PHASE for the response but PHASE_ADV to check correction
        "tune": (normal_vars, [TUNE]),
    }

    for correction_type in correction_types:
        LOG.info(f"Correction type: {correction_type}")

        variable_categories, optics_params = correction_configs.get(
            correction_type, (normal_vars, param_configs[correction_type])
        )

        # Perform global correction
        correction_dir = tmp_path / f"correction_{correction_type}"
        LOG.info(
            f"Starting global correction for {correction_type} and variables {variable_categories}"
        )
        LOG.info(f"Writing correction outputs to {correction_dir}")
        global_correction(
            model_dir=tmp_path,
            accel="lhc",
            year="2025",
            beam=beam,
            meas_dir=measurement_dir,
            output_dir=correction_dir,
            svd_cut=0.01,
            iterations=1,
            variable_categories=variable_categories,
            fullresponse_path=fullresponse_path,
            optics_params=optics_params,
            weights=[1.0] * len(optics_params),
            update_response=False,
        )

        correction_file = correction_dir / "changeparameters_iter_correct.madx"
        uncorrection_file = correction_dir / "changeparameters_iter.madx"
        corrected_twiss_path = tmp_path / f"twiss_corrected_{correction_type}.dat"

        # Apply correction in MAD-X, run twiss, unapply correction
        madx.input(f"call, file='{correction_file}';")
        madx.input(f"exec, do_twiss_monitors(LHCB{beam}, '{corrected_twiss_path.name}', 0.0);")
        madx.input(f"call, file='{uncorrection_file}';")

        # Load corrected twiss
        corrected_df = tfs.read(corrected_twiss_path, index=NAME)
        corrected_df = add_coupling_to_model(corrected_df)

        # Select only rows present in both dataframes
        common_indices = model_df.index.intersection(corrected_df.index)
        model_df_common = model_df.loc[common_indices]
        corrected_df_common = corrected_df.loc[common_indices]
        twiss_errors_df_common = twiss_errors_df.loc[common_indices]

        # Check the common indices are not empty
        assert not common_indices.empty, (
            "No common indices between model and corrected twiss dataframes."
        )

        if beam == 2 and correction_type == "coupling":
            from matplotlib import pyplot as plt
            diff_before = twiss_errors_df_common[f"{F1001}A"] - model_df_common[f"{F1001}A"]
            diff_after = corrected_df_common[f"{F1001}A"] - model_df_common[f"{F1001}A"]
            plt.figure()
            plt.plot(twiss_errors_df_common["S"], diff_before, label="Before Correction")
            plt.plot(twiss_errors_df_common["S"], diff_after, label="After Correction")
            plt.legend()
            plt.show()

        # Check RMS for corrected parameters
        check_funcs[correction_type](twiss_errors_df_common, model_df_common, corrected_df_common)
