"""
Measure Optics
--------------

This module contains high-level functions to manage most functionality of ``optics_measurements``.
It provides functions to compute various lattice optics parameters from frequency spectra.
"""
from __future__ import annotations
import datetime
import os
import sys
from copy import deepcopy

import numpy as np

import tfs
from omc3 import __version__ as VERSION
from omc3.definitions.constants import PLANES
from omc3.optics_measurements import (beta_from_amplitude, beta_from_phase,
                                      chromatic, dispersion, interaction_point, kick, phase, rdt,
                                      tune, crdt, coupling)
from omc3.optics_measurements.constants import (
    CHROM_BETA_NAME, EXT, CALIBRATION_FILE, NAME
)
from omc3.utils import iotools, logging_tools

from typing import TYPE_CHECKING 

if TYPE_CHECKING: 
    from generic_parser import DotDict 
    from omc3.optics_measurements.data_models import InputFiles


LOGGER = logging_tools.get_logger(__name__)
LOG_FILE = "measure_optics.log"


def measure_optics(input_files: InputFiles, measure_input: DotDict) -> None:
    """
    Main function to compute various lattice optics parameters from frequency spectra.

    Args:
        input_files: `InputFiles` object containing frequency spectra files (linx/y).
        measure_input: `OpticsInput` object containing analysis settings.

    Returns:
    """
    LOGGER.info(f"Calculating optics parameters - code version {VERSION}")
    iotools.create_dirs(measure_input.outputdir)
    logging_tools.add_module_handler(logging_tools.file_handler(
        os.path.join(measure_input.outputdir, LOG_FILE)))
    tune_dict = tune.calculate(measure_input, input_files)
    common_header = _get_header(measure_input, tune_dict)
    invariants = {}
    phase_dict = {}
    for plane in PLANES:
        phase_dict[plane], out_dfs = phase.calculate(measure_input, input_files, tune_dict, plane)
        phase.write(out_dfs, [common_header]*4, measure_input.outputdir, plane)
        phase.write_special(measure_input, phase_dict[plane]['free'], tune_dict[plane]["QF"], plane)
        if measure_input.only_coupling:
            continue
        beta_df, beta_header = beta_from_phase.calculate(measure_input, tune_dict, phase_dict[plane], common_header, plane)
        beta_from_phase.write(beta_df, beta_header, measure_input.outputdir, plane)

        ratio = beta_from_amplitude.calculate(measure_input, input_files, tune_dict, beta_df, common_header, plane)
        invariants[plane] = kick.calculate(measure_input, input_files, ratio, common_header, plane)
        ip_df = interaction_point.betastar_from_phase(measure_input, phase_dict[plane]['free'])
        interaction_point.write(ip_df, common_header, measure_input.outputdir, plane)
        dispersion.calculate_orbit(measure_input, input_files, common_header, plane)
        dispersion.calculate_dispersion(measure_input, input_files, common_header, plane)
        if plane == "X":
            dispersion.calculate_normalised_dispersion(measure_input, input_files, beta_df, common_header)
    coupling.calculate_coupling(measure_input, input_files, phase_dict, tune_dict, common_header)
    if measure_input.only_coupling:
        return
    if 'rdt' in measure_input.nonlinear:
        iotools.create_dirs(os.path.join(measure_input.outputdir, "rdt"))
        rdt.calculate(measure_input, input_files, tune_dict, phase_dict, invariants, common_header)
    if 'crdt' in measure_input.nonlinear:
        iotools.create_dirs(os.path.join(measure_input.outputdir, "crdt"))
        crdt.calculate(measure_input, input_files, invariants, common_header)
    if measure_input.chromatic_beating:
        chromatic_beating(input_files, measure_input, tune_dict)


def chromatic_beating(input_files: InputFiles, measure_input: DotDict, tune_dict):
    """
    Main function to compute chromatic optics beating.

    Args:
        tune_dict:
        input_files: `InputFiles` object containing frequency spectra files (linx/y).
        measure_input:` OpticsInput` object containing analysis settings.

    Returns:
    """
    dpps = np.array([dpp_val for dpp_val in set(input_files.dpps("X"))])
    if np.max(dpps) - np.min(dpps) == 0.0:
        return
    for plane in PLANES:
        betas = []
        for dpp_val in dpps:
            dpp_meas_input = deepcopy(measure_input)
            dpp_meas_input["dpp"] = dpp_val
            phase_dict, out_dfs = phase.calculate(dpp_meas_input, input_files, tune_dict, plane)
            beta_df, _ = beta_from_phase.calculate(dpp_meas_input, tune_dict, phase_dict, {}, plane)
            betas.append(beta_df)
        output_df = chromatic.calculate_w_and_phi(betas, dpps, input_files, measure_input, plane)
        tfs.write(os.path.join(measure_input.outputdir, f"{CHROM_BETA_NAME}{plane.lower()}{EXT}"), output_df, {}, save_index="NAME")


def _get_header(meas_input, tune_dict):
    compensation = {'model': "by model", 'equation': "by equation", 'none': "None"}
    return dict([('Measure_optics:version', VERSION),
                 ('Command', f"{sys.executable} {' '.join(sys.argv)}"),
                 ('CWD', os.getcwd()),
                 ('Date', datetime.datetime.today().strftime("%d. %B %Y, %H:%M:%S")),
                 ('Model_directory', meas_input.accelerator.model_dir),
                 ('Compensation', compensation[meas_input.compensation]),
                 ('Q1', tune_dict["X"]["QF"]),
                 ('Q2', tune_dict["Y"]["QF"])])


def copy_calibration_files(outputdir, calibrationdir):
    if calibrationdir is None:
        return None
    calibs = {}
    for plane in PLANES:
        cal_file = CALIBRATION_FILE.format(plane=plane.lower())
        iotools.copy_item(os.path.join(calibrationdir, cal_file), os.path.join(outputdir, cal_file))
        calibs[plane] = tfs.read(os.path.join(outputdir, cal_file)).set_index(NAME)
    return calibs
