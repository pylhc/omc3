"""
Measure optics
----------------

:module: optics_measurements.measure_optics
:author: Lukas Malina

Computes various lattice optics parameters from frequency spectra
"""

import os
import sys
import datetime
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import pandas as pd
import tfs
from utils import logging_tools, iotools
from optics_measurements import dpp, tune, phase, beta_from_phase, iforest, chromatic, rdt
from optics_measurements import beta_from_amplitude, dispersion, interaction_point, kick
from optics_measurements.constants import PLANES, ERR, EXT, CHROM_BETA_NAME
from utils.contexts import timeit

VERSION = '0.4.0'
LOGGER = logging_tools.get_logger(__name__, level_console=logging_tools.INFO)
LOG_FILE = "measure_optics.log"


def measure_optics(input_files, measure_input):
    """
    Main function to compute various lattice optics parameters from frequency spectra
    Args:
        input_files: InputFiles object containing frequency spectra files (linx/y)
        measure_input: OpticsInput object containing analysis settings

    Returns:
    """
    LOGGER.info(f"Calculating optics parameters - code version {VERSION}")
    iotools.create_dirs(measure_input.outputdir)
    logging_tools.add_module_handler(logging_tools.file_handler(
        os.path.join(measure_input.outputdir, LOG_FILE)))
    tune_dict = tune.calculate(measure_input, input_files)
    common_header = _get_header(measure_input, tune_dict)
    invariants = {}
    for plane in PLANES:
        phase_dict, out_dfs = phase.calculate(measure_input, input_files, tune_dict, plane)
        phase.write(out_dfs, [common_header, common_header], measure_input.outputdir, plane)
        phase.write_special(measure_input, phase_dict, tune_dict[plane]["QF"], plane)
        if measure_input.only_coupling:
            continue
        beta_df, beta_header = beta_from_phase.calculate(measure_input, tune_dict, phase_dict, common_header, plane)
        beta_from_phase.write(beta_df, beta_header, measure_input.outputdir, plane)

        ratio = beta_from_amplitude.calculate(measure_input, input_files, tune_dict, beta_df, common_header, plane)
        invariants[plane] = kick.calculate(measure_input, input_files, ratio, common_header, plane)
        ip_df = interaction_point.betastar_from_phase(measure_input, phase_dict)
        interaction_point.write(ip_df, common_header, measure_input.outputdir, plane)
        dispersion.calculate_orbit(measure_input, input_files, common_header, plane)
        dispersion.calculate_dispersion(measure_input, input_files, common_header, plane)
        if plane == "X":
            dispersion.calculate_normalised_dispersion(measure_input, input_files, beta_df, common_header)

    # coupling.calculate_coupling(measure_input, input_files, phase_dict, tune_dict, common_header)
    if measure_input.nonlinear:
        iotools.create_dirs(os.path.join(measure_input.outputdir, "rdt"))
        rdt.calculate(measure_input, input_files, tune_dict, invariants, common_header)
    if measure_input.chromatic_beating:
        chromatic_beating(input_files, measure_input, tune_dict)


def chromatic_beating(input_files, measure_input, tune_dict):
    """
    Main function to compute chromatic optics beating
    Args:
        tune_dict:
        input_files: InputFiles object containing frequency spectra files (linx/y)
        measure_input: OpticsInput object containing analysis settings

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
            beta_df, _ = beta_from_phase.calculate(dpp_meas_input, tune_dict, phase_dict, OrderedDict(), plane)
            betas.append(beta_df)
        output_df = chromatic.calculate_w_and_phi(betas, dpps, input_files, measure_input, plane)
        tfs.write(os.path.join(measure_input.outputdir, f"{CHROM_BETA_NAME}{plane.lower()}{EXT}"), output_df, {}, save_index="NAME")


def _get_header(meas_input, tune_dict):
    compensation = {'model': "by model", 'equation': "by equation", 'none': "None"}
    return OrderedDict([('Measure_optics:version', VERSION),
                        ('Command', f"{sys.executable} {' '.join(sys.argv)}"),
                        ('CWD', os.getcwd()),
                        ('Date', datetime.datetime.today().strftime("%d. %B %Y, %H:%M:%S")),
                        ('Model_directory', meas_input.accelerator.model_dir),
                        ('Compensation', compensation[meas_input.compensation]),
                        ('Q1', tune_dict["X"]["QF"]),
                        ('Q2', tune_dict["Y"]["QF"])])


class InputFiles(dict):
    """
    Stores the input files, provides methods to gather quantity specific data

    Public methods:
        get_dpps(plane)
        get_joined_frame(plane, columns, zero_dpp=False, how='inner')
        get_columns(frame, column)
        get_data(frame, column)
    """
    def __init__(self, files_to_analyse, optics_opt):
        super(InputFiles, self).__init__(zip(PLANES, ([], [])))
        read_files = isinstance(files_to_analyse[0], str)
        for file_in in files_to_analyse:
            for plane in PLANES:
                df_to_load = (tfs.read(f"{file_in}.lin{plane.lower()}").set_index("NAME")
                              if read_files else file_in[plane])
                self[plane].append(self._repair_backwards_compatible_frame(df_to_load, plane))

        if len(self['X']) + len(self['Y']) == 0:
            raise IOError("No valid input files")
        dpp_values = dpp.calculate_dpoverp(self, optics_opt)
        LOGGER.info(f"DPPS: {dpp_values}")
        amp_dpp_values = dpp.calculate_amp_dpoverp(self, optics_opt)
        LOGGER.info(f"DPP_AMPS: {amp_dpp_values}")
        for plane in PLANES:
            if optics_opt.isolation_forest:
                self[plane] = iforest.clean_with_isolation_forest(self[plane], optics_opt, plane)
            self[plane] = dpp.append_dpp(self[plane], dpp.arrange_dpps(dpp_values))
            self[plane] = dpp.append_amp_dpp(self[plane], amp_dpp_values)

    @staticmethod  # TODO later remove
    def _repair_backwards_compatible_frame(df, plane):
        """
        Multiplies unscaled amplitudes by 2 to get from complex amplitudes to the real ones
        This is for backwards compatibility with Drive
        """
        df[f"AMP{plane}"] = df.loc[:, f"AMP{plane}"].values * 2
        if f"NATAMP{plane}" in df.columns:
            df[f"NATAMP{plane}"] = df.loc[:, f"NATAMP{plane}"].values * 2
        return df

    def dpps(self, plane):
        """
        Gathers measured DPPs from input files corresponding to given plane
        Parameters:
            plane: "X" or "Y"

        Returns:
            numpy array of DPPs
        """
        return np.array([df.DPP for df in self[plane]])

    def dpp_frames(self, plane, dpp_value):
        dpp_dfs = []
        for i in np.argwhere(np.abs(self.dpps(plane) - dpp_value) < 1e-6).T[0]:
            dpp_dfs.append(self[plane][i])
        if len(dpp_dfs) == 0:
            raise ValueError(f"No data found for dp/p {dpp}")
        return dpp_dfs

    def _all_frames(self, plane):
        return self[plane]

    def joined_frame(self, plane, columns, dpp_value=None, dpp_amp=False, how='inner'):
        """
        Constructs merged DataFrame from InputFiles
        Parameters:
            plane:  "X" or "Y"
            columns: list of columns from input files
            dpp_value: merges only files with given dpp_value
            dpp_amp: merges only files with non-zero dpp amplitude (i.e. 3Dkicks)
            how: way of merging:  'inner' (intersection) or 'outer' (union), default is 'inner'
        Returns:
            merged DataFrame from InputFiles
        """
        if how not in ['inner', 'outer']:
            raise RuntimeWarning("'how' should be either 'inner' or 'outer', 'inner' will be used.")
        frames_to_join = self.dpp_frames(plane, dpp_value) if dpp_value is not None else self._all_frames(plane)
        if dpp_amp:
            frames_to_join = [df for df in frames_to_join if df.DPPAMP > 0]
        if len(frames_to_join) == 0:
            raise ValueError(f"No data found for non-zero |dp/p|")
        joined_frame = pd.DataFrame(frames_to_join[0]).loc[:, columns]
        if len(frames_to_join) > 1:
            for i, df in enumerate(frames_to_join[1:]):
                joined_frame = pd.merge(joined_frame, df.loc[:, columns], how=how, left_index=True,
                                        right_index=True, suffixes=('', '__' + str(i + 1)))
        for column in columns:
            joined_frame.rename(columns={column: column + '__0'}, inplace=True)
        return joined_frame

    def bpms(self, plane=None, dpp_value=None):
        if plane is None:
            return self.bpms(plane="X", dpp_value=dpp_value).intersection(self.bpms(plane="Y", dpp_value=dpp_value))
        indices = [df.index for df in (self.dpp_frames(plane, dpp_value) if dpp_value is not None else self._all_frames(plane))]
        for ind in indices[1:]:
            indices[0] = indices[0].intersection(ind)
        return indices[0]

    def calibrate(self, calibs):
        if calibs is None:
            return
        for plane in PLANES:
            for i in range(len(self[plane])):
                data = pd.merge(self[plane][i].loc[:, ["AMP" + plane]], calibs[plane], how='left',
                                left_index=True, right_index=True).fillna(
                    value={"CALIBRATION": 1., "ERROR_CALIBRATION": 0.})
                self[plane][i][f"AMP{plane}"] = self[plane][i].loc[:, f"AMP{plane}"] * data.loc[:, "CALIBRATION"]
                self[plane][i][f"{ERR}AMP{plane}"] = data.loc[:, "ERROR_CALIBRATION"]  # TODO

    @ staticmethod
    def get_columns(frame, column):
        """
        Returns list of columns of frame corresponding to column in original files
        Parameters:
            frame:  joined frame
            column: name of column in original files
        Returns:
            list of columns
        """
        str_list = list(frame.columns[frame.columns.str.startswith(column + '__')].values)
        new_list = list(map(lambda s: s[len(f"{column}__"):], str_list))
        new_list.sort(key=int)
        return [f"{column}__{x}" for x in new_list]

    def get_data(self, frame, column):
        """
        Returns data in columns of frame corresponding to column in original files
        Parameters:
            frame:  joined frame
            column: name of column in original files
        Returns:
            data in numpy array corresponding to column in original files
        """
        columns = self.get_columns(frame, column)
        return frame.loc[:, columns].values


def copy_calibration_files(outputdir, calibrationdir):
    if calibrationdir is None:
        return None
    calibs = {}
    for plane in PLANES:
        cal_file = f"calibration_{plane.lower()}.out"
        iotools.copy_item(os.path.join(calibrationdir, cal_file), os.path.join(outputdir, cal_file))
        calibs[plane] = tfs.read(os.path.join(outputdir, cal_file)).set_index("NAME")
    return calibs
