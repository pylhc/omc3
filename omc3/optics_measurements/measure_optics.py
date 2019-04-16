"""
.. module: measure_optics

Created on 11/07/18

:author: Lukas Malina

Computes various lattice optics parameters from frequency spectra
"""

import os
import sys
import traceback
import datetime
import re
from collections import OrderedDict
import numpy as np
import pandas as pd

import tfs
from utils import logging_tools, iotools
from optics_measurements import dpp, tune, phase, beta_from_phase, iforest, rdt
from optics_measurements import beta_from_amplitude, dispersion, interaction_point, kick
from optics_measurements.constants import PLANES


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
    common_header = _get_header(measure_input)

    tune_dict = tune.calculate(measure_input, input_files)
    ratio = {}
    for plane in PLANES:
        phase_dict = phase.calculate(measure_input, input_files, tune_dict, common_header, plane)
        if measure_input.only_coupling:
            break
        beta_phase = beta_from_phase.calculate(measure_input, tune_dict, phase_dict, common_header, plane)
        ratio[plane] = beta_from_amplitude.calculate(measure_input, input_files, tune_dict, beta_phase, common_header, plane)

        # in the following functions, nothing should change, so we choose the models now
        mad_ac, mad_twiss = get_driven_and_free_models(measure_input)
        interaction_point.write_betastar_from_phase(interaction_point.betastar_from_phase(measure_input.accelerator, phase_dict, mad_twiss), common_header, measure_input.outputdir, plane)
        dispersion.calculate_orbit(measure_input, input_files, mad_twiss, common_header, plane)
        if not measure_input.three_d:
            dispersion.calculate_dispersion(measure_input, input_files, mad_twiss, common_header, plane)
            if plane == "X":
                dispersion.calculate_normalised_dispersion(measure_input, input_files, mad_twiss,
                                                   beta_phase, common_header)
        elif plane == "X":
            dispersion.calculate_normalised_dispersion_3d(measure_input, input_files, mad_twiss, mad_ac,
                                                          beta_phase, common_header)
    if measure_input.three_d:
        dispersion.calculate_dispersion_3d(measure_input, input_files, mad_twiss, common_header)
    inv_x, inv_y = kick.calculate_kick(measure_input, input_files, mad_twiss, mad_ac, ratio, common_header)
    # coupling.calculate_coupling(measure_input, input_files, phase_dict, tune_dict, common_header)
    #if measure_input.nonlinear:
    #    pass #
        #rdt.calculate_RDTs(measure_input, input_files, mad_twiss, phase_dict, common_header, inv_x, inv_y)


def get_driven_and_free_models(measure_input):
    mad_twiss = measure_input.accelerator.get_model_tfs()
    if measure_input.accelerator.excitation:
        mad_ac = measure_input.accelerator.get_driven_tfs()
    else:
        mad_ac = mad_twiss
    return mad_ac, mad_twiss


def _get_header(meas_input):
    compensation = {'model': "by model", 'equation': "by equation", 'none': "None"}
    return OrderedDict([('Measure_optics:version', VERSION),
                        ('Command', f"{sys.executable} {' '.join(sys.argv)}"),
                        ('CWD', os.getcwd()),
                        ('Date', datetime.datetime.today().strftime("%d. %B %Y, %H:%M:%S")),
                        ('Model_directory', meas_input.accelerator.model_dir),
                        ('Compensation', compensation[meas_input.compensation])])


def _tb_():
    if sys.stdout.isatty():
        err_exc = re.sub(r"line\s([0-9]+)", "\33[1mline \33[38;2;80;160;255m\\1\33[0m\33[21m",
                         traceback.format_exc())
        err_exc = re.sub("File\\s\"([^\"]+)\",", "File \33[38;2;0;255;100m\\1\33[0m", err_exc)
        err_excs = err_exc.split("\n")
        for line in err_excs:
            LOGGER.error(line)
    else:
        LOGGER.error(traceback.format_exc())


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
        if isinstance(files_to_analyse[0], str):
            for file_in in files_to_analyse:
                for plane in PLANES:
                    self[plane].append(tfs.read(f"{file_in}.lin{plane.lower()}").set_index("NAME"))
        else:
            for file_in in files_to_analyse:
                for plane in PLANES:
                    self[plane].append(file_in[plane])

        if len(self['X']) + len(self['Y']) == 0:
            raise IOError("No valid input files")
        
        self.optics_opt = optics_opt
        dpp_values = dpp.calculate_dpoverp(self, optics_opt)
        amp_dpp_values = dpp.calculate_amp_dpoverp(self, optics_opt)

        for plane in PLANES:
            #self[plane] = iforest.clean_with_isolation_forest(self[plane], optics_opt, plane)
            self[plane] = dpp.append_dpp(self[plane], dpp.arrange_dpps(dpp_values))
            self[plane] = dpp.append_amp_dpp(self[plane], amp_dpp_values)

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

    def joined_frame(self, plane, columns, dpp=None, dpp_amp=False, how='inner'):
        """
        Constructs merged DataFrame from InputFiles
        Parameters:
            plane:  "X" or "Y"
            columns: list of columns from input files
            zero_dpp: if True merges only zero-dpp files, default is False
            how: way of merging:  'inner' (intersection) or 'outer' (union), default is 'inner'
        Returns:
            merged DataFrame from InputFiles
        """
        if how not in ['inner', 'outer']:
            raise RuntimeWarning("'how' should be either 'inner' or 'outer', 'inner' will be used.")
        frames_to_join = self.dpp_frames(plane, dpp) if dpp is not None else self._all_frames(plane)
        if dpp_amp:
            frames_to_join = [df for df in frames_to_join if df.DPP_AMP > 0]
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

    def calibrate(self, calibs):
        if calibs is None:
            return
        for plane in PLANES:
            for i in range(len(self[plane])):
                data = pd.merge(self[plane][i].loc[:, ["AMP" + plane]], calibs[plane], how='left',
                                left_index=True, right_index=True).fillna(
                    value={"CALIBRATION": 1., "ERROR_CALIBRATION": 0.})
                self[plane][i]["AMP" + plane] = self[plane][i].loc[:, "AMP" + plane] * data.loc[:,"CALIBRATION"]
                self[plane][i]["ERRAMP" + plane] = data.loc[:, "ERROR_CALIBRATION"]  # TODO

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
        return frame.loc[:, self.get_columns(frame, column)].values


def copy_calibration_files(outputdir, calibrationdir):
    if calibrationdir is None:
        return None
    calibs = {}
    for plane in PLANES:
        cal_file = "calibration_{}.out".format(plane.lower())
        iotools.copy_item(os.path.join(calibrationdir, cal_file), os.path.join(outputdir, cal_file))
        calibs[plane] = tfs.read(os.path.join(outputdir, cal_file)).set_index("NAME")
    return calibs
