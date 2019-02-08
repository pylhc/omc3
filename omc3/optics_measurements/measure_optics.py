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
from optics_measurements import dpp, tune, phase, beta_from_phase
from optics_measurements import beta_from_amplitude, dispersion, interaction_point, kick


VERSION = '0.4.0'
LOGGER = logging_tools.get_logger(__name__, level_console=logging_tools.INFO)
PLANES = ('X', 'Y')
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
    
    
    try:
        tune_dict = tune.calculate_tunes(measure_input, input_files)
        phase_dict = phase.calculate_phases(measure_input, input_files, tune_dict, common_header)
    except:
        raise ValueError("Phase advance or tune calculation failed: No other calculation will run")
    # try:
    #    coupling.calculate_coupling(measure_input, input_files, phase_dict, tune_dict, common_header)
    # except:
    #    _tb_()

    if measure_input.only_coupling:
        LOGGER.info("Finished as only coupling calculation was requested.")
        return
    try:
        beta_dict = beta_from_phase.calculate_beta_from_phase(
            measure_input, tune_dict, phase_dict, common_header)
    except:
        _tb_()
    if beta_dict["X"]["D"] is None:
        beta_df_dict = {"X": beta_dict["X"]["F"], "Y": beta_dict["Y"]["F"]}
    else:
        beta_df_dict = {"X": beta_dict["X"]["D"], "Y": beta_dict["Y"]["D"]}

    try:
        ratio = beta_from_amplitude.calculate_beta_from_amplitude(measure_input, input_files,
                                                                  tune_dict, phase_dict,
                                                                  beta_df_dict, common_header)
    except:
        _tb_()
    # in the following functions, nothing should change, so we choose the models now
    mad_twiss = measure_input.accelerator.get_model_tfs()
    #  mad_elements = measure_input.accelerator.get_elements_tfs()
    if measure_input.accelerator.excitation:
        mad_ac = measure_input.accelerator.get_driven_tfs()
    else:
        mad_ac = mad_twiss
    try:
        interaction_point.write_betastar_from_phase(
            interaction_point.betastar_from_phase(
                measure_input.accelerator, phase_dict, mad_twiss
            ), common_header, measure_input.outputdir)
    except:
        _tb_()
    
    # dpps = dpp.arrange_dpp(measure_input, input_files, mad_twiss, common_header) 
    
    try:
        dispersion.calculate_dx_from_3d(measure_input, input_files, mad_twiss, common_header, tune_dict)
        dispersion.calculate_ndx_from_3d(measure_input, input_files, mad_twiss, mad_ac, beta_dict["X"]["F"],
                                         dispersion._get_header(common_header, tune_dict, 'getNDx.out'))
    except:
        #LOGGER.info("Calculate dispersion from 3D kicks failed.")
        try:
            dispersion.calculate_orbit_and_dispersion(measure_input, input_files, tune_dict, mad_twiss,
                                                  beta_df_dict, common_header)
        except:
            _tb_()
    try:
        inv_x, inv_y = kick.calculate_kick(measure_input, input_files, mad_twiss, mad_ac, ratio, common_header)
    except:
        _tb_()
    #if measure_input.nonlinear:
    #    try:
    #        resonant_driving_terms.calculate_RDTs(measure_input, input_files, mad_twiss, phase_dict, common_header, inv_x, inv_y)
    #    except:
    #        _tb_()


def _get_header(meas_input):
    return OrderedDict([('Measure_optics:version', VERSION),
                        ('Command', f"{sys.executable} {' '.join(sys.argv)}"),
                        ('CWD', os.getcwd()),
                        ('Date', datetime.datetime.today().strftime("%d. %B %Y, %H:%M:%S")),
                        ('Model_directory', meas_input.accelerator.model_dir)])


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
        
        self.optics_opt = optics_opt
        dpp_values = dpp.calculate_dpoverp(self, optics_opt)

        for plane in PLANES:
            self[plane] = dpp.arrange_dpp(self[plane], dpp_values)
            
        if len(self['X']) + len(self['Y']) == 0:
            raise IOError("No valid input files")

    def dpps(self, plane):
        """
        Gathers measured DPPs from input files corresponding to given plane
        Parameters:
            plane: "X" or "Y"

        Returns:
            numpy array of DPPs
        """
        return np.array([df.DPP for df in self[plane]])

    def zero_dpp_frames(self, plane):
        _zero_dpp_frames = []
        for i in np.argwhere(self.dpps(plane) == 0.0).T[0]:
            _zero_dpp_frames.append(self[plane][i])
        if len(_zero_dpp_frames) > 0:
            return _zero_dpp_frames
        return self._all_frames(plane)

    def _all_frames(self, plane):
        return self[plane]

    def joined_frame(self, plane, columns, zero_dpp=False, how='inner'):
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
        if zero_dpp:
            frames_to_join = self.zero_dpp_frames(plane)
        else:
            frames_to_join = self._all_frames(plane)
        if len(frames_to_join) == 0:
            raise ValueError("No data found")
        joined_frame = pd.DataFrame(self[plane][0]).loc[:, columns]
        for i, df in enumerate(self[plane][1:]):
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
