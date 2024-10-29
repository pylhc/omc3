"""
Data Models
-----------

Models used in optics measurements to store and pass around data.
"""
from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd

import tfs
from omc3.definitions.constants import PLANES
from omc3.optics_measurements import dpp, iforest
from omc3.optics_measurements.constants import BPM_RESOLUTION, AMPLITUDE, CALIBRATION, ERR, ERR_CALIBRATION
from omc3.utils import logging_tools

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from generic_parser import DotDict

LOGGER = logging_tools.get_logger(__name__)

class InputFiles(dict):
    """
    Stores the input files, provides methods to gather quantity specific data

    Public methods:
        - ``get_dpps`` (plane)
        - ``get_joined_frame`` (plane, columns, zero_dpp=False, how='inner')
        - ``get_columns`` (frame, column)
        - ``get_data`` (frame, column)
    """
    def __init__(self, files_to_analyse: Sequence[str|Path|tfs.TfsDataFrame], optics_opt: DotDict):
        super(InputFiles, self).__init__(zip(PLANES, ([], [])))
        read_files = isinstance(files_to_analyse[0], (str, Path))
        for file_in in files_to_analyse:
            for plane in PLANES:
                df_to_load = (tfs.read(f"{file_in}.lin{plane.lower()}").set_index("NAME", drop=False)
                              if read_files else file_in[plane])
                df_to_load.index.name = None
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
    def _repair_backwards_compatible_frame(df, plane: str):
        """
        Multiplies unscaled amplitudes by 2 to get from complex amplitudes to the real ones.
        This is for backwards compatibility with Drive,
        i.e. harpy has this
        """
        df[f"AMP{plane}"] = df.loc[:, f"AMP{plane}"].to_numpy() * 2
        if f"NATAMP{plane}" in df.columns:
            df[f"NATAMP{plane}"] = df.loc[:, f"NATAMP{plane}"].to_numpy() * 2
        return df

    def dpps(self, plane: str) -> np.ndarray:
        """
        Gathers measured DPPs from input files corresponding to given plane

        Args:
            plane: marking the horizontal or vertical plane, **X** or **Y**.

        Returns:
            A `np.ndarray` of DPPs.
        """
        return np.array([df.DPP for df in self[plane]])

    def dpp_frames(self, plane: str, dpp_value: float | None):
        if dpp_value is None:
            return self._all_frames(plane)

        dpp_dfs = [self[plane][i] for i in self.dpp_frames_indices(plane, dpp_value)]
        if len(dpp_dfs) == 0:
            raise ValueError(f"No data found for dp/p {dpp}")
        return dpp_dfs
    
    def dpp_frames_indices(self, plane: str, dpp_value: float | None):
        """ Return the indices of the frames that match the dpp. """
        if dpp_value is None:
            return list(range(len(self[plane])))

        return np.argwhere(np.abs(self.dpps(plane) - dpp_value) < dpp.DPP_TOLERANCE).T[0]

    def _all_frames(self, plane: str):
        return self[plane]

    def joined_frame(self, 
        plane: str, 
        columns: Sequence[str], 
        dpp_value: float | None = None, 
        dpp_amp: bool = False, 
        how: str = 'inner', 
        dtype: np.dtype = np.float64
    ) -> pd.DataFrame:
        """
        Constructs merged DataFrame from collected DataFrames in InputFiles,
        i.e. from the harpy output of the given measurements.

        The input parameters to this function determine which data will be present in the 
        joined frame:
        which `plane` to use, the `columns` to be included, 
        a `dpp_value` filter (with tolerance from :data:`omc3.optics_measurements.dpp.DPP_TOLERANCE`) 
        and `how` to perform the merge.

        You can also specify what `dtype` the resulting data will have, this should normally be `float64`.

        The columns in the resulting Dataframe will be suffixed by `__#`, starting from `0`
        with increasing integers for each of the input files.    

        Args:
            plane: marking the horizontal or vertical plane, **X** or **Y**.
            columns: list of columns from input files.
            dpp_value: merges only files with given ``dpp_value``.
            dpp_amp: merges only files with non-zero dpp amplitude (i.e. 3Dkicks).
            how: whi way to use for merging: ``inner`` (intersection) or ``outer`` (union),
                default is ``inner``.
            dtype: dtype of the merged DataFrame. Usually `np.float64` should be used, 
                   but in case you have string- or complex- data you will need to set this accordingly 
                   or to `None` to avoid conversion.

        Returns:
            A merged `TfsDataFrame` from `InputFiles`.
        """
        if how not in ['inner', 'outer']:
            raise RuntimeWarning("'how' should be either 'inner' or 'outer', 'inner' will be used.")
        
        # select frames ---
        frames_to_join = self.dpp_frames(plane, dpp_value)
        if dpp_amp:
            frames_to_join = [df for df in frames_to_join if df.DPPAMP > 0]

        if len(frames_to_join) == 0:
            raise ValueError("No data found for non-zero |dp/p|")
        
        # join frames ---
        joined_frame = frames_to_join[0].reindex(columns=columns, fill_value=np.nan)
        if len(frames_to_join) > 1:
            for i, df in enumerate(frames_to_join[1:]):
                joined_frame = pd.merge(joined_frame, df.reindex(columns=columns, fill_value=np.nan),
                                        how=how, left_index=True,
                                        right_index=True, suffixes=('', f'__{i+1}'))
        joined_frame = joined_frame.rename(columns={column: column + '__0' for column in columns})
        if dtype is not None:
            joined_frame = joined_frame.astype(dtype)
        return joined_frame

    def bpms(self, plane=None, dpp_value=None):
        if plane is None:
            return self.bpms(plane="X", dpp_value=dpp_value).intersection(self.bpms(plane="Y", dpp_value=dpp_value))
        indices = [df.index for df in (self.dpp_frames(plane, dpp_value) if dpp_value is not None else self._all_frames(plane))]
        for ind in indices[1:]:
            indices[0] = indices[0].intersection(ind)
        return indices[0]

    def calibrate(self, calibs: dict[str, pd.DataFrame]):
        """
        Use calibration data to rescale amplitude and amplitude error (if present).

        Args:
            calibs (dict): Plane-Dictionary with DataFrames of calibration data.

        """
        if calibs is None:
            return

        for plane in PLANES:
            bpm_resolution = calibs[plane].headers.get(BPM_RESOLUTION, 1e-4)  # TODO: Default of 0.1 mm is LHC specific
            for i in range(len(self[plane])):
                # Merge all measurement BPMs into calibration data (only few BPMs),
                # fill missing values with a scaling of 1 and estimated error of 0.1mm (emaclean estimate)
                data = pd.merge(self[plane][i].loc[:, [f"{AMPLITUDE}{plane}"]], calibs[plane],
                                how='left', left_index=True, right_index=True).fillna(
                    value={CALIBRATION: 1.})  # ERR_CALIBRATION is relative, NaN filled with absolute value below

                # Scale amplitude with the calibration
                self[plane][i][f"{AMPLITUDE}{plane}"] = self[plane][i].loc[:, f"{AMPLITUDE}{plane}"] * data.loc[:, CALIBRATION]

                if f"{ERR}{AMPLITUDE}{plane}" in self[plane][i].columns: 
                    # Sum Amplitude Error (absolute) and Calibration Error (relative)
                    self[plane][i][f"{ERR}{AMPLITUDE}{plane}"] = np.sqrt(
                        self[plane][i][f"{ERR}{AMPLITUDE}{plane}"]**2 +
                        ((self[plane][i][f"{AMPLITUDE}{plane}"] * data.loc[:, ERR_CALIBRATION]).fillna(bpm_resolution))**2
                    )

    @ staticmethod
    def get_columns(frame, column):
        """
        Returns list of columns of frame corresponding to column in original files.

        Args:
            frame:  joined frame.
            column: name of column in original files.

        Returns:
            list of columns.
        """
        str_list = list(frame.columns[frame.columns.str.startswith(column + '__')].to_numpy())
        new_list = list(map(lambda s: s[len(f"{column}__"):], str_list))
        new_list.sort(key=int)
        return [f"{column}__{x}" for x in new_list]

    @ staticmethod
    def get_data(frame, column) -> np.ndarray:
        """
        Returns data in columns of frame corresponding to column in original files.

        Args:
            frame:  joined frame.
            column: name of column in original files.

        Returns:
            A `np.narray` corresponding to column in original files.
        """
        columns = InputFiles.get_columns(frame, column)
        return frame.loc[:, columns].to_numpy(dtype=np.float64)


# DPP Filtering related functions ------------------------------------------------------------------

def check_and_warn_about_offmomentum_data(input_files: InputFiles, plane: str, id_: str = None):
    """ A helper function to check if off-momentum data is present in the input files, 
    but no dpp-value is given by the user. 
    
    See https://github.com/pylhc/omc3/issues/456 .
    """
    on_momentum_files = input_files.dpp_frames_indices(plane, dpp_value=0)
    if len(on_momentum_files) == len(input_files[plane]):
        return  # no off-momentum data, nothing to warn

    msg = (
        "Off-momentum files for analysis found!\n"
        "They will be included"
    )
    
    if id_ is not None:
        msg += f" in the {id_}"

    msg += (
        ", which can make the results more inaccurate.\n"
        "To avoid, specify `analyse_dpp` or run the analysis only on on-momentum files"
    )
    
    if id_ is not None:
        msg += f" or possibly deactivate the {id_}"
    
    msg += "."
    LOGGER.warning(msg)


def filter_for_dpp(to_filter: dict[str, Sequence], input_files: InputFiles, dpp_value: float):
    """ Filter the given data for the given dpp-value. """
    return {plane: values[input_files.dpp_frames_indices(plane, dpp_value)] for plane, values in to_filter.items()}