from dataclasses import dataclass, fields

import tfs
from omc3.correction.constants import EXPECTED, DIFF
from omc3.optics_measurements.constants import (
    BETA_NAME, AMP_BETA_NAME, ORBIT_NAME, DISPERSION_NAME, NORM_DISP_NAME,
    PHASE_NAME, TOTAL_PHASE_NAME, AMPLITUDE, REAL, IMAG, DISPERSION, NORM_DISPERSION,
    PHASE_ADV, MDL, S, KMOD_BETA_NAME,
    EXT, NAME, DELTA, ERR, DRIVEN_TOTAL_PHASE_NAME, DRIVEN_PHASE_NAME, IP_NAME,
    KMOD_IP_NAME, KICK_NAME, BETA, PHASE, ORBIT, F1001_NAME, F1010_NAME, TUNE, RMS, MASKED)
from omc3.plotting.utils.annotations import ylabels
from tfs import TfsDataFrame
from tfs.collection import TfsCollection, Tfs


# Files ------------------------------------------------------------------------

class OpticsMeasurement(TfsCollection):
    """Class to hold and load the measurements from `omc3.optics_measurements`.

    Arguments:
        directory: The path to the measurement directory, usually the
                   `optics_measurements` output directory.
    """
    INDEX = NAME

    beta_phase = Tfs(BETA_NAME)
    beta_amplitude = Tfs(AMP_BETA_NAME)
    beta_kmod = Tfs(KMOD_BETA_NAME)
    phase = Tfs(PHASE_NAME)
    total_phase = Tfs(TOTAL_PHASE_NAME)
    phase_driven = Tfs(DRIVEN_PHASE_NAME)
    total_phase_driven = Tfs(DRIVEN_TOTAL_PHASE_NAME)
    dispersion = Tfs(DISPERSION_NAME)
    norm_dispersion = Tfs(NORM_DISP_NAME)
    orbit = Tfs(ORBIT_NAME)
    kick = Tfs(KICK_NAME)
    ip = Tfs(IP_NAME)
    ip_kmod = Tfs(KMOD_IP_NAME)
    f1001 = Tfs(F1001_NAME, two_planes=False)
    f1010 = Tfs(F1010_NAME, two_planes=False)

    def _get_filename(self, name, plane="") -> str:
        """ Default way `optics_measurements` filenames are defined,
        where `name` is the first argument in `Tfs` above.
        `plane` is added if `two_planes` is `True` or not given."""
        return f"{name}{plane}{EXT}"

    def read_tfs(self, filename: str) -> TfsDataFrame:
        """ Override for NAME convenience. """
        return tfs.read(self.directory / filename, index=self.INDEX)

    def write_tfs(self, filename: str, data_frame: TfsDataFrame):
        """ Override for NAME convenience. """
        tfs.write(self.directory / filename, data_frame, save_index=self.INDEX)


# Columns ----------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnsAndLabels:
    """ Class to store information about derived columns from the main column.
    For convenience, also labels (e.g. for plotting) related to that column
    are stored in this dataclass.
    """
    # Columns
    _column: str  # Main data column (Measurement)
    _error_column: str = None  # Error on the data
    _model_column: str = None  # Model value of the data
    _delta_column: str = None   # Difference between Measurement and Model
    _error_delta_column: str = None   # Difference between Measurement and Model
    _expected_column: str = None   # Expected (delta) value after a correction
    _error_expected_column: str = None   # Expected error value after a correction
    _diff_correction_column: str = None   # Expected difference coming from correction (models)
    # Labels
    _label: str  = None  # Name for plot axis
    _delta_label: str = None   # Name for delta column on a plot axis
    _text_label: str = None # Name in text
    # Headers:
    _delta_rms_header: str = None
    _expected_rms_header: str = None
    # Other
    needs_plane: bool = True

    def set_plane(self, plane: str):
        """ Fixes the plane in a new object. """
        if not self.needs_plane:
            raise AttributeError("Cannot set the plane of a non-planed definition.")
        values_fixed_plane = {f.name: getattr(self, f.name[1:]).format(plane) for f in fields(self) if f.name[0] == "_"}
        return ColumnsAndLabels(
            needs_plane=False,
            **values_fixed_plane,
        )

    # Properties ----
    @property
    def column(self):
        if self.needs_plane and not any(ph in self._column for ph in ("{}", "{0}")):
            return f"{self._column}{{0}}"
        return self._column

    @property
    def error_column(self):
        if self._error_column:
            return self._error_column
        return f"{ERR}{self.column}"

    # With Model ---
    @property
    def model_column(self):
        if self._model_column:
            return self._model_column
        return f"{self.column}{MDL}"

    @property
    def delta_column(self):
        if self._delta_column:
            return self._delta_column
        return f"{DELTA}{self.column}"

    @property
    def error_delta_column(self):
        if self._error_delta_column:
            return self._error_delta_column
        return f"{ERR}{self.delta_column}"

    # Corrections ---
    @property
    def expected_column(self):
        if self._expected_column:
            return self._expected_column
        return f"{EXPECTED}{DELTA}{self.column}"

    @property
    def error_expected_column(self):
        if self._error_expected_column:
            return self._error_expected_column
        return f"{ERR}{self.expected_column}"

    @property
    def diff_correction_column(self):
        if self._diff_correction_column:
            return self._diff_correction_column
        return f"{DIFF}{self.column}{MDL}"

    # Headers ---
    @property
    def delta_rms_header(self):
        if self._delta_rms_header:
            return self._delta_rms_header
        return f"{self.delta_column}{RMS}"

    @property
    def expected_rms_header(self):
        if self._expected_rms_header:
            return self._expected_rms_header
        return f"{self.expected_column}{RMS}"

    @property
    def delta_masked_rms_header(self):
        return f"{self.delta_rms_header}{MASKED}"

    @property
    def expected_masked_rms_header(self):
        return f"{self.expected_rms_header}{MASKED}"

    # Labels ---
    @property
    def label(self):
        if self._label:
            return self._label
        return self.text_label

    @property
    def text_label(self):
        if self._text_label:
            return self._text_label
        return self.column

    @property
    def delta_label(self):
        if self._delta_label:
            return self._delta_label

        if self.label.startswith("$"):
            return fr"$\Delta {self.label[1:]}"
        return fr"$\Delta$ {self.label}"


# Defined Columns --------------------------------------------------------------
TUNE_COLUMN =            ColumnsAndLabels(TUNE, _expected_column=f"{EXPECTED}{TUNE}{{0}}",  _label=ylabels['tune'], _text_label='tune')
BETA_COLUMN =            ColumnsAndLabels(BETA, _label=ylabels['beta'], _text_label='beta', _delta_label=ylabels['betabeat'])
ORBIT_COLUMN =           ColumnsAndLabels(ORBIT, _label=ylabels['co'], _text_label='orbit')
DISPERSION_COLUMN =      ColumnsAndLabels(DISPERSION, _label=ylabels['dispersion'], _text_label='dispersion')
NORM_DISPERSION_COLUMN = ColumnsAndLabels(NORM_DISPERSION, _label=ylabels['norm_dispersion'], _text_label='normalized dispersion')
PHASE_COLUMN =           ColumnsAndLabels(PHASE, _label=ylabels['phase'], _text_label='phase')
TOTAL_PHASE_COLUMN =     ColumnsAndLabels(PHASE, _label=ylabels['phase'], _text_label='total phase')
PHASE_ADVANCE_COLUMN =   ColumnsAndLabels(f'{PHASE_ADV}{{0}}{MDL}', _label=r'Phase Advance [$2 \pi$]', _text_label='phase advance')
S_COLUMN =               ColumnsAndLabels(S, _label='Location [m]', _text_label='longitudinal location', needs_plane=False)

RDT_AMPLITUDE_COLUMN = ColumnsAndLabels(AMPLITUDE, _label=ylabels['absolute'], _text_label='amplitude', needs_plane=False)  # label needs rdt
RDT_PHASE_COLUMN =     ColumnsAndLabels(PHASE, _label=ylabels['phase'], _text_label='phase', needs_plane=False)  # label needs rdt
RDT_REAL_COLUMN =      ColumnsAndLabels(REAL, _label=ylabels['real'], _text_label='real', needs_plane=False)   # label needs rdt
RDT_IMAG_COLUMN =      ColumnsAndLabels(IMAG, _label=ylabels['imag'], _text_label='imaginary', needs_plane=False)  # label needs rdt


# And Column Mappings ----------------------------------------------------------

""" Map for the x-axis of plots. """
POSITION_COLUMN_MAPPING = {
    'location': S_COLUMN,
    'phase-advance': PHASE_ADVANCE_COLUMN,
}

""" Map the file name to it's main columns and the respective label for a plot. """
FILE_COLUMN_MAPPING = {
    # Based on Filename
    BETA_NAME:        BETA_COLUMN,
    AMP_BETA_NAME:    BETA_COLUMN,
    KMOD_BETA_NAME:   BETA_COLUMN,
    ORBIT_NAME:       ORBIT_COLUMN,
    DISPERSION_NAME:  DISPERSION_COLUMN,
    NORM_DISP_NAME:   NORM_DISPERSION_COLUMN,
    PHASE_NAME:       PHASE_COLUMN,
    TOTAL_PHASE_NAME: TOTAL_PHASE_COLUMN,
}

""" Find the Column Dataclass by column name for RDTs. """
RDT_COLUMN_MAPPING = {c.column: c for c in [RDT_AMPLITUDE_COLUMN, RDT_PHASE_COLUMN, RDT_IMAG_COLUMN, RDT_REAL_COLUMN]}