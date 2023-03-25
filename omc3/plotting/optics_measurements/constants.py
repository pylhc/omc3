"""
Constants
---------

Constants and definitions for the ``plotting`` module.
"""
from dataclasses import dataclass
from typing import Optional

from omc3.optics_measurements.constants import (
    S, ERR, DELTA, AMPLITUDE, BETA, PHASE, REAL, IMAG,
    AMP_BETA_NAME, BETA_NAME, CHROM_BETA_NAME, PHASE_NAME,
    SPECIAL_PHASE_NAME, TOTAL_PHASE_NAME, DISPERSION_NAME,
    NORM_DISP_NAME, ORBIT_NAME, KICK_NAME, IP_NAME, PHASE_ADV, MDL
)
from omc3.plotting.utils.annotations import ylabels

IP_POS_DEFAULT = {
    "LHCB1": {
        'IP1': 23519.36962,
        'IP2': 192.923,
        'IP3': 3525.207216,
        'IP4': 6857.491433,
        'IP5': 10189.77565,
        'IP6': 13522.21223,
        'IP7': 16854.64882,
        'IP8': 20175.8654,
    },
    "LHCB2": {
        'IP1': 3195.252584,
        'IP2': 6527.5368,
        'IP3': 9859.973384,
        'IP4': 13192.40997,
        'IP5': 16524.84655,
        'IP6': 19857.13077,
        'IP7': 23189.41498,
        'IP8': 26510.4792,
    }
}

MANUAL_STYLE = {
    # differences to the standard style
    u'lines.markersize': 5.0,
    u'lines.linestyle': u'',
    # u'figure.figsize': [6, 3.8],
}

COMPLEX_NAMES = [p+ext for p in ["1001", "1010"] for ext in "RI"]  # Endings of columns that contain complex data

DEFAULTS = {
    'ncol_legend': 3,
    'errorbar_alpha': .6,
}


@dataclass
class ColumnsAndLabels:
    column: str
    label: str  # for plot
    text_label: str  # in text
    _error_column: str = None
    _delta_column: str = None
    _delta_label: str = None

    @property
    def delta_label(self):
        if self._delta_label:
            return self._delta_label

        if self.label.startswith("$"):
            return f"$\Delta {self.label[1:]}"
        return f"$\Delta$ {self.label}"

    @property
    def delta_column(self):
        if self._delta_column:
            return self._delta_column
        return f"{DELTA}{self.column}"

    @property
    def error_column(self):
        if self._error_column:
            return self._error_column
        return f"{ERR}{self.column}"


XAXIS = {
    'location': ColumnsAndLabels(S, 'Location [m]', 'longitudinal location'),
    'phase-advance': ColumnsAndLabels(f'{PHASE_ADV}{{0}}{MDL}', 'Phase Advance [$2 \pi$]', 'phase advance'),
}


YAXIS = {
    BETA_NAME:        ColumnsAndLabels(BETA, ylabels['beta'], 'beta', _delta_label=ylabels['betabeat']),
    AMP_BETA_NAME:    ColumnsAndLabels(BETA, ylabels['beta'], 'beta', _delta_label=ylabels['betabeat']),
    ORBIT_NAME:       ColumnsAndLabels('', ylabels['co'], 'orbit'),
    DISPERSION_NAME:  ColumnsAndLabels('D', ylabels['dispersion'], 'dispersion'),
    NORM_DISP_NAME:   ColumnsAndLabels('ND', ylabels['norm_dispersion'], 'normalized dispersion'),
    PHASE_NAME:       ColumnsAndLabels(PHASE, ylabels['phase'], 'phase'),
    TOTAL_PHASE_NAME: ColumnsAndLabels(PHASE, ylabels['phase'], 'total phase'),
    'rdt_amp':        ColumnsAndLabels(AMPLITUDE, ylabels['absolute'], 'amplitude'),
    'rdt_phase':      ColumnsAndLabels(PHASE, ylabels['phase'], 'phase'),
    'rdt_real':       ColumnsAndLabels(REAL, ylabels['real'], 'real'),
    'rdt_imag':       ColumnsAndLabels(IMAG, ylabels['imag'], 'imaginary'),
}
