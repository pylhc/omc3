"""
LHC Xsuite Model Creator
------------------------

Model creator for the ``LHC`` that computes the ``twiss`` output with
`xsuite <https://xsuite.readthedocs.io>`_ (``xtrack``) instead of ``MAD-X``.

The accelerator *sequence* is built (and its tunes/coupling matched) by ``MAD-X``
and saved to a ``.seq`` file - exactly as done for the ``MAD-NG`` response
(:mod:`omc3.correction.response_madng`). That sequence is loaded into ``xtrack`` and
serialised to an xtrack lattice ``.json`` by :func:`create_xsuite_json`, which is then
loaded to produce the ``twiss.dat`` and ``twiss_elements.dat`` files.

The resulting twiss reproduces the ``MAD-X`` output to ~1e-6 relative on the beta
functions and ~1e-7 on the phase advances.

Not yet supported (falls back to :class:`~omc3.model.model_creators.lhc_model_creator.LhcModelCreator`
by raising a clear error here): AC-dipole / ADT driven excitation (see the next migration milestone).

:author: OMC Team
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tfs

from omc3.model.accelerators.accelerator import AccExcitationMode
from omc3.model.constants import TWISS_DAT, TWISS_ELEMENTS_DAT
from omc3.model.model_creators.lhc_model_creator import LhcModelCreator
from omc3.model.xsuite_bridge import (
    _PROTON_MASS_EV,
    XSUITE_JSON,
    create_xsuite_json,
    load_line,
)
from omc3.optics_measurements.constants import NAME
from omc3.utils import logging_tools

if TYPE_CHECKING:
    import pandas as pd

LOGGER = logging_tools.get_logger(__name__)

# Re-exported from the bridge for backwards compatibility (previously defined here).
__all__ = ["XSUITE_JSON", "LhcXsuiteModelCreator", "create_xsuite_json"]

# Mapping of the columns produced by ``xtrack``'s ``twiss4d`` (lowercase) onto the
# ``MAD-X`` twiss column names (uppercase) that ``omc3`` expects downstream.
_TWISS_COLUMN_MAP: dict[str, str] = {
    "s": "S",
    "betx": "BETX",
    "alfx": "ALFX",
    "bety": "BETY",
    "alfy": "ALFY",
    "mux": "MUX",
    "muy": "MUY",
    "dx": "DX",
    "dy": "DY",
    "dpx": "DPX",
    "dpy": "DPY",
    "x": "X",
    "y": "Y",
    "ddx": "DDX",
    "ddy": "DDY",
    "wx_chrom": "WX",
    "wy_chrom": "WY",
    "phix": "PHIX",
    "phiy": "PHIY",
    "dmux": "DMUX",
    "dmuy": "DMUY",
    "bx_chrom": "DBX",
    "by_chrom": "DBY",
    "r11_edw_teng": "R11",
    "r12_edw_teng": "R12",
    "r21_edw_teng": "R21",
    "r22_edw_teng": "R22",
}

# Strengths / geometry columns taken from the ``xtrack`` element table (``line.get_table``).
_TABLE_COLUMN_MAP: dict[str, str] = {
    "k1l": "K1L",
    "k1sl": "K1SL",
    "k2l": "K2L",
    "k3l": "K3L",
    "k4l": "K4L",
}

# Mapping of ``xtrack`` element classes onto ``MAD-X`` twiss ``KEYWORD`` values.
_KEYWORD_MAP: dict[str, str] = {
    "Drift": "DRIFT",
    "Bend": "SBEND",
    "RBend": "RBEND",
    "Quadrupole": "QUADRUPOLE",
    "Sextupole": "SEXTUPOLE",
    "Octupole": "OCTUPOLE",
    "Multipole": "MULTIPOLE",
    "Marker": "MARKER",
    "Cavity": "RFCAVITY",
    "UniformSolenoid": "SOLENOID",
    "Solenoid": "SOLENOID",
    "": "MARKER",
}

# Final column order, matching the ``MAD-X`` ``twiss.dat`` produced by ``do_twiss_monitors``.
_OUTPUT_COLUMNS: list[str] = [
    "S", "BETX", "ALFX", "BETY", "ALFY", "MUX", "MUY", "DX", "DY", "DPX", "DPY",
    "X", "Y", "DDX", "DDY", "K1L", "K1SL", "K2L", "K3L", "K4L", "WX", "WY",
    "PHIX", "PHIY", "DMUX", "DMUY", "KEYWORD", "DBX", "DBY", "R11", "R12", "R21", "R22",
]


class LhcXsuiteModelCreator(LhcModelCreator):
    """LHC model creator that computes the twiss output with ``xtrack`` instead of ``MAD-X``.

    The sequence is built and matched by ``MAD-X`` (reusing the base ``MAD-X`` script), saved
    to disk, and re-loaded into ``xtrack`` which produces the twiss files.
    """

    def full_run(self) -> None:
        """Build the xsuite lattice ``.json`` and compute + write the twiss files with xtrack.

        The lattice ``.json`` is produced from the MAD-X-built sequence by
        :func:`create_xsuite_json` (the only place MAD-X is used) and then loaded to compute
        the twiss with ``xtrack``.
        """
        accel = self.accel

        if accel.excitation != AccExcitationMode.FREE:
            raise NotImplementedError(
                "The xsuite model creator does not yet support AC-dipole / ADT driven "
                "excitation. Use the default (MAD-X) 'nominal' creator for driven models. "
                "Driven twiss with xsuite is planned for a later migration milestone."
            )

        if accel.beam != 1:
            # Beam 2 needs the reversed-direction handling (beam_direction == -1) that MAD-NG
            # applies via `loaded_sequence.dir`; this is not yet validated for the xsuite twiss,
            # so we refuse rather than silently produce a wrong-direction model.
            raise NotImplementedError(
                "The xsuite model creator currently only supports LHC beam 1. "
                "Use the default (MAD-X) 'nominal' creator for beam 2. "
                "Beam-2 support is planned for a later migration milestone."
            )

        # 1. Prepare the model directory (macros, symlink, modifiers) - unchanged from MAD-X path.
        self.prepare_run()

        json_file = accel.model_dir / XSUITE_JSON

        # 2. Build the xtrack lattice json from the MAD-X-built sequence.
        create_xsuite_json(self, json_file)

        # 3. Load the xsuite lattice and set the reference particle.
        LOGGER.info(f"Loading the xsuite lattice from {json_file}.")
        _env, line = load_line(accel, json_file, self.sequence_name)

        # 4. Compute the twiss (incl. Edwards-Teng coupling terms) and write the output files.
        LOGGER.info("Computing twiss with xtrack.")
        twiss = line.twiss4d(coupling_edw_teng=True, delta0=accel.dpp)
        table = line.get_table(attr=True)
        full_df = _twiss_to_tfs(
            twiss, table, qx=twiss.qx, qy=twiss.qy, energy=accel.energy, beam=accel.beam
        )

        self._write_twiss_files(full_df)

        # 5. Check the output and populate accel.model / accel.elements - unchanged from MAD-X path.
        self.post_run()

    def _write_twiss_files(self, full_df: pd.DataFrame) -> None:
        """Write ``twiss.dat`` (BPMs only) and ``twiss_elements.dat`` (all elements)."""
        from omc3.model.accelerators.accelerator import AccElementTypes

        accel = self.accel
        # Select BPMs by the accelerator's BPM name pattern for twiss.dat.
        bpm_pattern = accel.RE_DICT[AccElementTypes.BPMS]
        bpm_mask = full_df.index.str.match(bpm_pattern, case=False)

        tfs.write(
            accel.model_dir / TWISS_DAT,
            full_df[bpm_mask],
            save_index=NAME,
        )
        tfs.write(
            accel.model_dir / TWISS_ELEMENTS_DAT,
            full_df,
            save_index=NAME,
        )


def _twiss_to_tfs(twiss, table, qx: float, qy: float, energy: float, beam: int) -> tfs.TfsDataFrame:
    """Convert an xtrack twiss (+ element table) into a MAD-X-like ``TfsDataFrame``.

    Args:
        twiss: xtrack ``twiss4d`` table (with ``coupling_edw_teng=True``).
        table: xtrack ``line.get_table(attr=True)`` (for strengths / keyword / element type).
        qx, qy: the fractional+integer tunes, written to the ``Q1``/``Q2`` headers.
        energy: beam energy in GeV, written to the ``ENERGY`` header.

    Returns:
        TfsDataFrame indexed by (upper-cased) element name with the ``MAD-X`` twiss columns.
    """
    tw_df = twiss.to_pandas()
    tab_df = table.to_pandas()

    # Build the output frame indexed by name.
    out = tfs.TfsDataFrame(index=tw_df["name"].to_numpy())

    for xt_col, madx_col in _TWISS_COLUMN_MAP.items():
        if xt_col in tw_df.columns:
            out[madx_col] = tw_df[xt_col].to_numpy()
        else:  # chromatic column not available from twiss4d -> fill with zeros
            out[madx_col] = 0.0

    # Strengths and keyword come from the element table (aligned on name).
    tab_df = tab_df.set_index("name")
    tab_aligned = tab_df.reindex(out.index)
    for xt_col, madx_col in _TABLE_COLUMN_MAP.items():
        # reindex leaves NaN for twiss-only rows (e.g. _end_point); those carry no strength.
        col = tab_aligned[xt_col].fillna(0.0) if xt_col in tab_df.columns else 0.0
        out[madx_col] = col.to_numpy() if hasattr(col, "to_numpy") else col
    element_types = tab_aligned["element_type"].fillna("").to_numpy()
    out["KEYWORD"] = [_KEYWORD_MAP.get(t, str(t).upper()) for t in element_types]

    # Order the columns like MAD-X and upper-case the index (MAD-X TFS convention).
    out = out[_OUTPUT_COLUMNS]
    out.index = out.index.str.upper()
    out.index.name = NAME

    # Fill physically-meaningful headers that omc3 reads downstream (e.g. ENERGY, Q1, Q2).
    out.headers = {
        "NAME": "TWISS",
        "TYPE": "TWISS",
        "SEQUENCE": f"LHCB{beam}",
        "PARTICLE": "PROTON",
        "MASS": _PROTON_MASS_EV / 1e9,
        "CHARGE": 1.0,
        "ENERGY": float(energy),
        "PC": float(np.sqrt(energy**2 - (_PROTON_MASS_EV / 1e9) ** 2)),
        "GAMMA": float(energy / (_PROTON_MASS_EV / 1e9)),
        "Q1": float(qx),
        "Q2": float(qy),
    }
    return out
