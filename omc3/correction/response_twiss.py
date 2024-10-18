r"""
Response TWISS
--------------

Provides a class to get response matrices from Twiss parameters.

.. warning::
  The responses are only valid for MAD-X Beam 1 and Beam 2 twiss-files, **not for Beam 4** !!
  Also, it only works properly for on-orbit twiss files.


The calculation is based on formulas in [#FranchiAnalyticformulasrapid2017]_, [#TomasReviewlinearoptics2017]_
and is summarized in [#DillyUpdatedGlobalOpticsCorrection2018]_, where the following
equations can be found in Eq. 10 - Eq. 15.


* Beta Response:

.. math::

    \delta \beta_{z,j} = \mp \beta_{z,j} \sum_m \delta K_{1,m} \frac{\beta_{z,m}}{2}
    \frac{cos(2\tau_{z,mj})}{sin(2\pi Q_z)}


* Dispersion Response:

.. math::

    \delta D_{x,j} =&+ \sqrt{\beta_{x,j}} \sum_m (\delta K_{0,m} + \delta K_{1S,m} D_{y,m}
    - \delta K_{1,m} D_{x,m}) \frac{\sqrt{\beta_{x,m}}}{2}
    \frac{cos(\tau_{x,mj})}{sin(\pi Q_x)}
    \\
    \delta D_{y,j} =&- \sqrt{\beta_{y,j}} \sum_m (\delta K_{0S,m}
    - \delta K_{1S,m} D_{x,m} - \delta K_{1,m} D_{y,m}) \frac{\sqrt{\beta_{y,m}}}{2}
    \frac{cos(\tau_{y,mj})}{sin(\pi Q_y)}


* Norm. Dispersion Response: similar as above but with :math:`\frac{1}{\sqrt{\beta}}` linearized

.. math::

    \delta \frac{D_{x,j}}{\sqrt{\beta_{x,j}}} =&+ \sum_m (\delta K_{0,m} + \delta K_{1S,m} D_{y,m}
    - \delta K_{1,m} D_{x,m} ) \frac{\sqrt{\beta_{x,m}}}{2}
    \frac{cos(\tau_{x,mj})}{sin(\pi Q_x)}
    &&+ \frac{D_{x,j}}{\sqrt{\beta_{x,j}}} \delta K_{1,m}
    \frac{\beta_{x,m}}{4}\frac{cos(2\tau_{x,mj})}{2sin(\pi Q_x)}
    \\
    \delta \frac{D_{y,j}}{\sqrt{\beta_{y,j}}} =&- \sum_m (\delta K_{0S,m} - \delta K_{1S,m} D_{x,m}
    - \delta K_{1,m} D_{y,m}) \frac{\sqrt{\beta_{y,m}}}{2}
    \frac{cos(\tau_{y,mj})}{sin(\pi Q_y)}
    &&- \frac{D_{y,j}}{\sqrt{\beta_{y,j}}} \delta K_{1,m}
    \frac{\beta_{y,m}}{4}\frac{cos(2\tau_{y,mj})}{2sin(\pi Q_y)}


* Phase Advance Response:

.. math::

    \delta \Phi_{z,wj} = \pm \sum_m \delta K_{1,m} \frac{\beta_{z,m}}{4}
    \left\{ 2\left[ \Pi_{mj} - \Pi_{mw} + \Pi_{jw} \right] +
    \frac{sin(2\tau_{z,mj}) - sin(2\tau_{z,mw})}{sin(2\pi Q_z)} \right\}


* Tune Response:

.. math::

    \delta Q_z = \pm \sum_m \delta K_{1,m} \frac{\beta_{z,m}}{4\pi}


* Coupling Response:

.. math::

    \delta f_{\substack{\scriptscriptstyle 1001 \\ \scriptscriptstyle 1010},j} =
    \sum_m \delta J_{1,m} \, \frac{\sqrt{\beta_{x,m}\beta_{y,m}}}{4} \,
    \frac{\exp{(i(\Delta\Phi_{x,mj} \mp \Delta\Phi_{y,mj}))}}{1-\exp({2\pi i (Q_x \mp Q_y}))}

For people reading the code, the response matrices are first calculated like:

::

    |  Elements of interest (j) --> ... |
    |Magnets (m)                        |
    |  |                                |
    |  v                                |
    |  .                                |
    |  .                                |
    |  .                                |
    |                                   |

This avoids transposing all vectors individually in the beginning.
At the end (of the calculation) the matrix is then transposed
to fit the :math:`M \cdot \delta K` orientation.

Also :math:`\Delta \Phi_{z,wj}` needs to be multiplied by :math:`2\pi` to be consistent.


.. rubric:: References

..  [#FranchiAnalyticformulasrapid2017]
    A. Franchi et al.,
    Analytic formulas for the rapid evaluation of the orbit response matrix
    and chromatic functions from lattice parameters in circular accelerators. (2017)
    https://arxiv.org/abs/1711.06589

.. [#TomasReviewlinearoptics2017]
    R. Tomas, et al.,
    'Review of linear optics measurement and correction for charged particle
    accelerators.'
    Physical Review Accelerators and Beams, 20(5), 54801. (2017)
    https://doi.org/10.1103/PhysRevAccelBeams.20.054801

..  [#DillyUpdatedGlobalOpticsCorrection2018]
    J. Dilly et al.,
    An updated global optics correction scheme. (2018)
    https://cds.cern.ch/record/2632945/

"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from omc3.correction.response_io import read_varmap
from omc3.correction.sequence_evaluation import check_varmap_file
from omc3.model.accelerators.accelerator import Accelerator
from omc3.optics_measurements.constants import (
    BETA, DISPERSION, F1001, F1010, NORM_DISPERSION, PHASE_ADV, TUNE, S, PHASE
)
from omc3.utils import logging_tools
from omc3.utils.contexts import timeit

if TYPE_CHECKING:
    from collections.abc import Sequence

LOG = logging_tools.get_logger(__name__)

DUMMY_ID = "DUMMY_PLACEHOLDER"
PLANES = ("X", "Y")

# Twiss Response Class ########################################################


class TwissResponse:
    """Provides Response Matrices calculated from sequence, model and given variables.

    Args:
        accel_inst (accelerator): Accelerator Instance (needs to contain elements model).
        variable_categories (list): List of variable categories to get from the accelerator class.
        varmap_or_path (dict, string): mapping of the variables,
            either as dict-structure of Series or path to a pickled-file.
        at_elements (str): Get response matrix for these elements. Can be:
            'bpms': All BPMS (Default)
            'bpms+': BPMS+ used magnets (== magnets defined by variables in varfile)
            'all': All BPMS and Magnets given in the model (Markers are removed)

    """

    ################################
    #            INIT
    ################################

    def __init__(self, accel_inst, variable_categories, varmap_or_path, at_elements="bpms"):

        LOG.debug("Initializing TwissResponse.")
        with timeit(lambda t: LOG.debug(f"  Time initializing TwissResponse: {t} s")):
            # Get input
            self._twiss = self._get_model_twiss(accel_inst)
            self._beam_sign = 1 if getattr(accel_inst, "beam", 1) == 1 else -1
            self._variables = accel_inst.get_variables(classes=variable_categories)
            self._var_to_el = self._get_variable_mapping(varmap_or_path)
            self._elements_in = self._get_input_elements()
            self._elements_out = self._get_output_elements(at_elements)

            # calculate all phase advances
            self._phase_advances = get_phase_advances(self._twiss)

            # All responses are calcluated as needed, see getters below!
            # slots for response matrices
            self._beta = None
            self._dispersion = None
            self._phase = None
            self._phase_adv = None
            self._tune = None
            self._coupling = None
            self._beta_beat = None
            self._norm_dispersion = None

            # slots for mapped response matrices
            self._coupling_mapped = None
            self._beta_mapped = None
            self._dispersion_mapped = None
            self._phase_mapped = None
            self._phase_adv_mapped = None
            self._tune_mapped = None
            self._beta_beat_mapped = None
            self._norm_dispersion_mapped = None

    @staticmethod
    def _get_model_twiss(accel_inst):
        """ Load model, but keep only BPMs and Magnets """
        model = accel_inst.elements
        LOG.debug("Removing non-necessary entries:")
        LOG.debug(f"  Entries total: {model.shape[0]}")
        mask = accel_inst.get_element_types_mask(model.index, types=["bpm", "magnet"])
        model = model.loc[mask, :].copy()  # make a copy to suppress "SettingWithCopyWarning"
        LOG.debug(f"  Entries left: {model.shape[0]}")

        # Add Dummy for Phase Calculations
        model.loc[DUMMY_ID, [f"{S}", f"{PHASE_ADV}X", f"{PHASE_ADV}Y"]] = 0.0
        return model

    def _get_variable_mapping(self, varmap_or_path):
        """Get variable mapping as dictionary

        Dev hint: Define _variables first!
        """
        LOG.debug("Converting variables to magnet names.")
        variables = self._variables

        if not len(variables):
            raise ValueError("No variables found. Maybe wrong categories?")

        try:
            mapping = read_varmap(varmap_or_path)
        except AttributeError:
            LOG.debug("Received varmap as dictionary.")
            mapping = varmap_or_path
        else:
            LOG.debug(f"Loaded varmap from file '{varmap_or_path}'")

        # check if all variables can be found
        check_var = [var for var in variables if all(var not in mapping[order] for order in mapping)]
        if len(check_var) > 0:
            raise ValueError(f"Variables '{', '.join(check_var)}' cannot be found in sequence!")

        # drop mapping for unused variables
        [mapping[order].pop(var) for order in mapping for var in mapping[order].keys() if var not in variables]
        return mapping

    def _get_input_elements(self):
        """Return variable names of input elements.

        Dev hint: Define _var_to_el and _twiss first!
        """
        v2e = self._var_to_el
        tw = self._twiss

        el_in = dict.fromkeys(k for k in v2e.keys() if k[-1] == 'L')
        for order in el_in:
            el_order = []
            for var in v2e[order]:
                el_order += upper(v2e[order][var].index)
            el_in[order] = tw.loc[list(set(el_order)), f"{S}"].sort_values().index.tolist()

        for order in ["K0L", "K0SL", "K1L", "K1SL"]:
            if order not in el_in:
                el_in[order] = []
        return el_in

    def _get_output_elements(self, at_elements):
        """Return name-array of elements to use for output.

        Dev hint: Define _elements_in first!
        """
        tw_idx = self._twiss.index

        if isinstance(at_elements, list):
            # elements specified
            if any(el not in tw_idx for el in at_elements):
                LOG.warning("One or more specified elements are not in the model.")
            return [idx for idx in tw_idx if idx in at_elements]

        if at_elements == "bpms":
            # bpms only
            return [idx for idx in tw_idx if idx.upper().startswith("B")]

        if at_elements == "bpms+":
            # bpms and the used magnets
            el_in = self._elements_in
            return [idx for idx in tw_idx
                    if (idx.upper().startswith('B') or any(idx in el_in[order] for order in el_in))]

        if at_elements == "all":
            # all, obviously
            return [idx for idx in tw_idx if idx != DUMMY_ID]

    ################################
    #       Response Matrix
    ################################

    def _calc_coupling_response(self):
        """Response Matrix for coupling."""
        LOG.debug("Calculate Coupling Matrix")
        with timeit(lambda t: LOG.debug(f"  Time needed: {t} s")):
            tw = self._twiss
            adv = self._phase_advances
            el_out = self._elements_out
            k1s_el = self._elements_in["K1SL"]
            dcoupl = dict.fromkeys(["1001", "1010"])

            i2pi = 2j * np.pi
            phx = dphi(adv["X"].loc[k1s_el, el_out], tw.Q1).to_numpy()
            phy = dphi(adv["Y"].loc[k1s_el, el_out], tw.Q2).to_numpy()
            bet_term = np.sqrt(tw.loc[k1s_el, f"{BETA}X"].to_numpy() * tw.loc[k1s_el, f"{BETA}Y"].to_numpy())

            for coupling_rdt in ["1001", "1010"]:
                phs_sign = -1 if coupling_rdt == "1001" else 1
                dcoupl[coupling_rdt] = pd.DataFrame(
                    bet_term[:, None] * np.exp(i2pi * (phx + phs_sign * phy)) /
                    (4 * (1 - np.exp(i2pi * (tw.Q1 + phs_sign * tw.Q2)))),
                    index=k1s_el, columns=el_out).transpose()
        return dcoupl

    def _calc_beta_response(self):
        """Response Matrix for delta beta."""
        LOG.debug("Calculate Beta Response Matrix")
        with timeit(lambda t: LOG.debug(f"  Time needed: {t} s")):
            tw = self._twiss
            adv = self._phase_advances
            el_out = self._elements_out
            k1_el = self._elements_in["K1L"]
            dbeta = dict.fromkeys(PLANES)

            for plane in PLANES:
                col_beta = f"{BETA}{plane}"
                q = tw.Q1 if plane == "X" else tw.Q2
                coeff_sign = -1 if plane == "X" else 1

                pi2tau = 2 * np.pi * tau(adv[plane].loc[k1_el, el_out], q)

                dbeta[plane] = pd.DataFrame(
                    tw.loc[el_out, col_beta].to_numpy()[None, :]
                    * tw.loc[k1_el, col_beta].to_numpy()[:, None]
                    * np.cos(2 * pi2tau.to_numpy())
                    * (coeff_sign / (2 * np.sin(2 * np.pi * q))),
                    index=k1_el,
                    columns=el_out,
                ).transpose()

        return dbeta

    def _calc_dispersion_response(self):
        """Response Matrix for delta normalized dispersion."""
        LOG.debug("Calculate Dispersion Response Matrix")
        with timeit(lambda t: LOG.debug(f"  Time needed: {t} s")):
            tw = self._twiss
            adv = self._phase_advances
            el_out = self._elements_out
            els_in = self._elements_in

            sign_map = {
                "X": {"K0L": 1, "K1L": -1, "K1SL": 1},
                "Y": {"K0SL": -1, "K1L": 1, "K1SL": 1},
            }
            col_disp_map = {
                "X": {"K1L": f"{DISPERSION}X", "K1SL": f"{DISPERSION}Y"},
                "Y": {"K1L": f"{DISPERSION}Y", "K1SL": f"{DISPERSION}X"},
            }
            q_map = {"X": tw.Q1, "Y": tw.Q2}
            disp_resp = dict.fromkeys([f"{p}_{t}" for p in sign_map for t in sign_map[p]])

            for plane in sign_map:
                q = q_map[plane]
                col_beta = f"{BETA}{plane}"
                el_types = sign_map[plane].keys()
                els_per_type = [els_in[el_type] for el_type in el_types]

                coeff = np.sqrt(tw.loc[el_out, col_beta].to_numpy()) / (2 * np.sin(np.pi * q))

                for el_in, el_type in zip(els_per_type, el_types):
                    coeff_sign = sign_map[plane][el_type]
                    out_str = f"{plane}_{el_type}"

                    if len(el_in):
                        pi2tau = 2 * np.pi * tau(adv[plane].loc[el_in, el_out], q)
                        bet_term = np.sqrt(tw.loc[el_in, col_beta])

                        try:
                            col_disp = col_disp_map[plane][el_type]
                        except KeyError:
                            pass
                        else:
                            bet_term *= tw.loc[el_in, col_disp]

                        disp_resp[out_str] = (
                            coeff_sign * coeff[None, :] * bet_term.to_numpy()[:, None] * np.cos(pi2tau)
                        ).transpose()
                    else:
                        LOG.debug(f"  No '{el_type}' variables found. "
                                  f"Dispersion Response '{out_str}' will be empty.")
                        disp_resp[out_str] = pd.DataFrame(None, index=el_out)
        return disp_resp

    def _calc_norm_dispersion_response(self):
        """Response Matrix for delta normalized dispersion."""
        LOG.debug("Calculate Normalized Dispersion Response Matrix")
        with timeit(lambda t: LOG.debug(f"  Time needed: {t} s")):
            tw = self._twiss
            adv = self._phase_advances
            el_out = self._elements_out
            els_in = self._elements_in

            sign_map = {
                "X": {"K0L": 1, "K1L": -1, "K1SL": 1},
                "Y": {"K0SL": -1, "K1L": 1, "K1SL": 1},
            }
            col_disp_map = {
                "X": {"K1L": f"{DISPERSION}X", "K1SL": f"{DISPERSION}Y"},
                "Y": {"K1L": f"{DISPERSION}Y", "K1SL": f"{DISPERSION}X"},
            }
            sign_correct_term = {
                "X": {"K1L": 1},
                "Y": {"K1L": -1},
            }
            q_map = {"X": tw.Q1, "Y": tw.Q2}
            disp_resp = dict.fromkeys(["{p:s}_{t:s}".format(p=p, t=t) for p in sign_map for t in sign_map[p]])

            for plane in sign_map:
                q = q_map[plane]
                col_beta = f"{BETA}{plane}"
                el_types = sign_map[plane].keys()
                els_per_type = [els_in[el_type] for el_type in el_types]

                coeff = 1 / (2 * np.sin(np.pi * q))
                coeff_corr = 1 / (4 * np.sin(2 * np.pi * q))
                for el_in, el_type in zip(els_per_type, el_types):
                    coeff_sign = sign_map[plane][el_type]
                    out_str = "{p:s}_{t:s}".format(p=plane, t=el_type)

                    if len(el_in):
                        pi2tau = 2 * np.pi * tau(adv[plane].loc[el_in, el_out], q)
                        beta_in = tw.loc[el_in, col_beta]
                        bet_term = np.sqrt(beta_in)

                        try:
                            col_disp = col_disp_map[plane][el_type]
                        except KeyError:
                            pass
                        else:
                            bet_term *= tw.loc[el_in, col_disp]

                        result = (coeff_sign * coeff * bet_term).to_numpy()[:, None] * np.cos(pi2tau)

                        # correction term
                        try:
                            sign_corr = sign_correct_term[plane][el_type]
                        except KeyError:
                            pass
                        else:
                            norm_disp_corr = (tw.loc[el_out, col_disp] /
                                          np.sqrt(tw.loc[el_out, col_beta]))
                            result += (sign_corr * coeff_corr * norm_disp_corr.to_numpy()[None, :] *
                                       beta_in.to_numpy()[:, None] * np.cos(2 * pi2tau))

                        disp_resp[out_str] = result.transpose()
                    else:
                        LOG.debug(
                            f"  No '{el_type:s}' variables found. "
                            f"Normalized Dispersion Response '{out_str:s}' will be empty."
                        )
                        disp_resp[out_str] = pd.DataFrame(None, index=el_out)
        return disp_resp

    def _calc_phase_advance_response(self):
        """Response Matrix for delta DPhi.

        Reduced to only phase advances between consecutive elements,
        as the 3D-Matrix of all elements exceeds memory space
        (~11000^3 = 1331 Giga Elements)
        --> w = j-1:  DPhi(z,j) = DPhi(x, (j-1)->j)
        """
        LOG.debug("Calculate Phase Advance Response Matrix")
        with timeit(lambda elapsed: LOG.debug(f"  Time needed: {elapsed} s")):
            tw = self._twiss
            adv = self._phase_advances
            k1_el = self._elements_in["K1L"]

            el_out_all = [DUMMY_ID] + self._elements_out  # Add MU[XY] = 0.0 to the start
            el_out = el_out_all[1:]  # in these we are actually interested
            el_out_mm = el_out_all[0:-1]  # elements--

            if len(k1_el) > 0:
                dmu = dict.fromkeys(PLANES)

                pi = pd.DataFrame(tw[f"{S}"].to_numpy()[:, None] < tw[f"{S}"].to_numpy()[None, :],  # pi(i,j) = s(i) < s(j)
                                      index=tw.index, columns=tw.index, dtype=int)

                pi_term = (
                    pi.loc[k1_el, el_out].to_numpy()
                    - pi.loc[k1_el, el_out_mm].to_numpy()
                    + np.diag(pi.loc[el_out, el_out_mm].to_numpy())[None, :]
                )

                for plane in PLANES:
                    col_beta = f"{BETA}{plane}"
                    q = tw.Q1 if plane == "X" else tw.Q2
                    coeff_sign = 1 if plane == "X" else -1

                    pi2tau = 2 * np.pi * tau(adv[plane].loc[k1_el, el_out_all], q)
                    brackets = 2 * pi_term + (
                        (np.sin(2 * pi2tau.loc[:, el_out].to_numpy())
                         - np.sin(2 * pi2tau.loc[:, el_out_mm].to_numpy())) / np.sin(2 * np.pi * q)
                    )
                    dmu[plane] = pd.DataFrame(
                        tw.loc[k1_el, col_beta].to_numpy()[:, None] * brackets * (coeff_sign / (8 * np.pi)),
                        index=k1_el,
                        columns=el_out,
                    ).transpose()
            else:
                LOG.debug("  No 'K1L' variables found. Phase Response will be empty.")
                dmu = {"X": pd.DataFrame(None, index=el_out),
                       "Y": pd.DataFrame(None, index=el_out)}

        return dmu

    def _calc_phase_response(self):
        """Response Matrix for delta DPhi.

        This calculation could also be achieved by applying np.cumsum to the DataFrames of
        _calc_phase_adv_response() (tested!), but _calc_phase_response() is about 4x faster.
        """
        LOG.debug("Calculate Phase Response Matrix")
        with timeit(lambda t: LOG.debug(f"  Time needed: {t} s")):
            tw = self._twiss
            adv = self._phase_advances
            k1_el = self._elements_in["K1L"]
            el_out = self._elements_out

            if len(k1_el) > 0:
                dmu = dict.fromkeys(PLANES)

                pi = pd.DataFrame(tw["S"].to_numpy()[:, None] < tw["S"].to_numpy()[None, :],  # pi(i,j) = s(i) < s(j)
                                      index=tw.index, columns=tw.index, dtype=int)

                pi_term = pi.loc[k1_el, el_out].to_numpy()

                for plane in PLANES:
                    col_beta = f"{BETA}{plane}"
                    q = tw.Q1 if plane == "X" else tw.Q2
                    coeff_sign = 1 if plane == "X" else -1

                    pi2tau = 2 * np.pi * tau(adv[plane].loc[k1_el, [DUMMY_ID] + el_out], q)
                    brackets = 2 * pi_term + (
                        (np.sin(2 * pi2tau.loc[:, el_out].to_numpy())
                         - np.sin(2 * pi2tau.loc[:, DUMMY_ID].to_numpy()[:, None])
                        ) / np.sin(2 * np.pi * q)
                    )
                    dmu[plane] = pd.DataFrame(
                        tw.loc[k1_el, col_beta].to_numpy()[:, None] * brackets * (coeff_sign / (8 * np.pi)),
                        index=k1_el,
                        columns=el_out,
                    ).transpose()
            else:
                LOG.debug("  No 'K1L' variables found. Phase Response will be empty.")
                dmu = {"X": pd.DataFrame(None, index=el_out),
                       "Y": pd.DataFrame(None, index=el_out)}

        return dmu

    def _calc_tune_response(self):
        """Response vectors for Tune."""
        LOG.debug("Calculate Tune Response Matrix")
        with timeit(lambda t: LOG.debug(f"  Time needed: {t} s")):
            tw = self._twiss
            k1_el = self._elements_in["K1L"]

            if len(k1_el) > 0:
                dtune = dict.fromkeys(PLANES)

                dtune["X"] = 1 / (4 * np.pi) * tw.loc[k1_el, [f"{BETA}X"]].transpose()
                dtune["X"].index = ["DQX"]

                dtune["Y"] = -1 / (4 * np.pi) * tw.loc[k1_el, [f"{BETA}Y"]].transpose()
                dtune["Y"].index = ["DQY"]
            else:
                LOG.debug("  No 'K1L' variables found. Tune Response will be empty.")
                dtune = {"X": pd.DataFrame(None, index=["DQX"]),
                         "Y": pd.DataFrame(None, index=["DQY"])}
        return dtune

    ################################
    #       Normalizing
    ################################

    def _normalize_beta_response(self, beta):
        """ Convert to Beta Beating """
        el_out = self._elements_out
        tw = self._twiss

        beta_norm = dict.fromkeys(beta.keys())
        for plane in beta:
            col = f"{BETA}{plane}"
            beta_norm[plane] = beta[plane].div(tw.loc[el_out, col], axis="index")
        return beta_norm

    ################################
    #       Mapping
    ################################

    def _map_dispersion_response(self, disp):
        """ Maps all dispersion matrices """
        disp_mapped = dict.fromkeys(disp.keys())
        m2v = self._map_to_variables
        for plane in disp:
            mapping = self._var_to_el[plane.split("_")[1]]
            disp_mapped[plane] = m2v(disp[plane], mapping)
        return disp_mapped

    @staticmethod
    def _map_to_variables(df, mapping):
        r"""Maps from magnets to variables using self._var_to_el.
        Could actually be done by matrix multiplication :math:'A \cdot var_to_el',
         yet, as var_to_el is very sparsely populated, looping is easier.

        Args:
            df: DataFrame or dictionary of DataFrames to map
            mapping: mapping to be applied (e.g. var_to_el[order])
        Returns:
            DataFrame or dictionary of mapped DataFrames
        """

        def map_fun(df, mapping):
            """ Actual mapping function """
            df_map = pd.DataFrame(index=df.index, columns=mapping.keys())
            for var, magnets in mapping.items():
                df_map[var] = df.loc[:, upper(magnets.index)].mul(magnets.to_numpy(), axis="columns").sum(axis="columns")
            return df_map

        # convenience wrapper for dicts
        if isinstance(df, dict):
            mapped = dict.fromkeys(df.keys())
            for plane in mapped:
                mapped[plane] = map_fun(df[plane], mapping)
        else:
            mapped = map_fun(df, mapping)
        return mapped

    ################################
    #          Getters
    ################################

    def get_beta_beat(self, mapped=True):
        """ Returns Response Matrix for Beta Beating """
        if not self._beta:
            self._beta = self._calc_beta_response()
        if not self._beta_beat:
            self._beta_beat = self._normalize_beta_response(self._beta)

        if mapped and not self._beta_beat_mapped:
            self._beta_beat_mapped = self._map_to_variables(self._beta_beat, self._var_to_el["K1L"])
        return self._beta_beat_mapped if mapped else self._beta_beat

    def get_dispersion(self, mapped=True):
        """ Returns Response Matrix for Dispersion """
        if not self._dispersion:
            self._dispersion = self._calc_dispersion_response()

        if mapped and not self._dispersion_mapped:
            self._dispersion_mapped = self._map_dispersion_response(self._dispersion)

        return self._dispersion_mapped if mapped else self._dispersion

    def get_norm_dispersion(self, mapped=True):
        """ Returns Response Matrix for Normalized Dispersion """
        if not self._norm_dispersion:
            self._norm_dispersion = self._calc_norm_dispersion_response()

        if mapped and not self._norm_dispersion_mapped:
            self._norm_dispersion_mapped = self._map_dispersion_response(self._norm_dispersion)
        return self._norm_dispersion_mapped if mapped else self._norm_dispersion

    def get_phase(self, mapped=True):
        """ Returns Response Matrix for Total Phase """
        if not self._phase:
            self._phase = self._calc_phase_response()

        if mapped and not self._phase_mapped:
            self._phase_mapped = self._map_to_variables(self._phase, self._var_to_el["K1L"])

        if mapped:
            return self._phase_mapped
        else:
            return self._phase

    def get_phase_adv(self, mapped=True):
        """ Returns Response Matrix for Phase Advance """
        if not self._phase_adv:
            self._phase_adv = self._calc_phase_advance_response()

        if mapped and not self._phase_adv_mapped:
            self._phase_adv_mapped = self._map_to_variables(self._phase_adv, self._var_to_el["K1L"])

        if mapped:
            return self._phase_adv_mapped
        else:
            return self._phase_adv

    def get_tune(self, mapped=True):
        """ Returns Response Matrix for the Tunes """
        if not self._tune:
            self._tune = self._calc_tune_response()

        if mapped and not self._tune_mapped:
            self._tune_mapped = self._map_to_variables(self._tune, self._var_to_el["K1L"])

        if mapped:
            return self._tune_mapped
        else:
            return self._tune

    def get_coupling(self, mapped=True):
        """ Returns Response Matrix for the coupling """
        if not self._coupling:
            self._coupling = self._calc_coupling_response()

        if mapped and not self._coupling_mapped:
            self._coupling_mapped = self._map_to_variables(self._coupling, self._var_to_el["K1SL"])

        if mapped:
            return self._coupling_mapped
        else:
            return self._coupling

    def get_variable_names(self):
        return self._variables

    def get_variable_mapping(self, order=None):
        if order is None:
            return self._var_to_el
        else:
            return self._var_to_el[order]

    def get_response_for(self, observables=None) -> dict:  # Dict[str, ???]
        """ Calculates and returns only desired response matrices """
        # calling functions for the getters to call functions only if needed
        def caller(func, plane):
            return func()[plane]

        def disp_caller(func, plane):
            disp = func()
            return response_add(*[disp[k] for k in disp.keys() if k.startswith(plane)])

        def tune_caller(func, _unused):
            tune = func()
            res = pd.concat([tune["X"], tune["Y"]], axis="index", ignore_index=True)
            res.index = [f"{TUNE}1", f"{TUNE}2"]
            return res

        def couple_caller(func, plane):
            # apply() converts empty DataFrames to Series! Cast them back.
            # Also: take care of minus-sign convention!
            sign = -1 if plane[-1] == "R" else 1
            part_func = np.real if plane[-1] == "R" else np.imag
            return sign * pd.DataFrame(func()[plane[:-1]].apply(part_func).astype(np.float64))

        # to avoid if-elif-elif-...
        obs_map = {
            f"{TUNE}": (tune_caller, self.get_tune, None),
            f"{BETA}X": (caller, self.get_beta_beat, "X"),
            f"{BETA}Y": (caller, self.get_beta_beat, "Y"),
            f"{PHASE_ADV}X": (caller, self.get_phase, "X"),
            f"{PHASE_ADV}Y": (caller, self.get_phase, "Y"),
            f"{DISPERSION}X": (disp_caller, self.get_dispersion, "X"),
            f"{DISPERSION}Y": (disp_caller, self.get_dispersion, "Y"),
            f"{NORM_DISPERSION}X": (disp_caller, self.get_norm_dispersion, "X"),
            f"{NORM_DISPERSION}Y": (disp_caller, self.get_norm_dispersion, "Y"),
            f"{F1001}R": (couple_caller, self.get_coupling, "1001R"),
            f"{F1001}I": (couple_caller, self.get_coupling, "1001I"),
            f"{F1010}R": (couple_caller, self.get_coupling, "1010R"),
            f"{F1010}I": (couple_caller, self.get_coupling, "1010I"),
        }

        if observables is None:
            observables = list(obs_map.keys())
        else:
            # rename to PHASE to MU
            observables = [f"{PHASE_ADV}{key[-1]}" if key.startswith(PHASE) else key for key in observables]

        LOG.debug(f"Calculating responses for {observables}.")
        with timeit(lambda t: LOG.debug(f"Total time getting responses: {t} s")):
            response = dict.fromkeys(observables)
            for key in observables:
                # response[key] = obs_map[key][0](*obs_map[key][1:3])
                # I still don't know why there is a sign-change between B1 and B2 (jdilly, 2021)
                response[key] = self._beam_sign * obs_map[key][0](*obs_map[key][1:3])
        return response


# Associated Functions #########################################################


def response_add(*args) -> pd.DataFrame:
    """ Merges two or more Response Matrix DataFrames """
    base_df = args[0]
    for df in args[1:]:
        base_df = base_df.add(df, fill_value=0.0)
    return base_df


def upper(list_of_strings: Sequence[str]) -> Sequence[str]:
    """ Set all items of list to uppercase """
    return [item.upper() for item in list_of_strings]


def get_phase_advances(twiss_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Calculate phase advances between all elements

    Returns:
        Matrices similar to DPhi(i,j) = Phi(j) - Phi(i)
    """
    LOG.debug("Calculating Phase Advances:")
    phase_advance_dict = dict.fromkeys(["X", "Y"])
    with timeit(lambda elapsed: LOG.debug(f"  Phase Advances calculated in {elapsed}s")):
        for plane in PLANES:
            colmn_phase = f"{PHASE_ADV}{plane}"
            phases_mdl = twiss_df.loc[twiss_df.index, colmn_phase]
            # Same convention as in [1]: DAdv(i,j) = Phi(j) - Phi(i)
            phase_advances = pd.DataFrame((phases_mdl.to_numpy()[None, :] - phases_mdl.to_numpy()[:, None]),
                                          index=twiss_df.index,
                                          columns=twiss_df.index)
            # Do not calculate dphi and tau here.
            # only slices of phase_advances as otherwise super slow
            phase_advance_dict[plane] = phase_advances
    return phase_advance_dict


def dphi(data, q):
    """Return dphi from phase advances in data, see Eq. 7 in [#DillyUpdatedGlobalOpticsCorrection2018]_"""
    return data + np.where(data <= 0, q, 0)  # '<=' seems to be what MAD-X does


def tau(data, q):
    """Return tau from phase advances in data, see Eq. 8 in [#DillyUpdatedGlobalOpticsCorrection2018]_"""
    return data + np.where(data <= 0, q / 2, -q / 2)  # '<=' seems to be what MAD-X does


# Wrapper ##################################################################


def create_response(
    accel_inst: Accelerator,
    vars_categories: Sequence[str],
    optics_params: list[str],
) -> dict:
    """ Wrapper to create response via TwissResponse """
    LOG.debug("Creating response via TwissResponse.")
    varmap_path = check_varmap_file(accel_inst, vars_categories)

    with timeit(lambda t: LOG.debug(f"Total time getting TwissResponse: {t} s")):
        tr = TwissResponse(accel_inst, vars_categories, varmap_path)
        response: dict = tr.get_response_for(optics_params)

    if not any([resp.size for resp in response.values()]):
        raise ValueError("Responses are all empty. "
                         f"Are variables {tr.get_variable_names()} correct for '{optics_params}'?")
    return response
