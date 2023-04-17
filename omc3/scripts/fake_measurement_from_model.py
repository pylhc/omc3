"""
Fake Measurement from Model
---------------------------

Script to generate a pseudo-measurement from a twiss-model.

The `model` the then generated measurement is compared to can be different
from the `twiss` given, e.g. if the `twiss` incorporates errors.


**Arguments:**

*--Required--*

- **twiss** *(PathOrStrOrDataFrame)*:

    Twiss dataframe or path to twiss-file.


*--Optional--*

- **model** *(PathOrStrOrDataFrame)*:

    Alternative Model (Dataframe or Path) to use. If not given, `twiss`
    will be used.


- **outputdir** *(PathOrStr)*:

    Path to the output directory for the fake measurement.


- **parameters** *(str)*:

    Optics parameters to use

    choices: ``('PHASEX', 'PHASEY', 'BETX', 'BETY', 'DX', 'DY', 'NDX', 'F1010', 'F1001')``

    default: ``['PHASEX', 'PHASEY', 'BETX', 'BETY', 'DX', 'DY', 'NDX', 'F1010', 'F1001']``


- **randomize** *(str)*:

    Randomize values and/or errors from gaussian distributions. If not
    randomized, measurement values will be equal to the model values and
    the errors will be equal to the relative error * measurement.

    choices: ``['values', 'errors']``

    default: ``[]``


- **relative_errors** *(float)*:

    Relative errors. Either single value for all paramters orlist of
    values in order of parameters.

    default: ``[0.0]``


- **seed** *(int)*:

    Set random seed.


"""
from pathlib import Path
from typing import Tuple, Sequence, List, Dict

import numpy as np
import pandas as pd
import tfs
from generic_parser import EntryPointParameters, entrypoint

from omc3.correction.constants import DISP, NORM_DISP, F1001, F1010
from omc3.correction.model_appenders import add_coupling_to_model
from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import (BETA_NAME, AMP_BETA_NAME, PHASE_NAME,
                                                TOTAL_PHASE_NAME,
                                                DISPERSION_NAME, NORM_DISP_NAME,
                                                EXT, DELTA, ERR,
                                                PHASE_ADV, BETA, PHASE,
                                                TUNE, NAME, NAME2, S, MDL, REAL, IMAG)
from omc3.optics_measurements.toolbox import df_rel_diff, df_ratio, df_diff, df_ang_diff, ang_interval_check, ang_diff
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStrOrDataFrame, PathOrStr

LOG = logging_tools.get_logger(__name__)

OUTPUTNAMES_MAP = {
    # Names to be output on input of certain parameters.
    f'{BETA}X': tuple(f"{name}x" for name in (BETA_NAME, AMP_BETA_NAME)),
    f'{BETA}Y': tuple(f"{name}y" for name in (BETA_NAME, AMP_BETA_NAME)),
    f'{DISP}X': tuple([f"{DISPERSION_NAME}x"]),
    f'{DISP}Y': tuple([f"{DISPERSION_NAME}y"]),
    f'{PHASE}X': tuple(f"{name}x" for name in (PHASE_NAME, TOTAL_PHASE_NAME)),
    f'{PHASE}Y': tuple(f"{name}y" for name in (PHASE_NAME, TOTAL_PHASE_NAME)),
    F1010: tuple([F1010.lower()]),
    F1001: tuple([F1001.lower()]),
    f'{NORM_DISP}X': tuple(f"{name}x" for name in (BETA_NAME, AMP_BETA_NAME, DISPERSION_NAME, NORM_DISP_NAME)),
}
FAKED_HEADER = "FAKED_FROM"
VALUES = 'values'
ERRORS = 'errors'
EPSILON = 1e-14  # smallest allowed relative error, empirical value
                 # could be smaller but then normal(mean, err) == mean sometimes


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="twiss",
        required=True,
        help="Twiss dataframe or path to twiss-file.",
        type=PathOrStrOrDataFrame,
    )
    params.add_parameter(
        name="model",
        help=("Alternative Model (Dataframe or Path) to use. "
             "If not given, `twiss` will be used."),
        type=PathOrStrOrDataFrame,
    )
    params.add_parameter(
        name="parameters",
        help="Optics parameters to use",
        choices=list(OUTPUTNAMES_MAP.keys()),
        default=list(OUTPUTNAMES_MAP.keys()),
        type=str,
        nargs="+",
    )
    params.add_parameter(
        name="relative_errors",
        help=("Relative errors. Either single value for all paramters or"
              "list of values in order of parameters."),
        default=[0.,],
        type=float,
        nargs="+",
    )
    params.add_parameter(
        name="randomize",
        help=("Randomize values and/or errors from gaussian distributions."
              " If not randomized, measurement values will be equal to the model "
              "values and the errors will be equal to the relative error * measurement."
              ),
        choices=[VALUES, ERRORS],
        default=[],
        type=str,
        nargs="*",
    )
    params.add_parameter(
        name="outputdir",
        help="Path to the output directory for the fake measurement.",
        type=PathOrStr,
    )
    params.add_parameter(
        name="seed",
        help="Set random seed.",
        type=int,
    )
    return params


@entrypoint(get_params(), strict=True)
def generate(opt) -> Dict[str, tfs.TfsDataFrame]:
    """
    Takes a twiss file and writes the parameters in optics_parameters to Output_dir in the format
    global_correction_entrypoint uses (same format you would get from hole_in_one).
    """
    LOG.info("Generating fake measurements.")
    # prepare data
    np.random.seed(opt.seed)
    randomize = opt.randomize if opt.randomize is not None else []
    df_twiss, df_model = _get_data(opt.twiss, opt.model,
                                   add_coupling=(F1001 in opt.parameters) or (F1010 in opt.parameters))

    # headers
    headers = {f"{TUNE}1": df_twiss[f"{TUNE}1"], f"{TUNE}2": df_twiss[f"{TUNE}2"]}
    if isinstance(opt.twiss, PathOrStr):
        headers[FAKED_HEADER] = str(opt.twiss)

    # create defaults
    results = {}
    for parameter, error in _get_loop_parameters(opt.parameters, opt.relative_errors):
        create = CREATOR_MAP[parameter[:-1]]
        new_dfs = create(df_twiss, df_model, parameter,
                         relative_error=error,
                         randomize=randomize,
                         headers=headers)
        results.update(new_dfs)

    # maybe create normalized dispersion
    for plane in PLANES:
        nd_param = f'{NORM_DISP}{plane}'
        if nd_param in opt.parameters:
            nd_df = create_normalized_dispersion(results[f'{DISPERSION_NAME}{plane.lower()}'],
                                                 results[f'{BETA_NAME}{plane.lower()}'],
                                                 df_model,
                                                 nd_param, headers)
            results.update(nd_df)

    # output
    if opt.outputdir is not None:
        LOG.info("Writing fake measurements to files.")
        output_path = Path(opt.outputdir)
        output_path.mkdir(parents=True, exist_ok=True)
        for filename, df in results.items():
            full_path = output_path / f"{filename}{EXT}"
            tfs.write(full_path, df, save_index=NAME)

    return results


# Main Creator Functions -------------------------------------------------------

def create_beta(df_twiss: pd.DataFrame, df_model: pd.DataFrame, parameter: str,
                relative_error: float, randomize: Sequence[str], headers: Dict):
    """ Create both beta_amp and beta_phase measurements. """
    LOG.info(f"Creating fake beta for {parameter}.")
    plane = parameter[-1]
    df = create_measurement(df_twiss, parameter, relative_error, randomize)
    df[parameter] = np.abs(df[parameter])

    df = append_model(df, df_model, parameter, plane, beat=True)
    df.headers = headers.copy()
    return {f'{BETA_NAME}{plane.lower()}': df,
            f'{AMP_BETA_NAME}{plane.lower()}': df}


def create_dispersion(df_twiss: pd.DataFrame, df_model: pd.DataFrame, parameter: str,
                      relative_error: float, randomize: Sequence[str], headers: Dict):
    """ Creates dispersion measurement. """
    LOG.info(f"Creating fake dispersion for {parameter}.")
    plane = parameter[-1]
    df = create_measurement(df_twiss, parameter, relative_error, randomize)
    df = append_model(df, df_model, parameter, plane)
    df.headers = headers.copy()
    return {f'{DISPERSION_NAME}{plane.lower()}': df}


def create_phase(df_twiss: pd.DataFrame, df_model: pd.DataFrame, parameter: str,
                 relative_error: float, randomize: Sequence[str], headers: Dict):
    """ Creates both phase advance and total phase measurements. """
    LOG.info(f"Creating fake phases for {parameter}.")
    results = {}
    dict_adv = create_phase_advance(df_twiss, df_model, parameter, relative_error, randomize, headers)
    results.update(dict_adv)

    dict_tot = create_total_phase(df_twiss, df_model, parameter, relative_error, randomize, headers)
    results.update(dict_tot)
    return results


def create_phase_advance(df_twiss: pd.DataFrame, df_model: pd.DataFrame, parameter: str,
                         relative_error: float, randomize: Sequence[str], headers: Dict):
    """ Creates phase advance measurements. """
    LOG.debug(f"Creating fake phase advance for {parameter}.")
    plane = parameter[-1]
    df_adv = tfs.TfsDataFrame(index=df_twiss.index[:-1])
    df_adv[NAME2] = df_twiss.index[1:].to_numpy()

    def get_phase_advances(df_source):
        return ang_diff(
                df_source.loc[df_adv[NAME2], f"{PHASE_ADV}{plane}"].to_numpy(),
                df_source.loc[df_adv.index, f"{PHASE_ADV}{plane}"].to_numpy()
        )

    values = get_phase_advances(df_twiss)
    errors = relative_error * np.ones_like(values)
    if ERRORS in randomize:
        errors = _get_random_errors(errors, np.ones_like(values)) % 0.5

    if VALUES in randomize:
        values = np.random.normal(values, errors)
        values = ang_interval_check(values)

    df_adv[parameter] = values
    df_adv[f'{ERR}{parameter}'] = errors

    # adv model
    df_adv[S] = df_model.loc[df_adv.index, S]
    df_adv[f'{S}2'] = df_model.loc[df_adv[NAME2], S].to_numpy()
    df_adv[f'{parameter}{MDL}'] = get_phase_advances(df_model)
    df_adv[f'{PHASE_ADV}{plane}{MDL}'] = df_model.loc[df_adv.index, f'{PHASE_ADV}{plane}']

    df_adv[f"{ERR}{DELTA}{parameter}"] = df_adv[f'{ERR}{parameter}']
    df_adv[f"{DELTA}{parameter}"] = df_ang_diff(df_adv, parameter, f'{parameter}{MDL}')
    df_adv.headers = headers.copy()
    return {f'{PHASE_NAME}{plane.lower()}': df_adv}


def create_total_phase(df_twiss: pd.DataFrame, df_model: pd.DataFrame, parameter: str,
                       relative_error: float, randomize: Sequence[str], headers: Dict):
    """ Creates total phase measurements. """
    LOG.debug(f"Creating fake total phase for {parameter}.")
    plane = parameter[-1]
    df_tot = tfs.TfsDataFrame(index=df_twiss.index)
    element0 = df_twiss.index[0]
    df_tot[NAME2] = element0

    values = df_twiss[f"{PHASE_ADV}{plane}"] - df_twiss.loc[element0, f"{PHASE_ADV}{plane}"]
    errors = relative_error * np.ones_like(values)
    if ERRORS in randomize:
        errors = _get_random_errors(errors, np.ones_like(values)) % 0.5
    errors[0] = 0.

    if VALUES in randomize:
        rand_val = np.random.normal(values, errors) % 1
        values += ang_diff(rand_val, values)

    df_tot[parameter] = values % 1
    df_tot[f'{ERR}{parameter}'] = errors

    # tot model
    df_tot[S] = df_model[S]
    df_tot[f'{S}2'] = df_model.loc[df_tot.index[0], S]
    df_tot[f'{parameter}{MDL}'] = (
                                          df_model[f"{PHASE_ADV}{plane}"]
                                          - df_model.loc[element0, f"{PHASE_ADV}{plane}"]
                                  ) % 1
    df_tot[f'{PHASE_ADV}{plane}{MDL}'] = df_model[f'{PHASE_ADV}{plane}']

    df_tot[f"{ERR}{DELTA}{parameter}"] = df_tot[f'{ERR}{parameter}']
    df_tot[f"{DELTA}{parameter}"] = df_ang_diff(df_tot, parameter, f'{parameter}{MDL}')
    df_tot = df_tot.fillna(0)
    df_tot.headers = headers.copy()
    return {f'{TOTAL_PHASE_NAME}{plane.lower()}': df_tot}


def create_coupling(df_twiss: pd.DataFrame, df_model: pd.DataFrame, parameter: str,
                    relative_error: float, randomize: Sequence[str], headers: Dict):
    """ Creates coupling measurements for either the difference or sum RDT. """
    LOG.info(f"Creating fake coupling for {parameter}.")
    # Naming with R and I as long as model is involved
    re = create_measurement(df_twiss, f'{parameter}R', relative_error, randomize)
    im = create_measurement(df_twiss, f'{parameter}I', relative_error, randomize)
    df = pd.concat([re, im], axis=1)
    df = append_model(df, df_model, f'{parameter}R')
    df = append_model(df, df_model, f'{parameter}I', planes='XY')

    # Go to RDT naming scheme
    df.columns = df.columns.str.replace(f'{parameter}R', REAL).str.replace(f'{parameter}I', IMAG)
    df.headers = headers.copy()
    return {parameter.lower(): df}


CREATOR_MAP = {
    BETA: create_beta,
    DISP: create_dispersion,
    PHASE: create_phase,
    F1010[:-1]: create_coupling,  # normally the plane is removed but here is no plane
    F1001[:-1]: create_coupling,
}


# Not mapped ---

def create_normalized_dispersion(df_disp: pd.DataFrame, df_beta: pd.DataFrame,
                                 df_model: pd.DataFrame, parameter: str, headers: Dict):
    """ Creates normalized dispersion from pre-created dispersion and beta dataframes. """
    LOG.info(f"Creating fake normalized dispersion for {parameter}.")
    plane = parameter[-1]  # 'X'

    # Measurement
    df = tfs.TfsDataFrame(index=df_disp.index)
    disp = df_disp.loc[:, f"{DISP}{plane}"]
    disp_err = df_disp.loc[:, f"{ERR}{DISP}{plane}"]
    beta = df_beta.loc[:, f"{BETA}{plane}"]
    beta_err = df_beta.loc[:, f"{ERR}{BETA}{plane}"]

    inv_beta = 1/beta
    df[parameter] = disp / np.sqrt(beta)
    df[f"{ERR}{parameter}"] = np.sqrt(
        0.25 * disp**2 * inv_beta**3 * beta_err**2 + inv_beta * disp_err**2
    )

    # Model
    df_model[f'{parameter}'] = df_disp[f"{DISP}{plane}{MDL}"] / np.sqrt(df_beta[f"{BETA}{plane}{MDL}"])
    df = append_model(df, df_model, parameter, plane)

    df.headers = headers.copy()
    output_name = f'{NORM_DISP_NAME}{plane.lower()}'
    return {output_name: df}


def create_measurement(df_twiss: pd.DataFrame, parameter: str, relative_error: float,
                       randomize: Sequence[str]) -> tfs.TfsDataFrame:
    """ Create a new measurement Dataframe from df_twiss from parameter. """
    LOG.debug(f"Creating fake measurement for {parameter}.")
    values = df_twiss.loc[:, parameter]
    errors = relative_error * values.abs()
    if all(values == 0):
        LOG.warning(f"All value for {parameter} are zero. "
                    f"Fake measurement will be zero as well.")
    else:
        if ERRORS in randomize:
            errors = _get_random_errors(errors, values)

        if VALUES in randomize:
            values = np.random.normal(values, errors)

    df = tfs.TfsDataFrame({parameter: values,
                           f"{ERR}{parameter}": errors},
                          index=df_twiss.index)
    return df


def append_model(df: pd.DataFrame, df_model: pd.DataFrame, parameter: str,
                 planes: str = '', beat: bool = False) -> pd.DataFrame:
    """ Add the values to the measurement. """
    LOG.debug(f"Appending model to fake measurement for {parameter}.")
    df[S] = df_model[S]

    for plane in planes:
        df[f'{PHASE_ADV}{plane}{MDL}'] = df_model[f'{PHASE_ADV}{plane}']

    df[f"{parameter}{MDL}"] = df_model[f'{parameter}']
    if beat:
        df[f"{DELTA}{parameter}"] = df_rel_diff(df, parameter, f"{parameter}{MDL}")
        df[f"{ERR}{DELTA}{parameter}"] = df_ratio(df, f"{ERR}{parameter}", f"{parameter}{MDL}")
    else:
        df[f"{DELTA}{parameter}"] = df_diff(df, f'{parameter}', f'{parameter}{MDL}')
        df[f"{ERR}{DELTA}{parameter}"] = df[f'{ERR}{parameter}']
    return df


# Other Functions --------------------------------------------------------------

def _get_data(twiss: tfs.TfsDataFrame, model: tfs.TfsDataFrame = None,
              add_coupling: bool = False) -> Tuple[tfs.TfsDataFrame, tfs.TfsDataFrame]:
    """ Get's the input data as TfsDataFrames. """
    LOG.debug("Loading data.")
    def try_reading(df_or_path):
        try:
            return tfs.read(df_or_path, index=NAME)
        except TypeError:
            return df_or_path

    twiss = try_reading(twiss)
    if add_coupling:
        twiss = add_coupling_to_model(twiss)

    if model is None:
        model = twiss.copy()
    else:
        model = try_reading(model)
        if add_coupling:
            model = add_coupling_to_model(model)
    return twiss, model


def _get_loop_parameters(parameters: Sequence[str], errors: Sequence[float]) -> List[str]:
    """ Special care for normalized dispersion"""
    parameters = list(parameters)
    if errors is None:
        errors = [0.]
    errors = list(errors)
    if len(errors) == 1:
        errors = errors * len(parameters)

    for plane in PLANES:
        nd_param = f'{NORM_DISP}{plane}'
        if nd_param in parameters:
            idx = parameters.index(nd_param)
            nd_error = errors[idx]
            del errors[idx]
            parameters.remove(nd_param)

            add_params = [p for p in (f'{DISP}{plane}', f'{BETA}{plane}') if p not in parameters]
            parameters += add_params
            errors += [nd_error/2] * len(add_params)  # not really knowable here
    return zip(parameters, errors)


def _get_random_errors(errors: np.array, values: np.array) -> np.array:
    """ Creates normal distributed error-values that will not be lower than EPSILON. """
    LOG.debug("Calculating normal distributed random errors.")
    random_errors = np.zeros_like(errors)
    too_small = np.ones_like(errors, dtype=bool)
    n_too_small = 1
    while n_too_small:
        random_errors[too_small] = np.random.normal(errors[too_small], errors[too_small])
        too_small = random_errors < EPSILON * np.abs(values)
        n_too_small = sum(too_small)
        LOG.debug(f"{n_too_small} error values are smaller than given eps.")
    LOG.debug("Random errors created.")
    return random_errors


# Script Mode ------------------------------------------------------------------

if __name__ == '__main__':
    generate()
