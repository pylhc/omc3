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

    default: ``['values', 'errors']``


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
                                                TUNE, NAME, NAME2, S, MDL)
from omc3.optics_measurements.toolbox import df_rel_diff, df_ratio, df_diff, df_ang_diff, ang_interval_check
from omc3.utils.iotools import PathOrStrOrDataFrame, PathOrStr

OPTICS_PARAMETERS = tuple([f'{param}{plane}' for param in (PHASE, BETA, DISP) for plane in PLANES] + [f'{NORM_DISP}X', F1010, F1001])
FAKED_HEADER = "FAKED_FROM"
VALUES = 'values'
ERRORS = 'errors'
EPSILON = 1e-17  # smallest allowed relative error


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
        choices=OPTICS_PARAMETERS,
        default=list(OPTICS_PARAMETERS),
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
        default=[VALUES, ERRORS],
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
    # prepare data
    np.random.seed(opt.seed)
    df_twiss, df_model = _get_data(opt.twiss, opt.model,
                                   add_coupling=(F1001 in opt.parameters) or (F1010 in opt.parameters))

    # headers
    headers = {f"{TUNE}1": df_twiss[f"{TUNE}1"], f"{TUNE}2": df_twiss[f"{TUNE}2"]}
    if isinstance(opt.twiss, PathOrStr):
        headers[FAKED_HEADER] = str(opt.twiss)

    # create defaults
    results = {}
    for parameter in _get_loop_parameters(opt.parameters):
        create = CREATOR_MAP[parameter]
        new_dfs = create(df_twiss, parameter, relative_error=opt.error, randomize=opt.randomize, headers=headers)
        results.update(new_dfs)

    # maybe create normalized dispersion
    for plane in PLANES:
        nd_param = f'{NORM_DISP}{plane}'
        if nd_param in opt.parameters:
            nd_df = create_normalized_dispersion(
                nd_param,
                results[f'{DISPERSION_NAME}{plane}'],
                results[f'{BETA_NAME}{plane}'],
                headers)
            results.update(nd_df)

    # output
    if opt.outputdir is not None:
        output_path = Path(opt.outputdir)
        for filename, df in results:
            full_path = output_path / f"{filename}{EXT}"
            tfs.write(full_path, df, save_index=NAME)

    return results


# Main Creator Functions -------------------------------------------------------

def create_beta(df_twiss, df_model, parameter, relative_error, randomize, headers):
    """ Create both beta_amp and beta_phase measurements. """
    plane = parameter[-1]

    # Measurement
    df = create_measurement(df_twiss, parameter, relative_error, randomize, headers)

    # Model
    df[S] = df_model[S]
    df[f'{PHASE_ADV}{plane}{MDL}'] = df_model[f'{PHASE_ADV}{plane}']
    df[f"{parameter}{MDL}"] = df_model[parameter]
    df[f"{DELTA}{parameter}"] = df_rel_diff(df, parameter, f"{parameter}{MDL}")
    df[f"{ERR}{DELTA}{parameter}"] = df_ratio(df, f"{ERR}{parameter}", f"{parameter}{MDL}")

    df.headers = headers.copy()
    return {f'{BETA_NAME}{plane.lower()}': df,
            f'{AMP_BETA_NAME}{plane}': df}


def create_dispersion(df_twiss, df_model, parameter, relative_error, randomize, headers):
    """ Creates dispersion measurement. """
    plane = parameter[-1]

    # Measurement
    df = create_measurement(df_twiss, parameter, relative_error, randomize, headers)

    # Model
    df[S] = df_model[S]
    df[f'{PHASE_ADV}{plane}{MDL}'] = df_model[f'{PHASE_ADV}{plane}']
    df[f"{parameter}{MDL}"] = df_model[parameter]
    df[f"{DELTA}{parameter}"] = df_diff(df, parameter, f"{parameter}{MDL}")
    df[f"{ERR}{DELTA}{parameter}"] = df[f"{ERR}{parameter}"]

    df.headers = headers.copy()
    return {f'{DISPERSION_NAME}{plane.lower()}': df}


def create_phase(df_twiss, df_model, parameter, relative_error, randomize, headers):
    """ Creates both phase advance and total phase measurements. """
    plane = parameter[-1]

    # phase advance
    df_adv = tfs.TfsDataFrame(index=df_twiss.index[:-1])
    df_adv[NAME2] = df_twiss.index[1:].to_numpy()

    def get_phase_advances(df_source):
        return (
                df_source.loc[df_adv[NAME2], f"{PHASE_ADV}{plane}"].to_numpy()
                - df_source.loc[df_adv.index, f"{PHASE_ADV}{plane}"].to_numpy()
        )

    values = get_phase_advances(df_twiss)
    errors = relative_error * values.abs()
    if ERRORS in randomize:
        errors = _get_random_errors(errors, values) % 0.5

    if VALUES in randomize:
        values = np.random.normal(values, errors)
        values = ang_interval_check(values)

    df_adv[parameter] = values
    df_adv[f'{ERR}{parameter}'] = errors

    # adv model
    df_adv[S] = df_model.loc[df_adv.index, S]
    df_adv[f'{S}2'] = df_model.loc[df_adv[NAME2], S]
    df_adv[f'{parameter}{MDL}'] = get_phase_advances(df_model)
    df_adv[f'{PHASE_ADV}{plane}{MDL}'] = df_model.loc[df_adv.index, f'{PHASE_ADV}{plane}']

    df_adv[f"{ERR}{DELTA}{parameter}"] = df_adv[f'{ERR}{parameter}']
    df_adv[f"{DELTA}{parameter}"] = df_ang_diff(df_adv, parameter, f'{parameter}{MDL}')
    df_adv.headers = headers.copy()

    # total phase
    df_tot = tfs.TfsDataFrame(0, index=df_twiss.index)
    df_tot[NAME2] = df_twiss.index[0]

    values = get_phase_advances(df_twiss).cumsum()
    if VALUES in randomize:
        rand_val = np.random.normal(values, errors)
        values += ang_interval_check(rand_val - values)

    df_tot.loc[df_adv[NAME2], parameter] = values
    df_tot.loc[df_adv[NAME2], f'{ERR}{parameter}'] = errors

    # tot model
    df_tot[S] = df_model[S]
    df_adv[f'{S}2'] = df_model.loc[df_tot.index[0], S]
    df_tot.loc[df_adv[NAME2], f'{parameter}{MDL}'] = (
            get_phase_advances(df_model) - df_model.loc[df_tot.index[0], f"{PHASE_ADV}{plane}"])
    df_tot[f'{PHASE_ADV}{plane}{MDL}'] = df_model[f'{PHASE_ADV}{plane}']

    df_tot[f"{ERR}{DELTA}{parameter}"] = df_tot[f'{ERR}{parameter}']
    df_tot[f"{DELTA}{parameter}"] = df_ang_diff(df_tot, parameter, f'{parameter}{MDL}')
    df_tot.headers = headers.copy()

    return {f'{PHASE_NAME}{plane.lower()}': df_adv,
            f'{TOTAL_PHASE_NAME}{plane.lower()}': df_tot}


def create_coupling(df_twiss, df_model, parameter, relative_error, randomize, headers):
    """ Creates coupling measurements for either the difference or sum RDT. """
    re = create_measurement(df_twiss, f'{parameter}R', relative_error, randomize)
    im = create_measurement(df_twiss, f'{parameter}I', relative_error, randomize)
    df = re.append(im)

    df[S] = df_model[S]
    df[f'{PHASE_ADV}X'] = df_model[f'{PHASE_ADV}X']
    df[f'{PHASE_ADV}Y'] = df_model[f'{PHASE_ADV}Y']

    df.headers = headers.copy()
    return {parameter.lower(): df}


CREATOR_MAP = {
    f'{BETA}X': create_beta,
    f'{BETA}Y': create_beta,
    f'{DISP}X': create_dispersion,
    f'{DISP}Y': create_dispersion,
    f'{PHASE}X': create_phase,
    f'{PHASE}Y': create_phase,
    F1010: create_coupling,
    F1001: create_coupling,
}


# Not mapped ---

def create_normalized_dispersion(parameter, df_disp, df_beta, headers):
    """ Creates normalized dispersion from pre-created dispersion and beta dataframes. """
    plane = parameter[-1]  # 'X'

    # Measurement
    df = tfs.TfsDataFrame(index=df_disp.index)
    disp = df_disp.loc[:, f"{DISP}{plane}"]
    disp_err = df_disp.loc[:, f"{DISP}{plane}"]
    beta = df_beta.loc[:, f"{ERR}{BETA}{plane}"]
    beta_err = df_beta.loc[:, f"{ERR}{BETA}{plane}"]

    inv_beta = 1/beta
    df[parameter] = disp / np.sqrt(beta)
    df[f"{ERR}{parameter}"] = np.sqrt(
        0.25 * disp**2 * inv_beta**3 * beta_err**2 + inv_beta * disp_err**2
    )
    df.headers = headers.copy()

    # Model
    df[f'{parameter}{MDL}'] = df_disp[f"{DISP}{plane}{MDL}"] / np.sqrt(df_beta[f"{BETA}{plane}{MDL}"])
    df[f'{PHASE_ADV}{plane}{MDL}'] = df_beta[f'{PHASE_ADV}{plane}{MDL}']
    df[f"{DELTA}{parameter}"] = df_diff(df, parameter, f"{parameter}{MDL}")
    df[f"{ERR}{DELTA}{parameter}"] = df[f"{ERR}{parameter}"]

    output_name = f'{NORM_DISP_NAME}{plane.lower()}'
    return {output_name: df}


def create_measurement(df_twiss: tfs.TfsDataFrame, parameter: str, relative_error: float,
                       randomize: Sequence[str]) -> tfs.TfsDataFrame:
    """ Create a new measurement Dataframe from df_twiss from parameter. """
    values = df_twiss.loc[:, parameter]
    errors = relative_error * values.abs()
    if ERRORS in randomize:
        errors = _get_random_errors(errors, values)

    if VALUES in randomize:
        values = np.random.normal(values, errors)

    df = tfs.TfsDataFrame({parameter: values,
                           f"{ERR}{parameter}": errors},
                          index=df_twiss.index)
    return df


# Other Functions --------------------------------------------------------------

def _get_data(twiss: tfs.TfsDataFrame, model: tfs.TfsDataFrame = None,
              add_coupling: bool = False) -> Tuple[tfs.TfsDataFrame, tfs.TfsDataFrame]:
    """ Get's the input data as TfsDataFrames. """
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


def _get_loop_parameters(parameters: Sequence[str]) -> List[str]:
    """ Special care for normalized dispersion"""
    parameters = list(parameters)
    for plane in PLANES:
        nd_param = f'{NORM_DISP}{plane}'
        if nd_param in parameters:
            parameters += [p for p in (f'{DISP}{plane}', f'{BETA}{plane}') if p not in parameters]
            parameters.remove(nd_param)
    return parameters


def _get_random_errors(errors: np.array, values: np.array) -> np.array:
    """ Creates normal distributed error-values that will not be lower than EPSILON. """
    errors = np.random.normal(errors, errors)  # yes: (errors, errors)

    # avoid == 0 errors:
    eps_errors = EPSILON * values
    too_small = errors < eps_errors
    errors[too_small] = eps_errors[too_small]
    return errors


# Script Mode ------------------------------------------------------------------

if __name__ == '__main__':
    generate()
