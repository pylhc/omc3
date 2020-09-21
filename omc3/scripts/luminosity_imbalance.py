from math import sqrt
from functools import reduce
from operator import mul

import tfs
from tfs.tools import significant_digits

from omc3.utils.logging_tools import get_logger
from omc3.definitions.constants import PLANES
from generic_parser import entrypoint, EntryPointParameters

LOG = get_logger(__name__)

BETASTAR = 'BETSTAR'
ERR = 'ERR'
LABEL = 'LABEL'  # Column containing the IP/Beam names
BEAMS = ('B1', 'B2')
IPS = ('ip1', 'ip5')
LABELS = [f'{i}{b}' for i in IPS for b in BEAMS]


def get_params():
    return EntryPointParameters(
            tfs=dict(
                required=True,
                type=str,
                help="TFS file containing the β* and error for "
                     "B{1,2}{H,V}IP{1,5}"
            ),
            inplace=dict(
                required=False,
                action='store_true',
                help="Add the luminosity imbalance in the source TFS file as a"
                     "header"
            )
    )


def _validate_tfs(df):
    # First validate the labels, which correspond to the IP and Beam
    expected_labels = set(LABELS)
    not_found_labels = [l for l in expected_labels \
                        if not df[LABEL].str.contains(l).any()]

    if any(not_found_labels):
        msg = 'The following required labels are not found in dataframe:' \
              f' {", ".join(not_found_labels)}'
        LOG.error(msg)
        raise KeyError(msg)

    # Check if we got several times the same labels
    for label in LABELS:
        if len(df.loc[df[LABEL] == label]) > 1:
            msg = f'Found label {label} several times. Expected only once.'
            LOG.error(msg)
            raise KeyError(msg)

    # Validate the columns we need: BETSTAR{X,Y} and ERRBETSTAR{X,Y}
    expected_columns = [f'{BETASTAR}{p}' for p in PLANES] + \
                       [f'{ERR}{BETASTAR}{p}' for p in PLANES]
    if not all([column in df.columns for column in expected_columns]):
        msg = 'Expected columns in the TFS file not found. Expected ' \
              f'columns: {expected_columns}'
        LOG.error(msg)
        raise KeyError(msg)


def get_imbalance(df):
    rel_error = {}
    eff_beta = {}
    for ip in IPS:
        # Get the individual rows for the IP1 and IP5
        row = df.loc[df[LABEL].str.startswith(ip)]

        # Compute the average β* for each plane with both beams
        # and the absolute errors for each plane
        average_beta = {}
        error = {}
        for p in PLANES:
            average_beta[p] = row.sum(axis=0)[f'{BETASTAR}{p}'] / 2
            error[p] = 0.5 * sqrt((row[f'{ERR}{BETASTAR}{p}'] ** 2).sum(axis=0))

        # Compute the relative error for this IP
        rel_error[ip] = 0.5 * \
          sqrt(sum([(error[p] ** 2) / (average_beta[p] ** 2) for p in PLANES]))

        # Get the effective beta
        eff_beta[ip] = 1 / sqrt(1 / (reduce(mul, average_beta.values())))

    # Compute the whole relative error
    relative_error = sum(rel_error.values())

    # And get the imbalance
    imbalance = eff_beta[IPS[0]] / eff_beta[IPS[1]]

    return {'imbalance': imbalance,
            'relative_error': relative_error,
            'eff_beta_ip1': eff_beta[IPS[0]],
            'rel_error_ip1': rel_error[IPS[0]],
            'eff_beta_ip5': eff_beta[IPS[1]],
            'rel_error_ip5': rel_error[IPS[1]]
            }


def print_luminosity(res):
    LOG.info(f'{"Luminosity Imbalance":22s}: {res["imbalance"]}')
    LOG.info(f'{"Relative error":22s}: {res["relative_error"]}')
    LOG.info(f'{"Effective beta IP1":22s}: {res["eff_beta_ip1"]}')
    LOG.info(f'{"Rel error IP1":22s}: {res["rel_error_ip1"]}')
    LOG.info(f'{"Effective beta IP5":22s}: {res["eff_beta_ip5"]}')
    LOG.info(f'{"Rel error IP5":22s}: {res["rel_error_ip5"]}')


@entrypoint(get_params(), strict=True)
def main(opt):
    df = tfs.read_tfs(opt['tfs'])
    _validate_tfs(df)

    res = get_imbalance(df)

    res['imbalance'], res['relative_error'] = significant_digits(
            res['imbalance'], res['relative_error'])
    res['eff_beta_ip1'], res['rel_error_ip1'] = significant_digits(
            res['eff_beta_ip1'], res['rel_error_ip1'])
    res['eff_beta_ip5'], res['rel_error_ip5'] = significant_digits(
            res['eff_beta_ip5'], res['rel_error_ip5'])

    print_luminosity(res)

    # Write to the source TFS file if asked
    if opt['inplace']:
        df.headers.update({"LUMINOSITY_IMBALANCE": res['imbalance'],
                           "RELATIVE_ERROR": res['relative_error'],
                           "EFF_BETA_IP1": res['eff_beta_ip1'],
                           "REL_ERROR_IP1": res['rel_error_ip1'],
                           "EFF_BETA_IP5": res['eff_beta_ip5'],
                           "REL_ERROR_IP5": res['rel_error_ip5']
                           })

        tfs.write_tfs(opt['tfs'], df)

    return res


if __name__ == "__main__":
    main()
