"""
Merge KMOD Results
-------------------

Script to merge the results from KMOD into one TfsDataFrame.
The script takes the kmod-results folders as input and merges the
lsa-result tfs-files in these together.

Some sanity checks are performed, e.g. that there is only one entry per element.
If IP1 and IP5 results are given for both planes and beams, the luminosity
imbalance between these IPs is also calculated and written into the header,
as well as logged.

The resulting TfsDataFrame is returned, but also written out if an `outputdir`
is given.

**Arguments:**

*--Required--*

- **kmod_dirs** *(Path)*:

    Path to kmod directories with stored KMOD measurement files,in particular lsa_results.tfs


*--Optional--*

- **outputdir** *(Path)*:

    Output directory where to write the result tfs

"""
import pathlib
import re
from typing import Dict, List, Tuple

import numpy as np
import tfs
from generic_parser import EntryPointParameters, entrypoint
from uncertainties import ufloat, UFloat, unumpy as up

from omc3.definitions.constants import PLANES
from omc3.kmod.constants import ERR, BETA, EXT
from omc3.kmod.constants import LSA_FILE_NAME as LSA_RESULTS
from omc3.utils.logging_tools import get_logger

LOG = get_logger(__name__)

NAME = "NAME"  # Column containing the IP/Beam names
BEAMS = ("B1", "B2")
IPS = ("ip1", "ip5")
IMBALANCE_NAMES = [f"{ip}{beam}" for ip in IPS for beam in BEAMS]

HEADER_IMBALANCE = "LUMINOSITY_IMBALANCE"
HEADER_REL_ERROR = "RELATIVE_ERROR"
HEADER_EFF_BETA_IP1 = "EFFECTIVE_BETA_IP1"
HEADER_REL_ERROR_IP1 = "RELATIVE_ERROR_IP1"
HEADER_EFF_BETA_IP5 = "EFFECTIVE_BETA_IP5"
HEADER_REL_ERROR_IP5 = "RELATIVE_ERROR_IP5"


def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="kmod_dirs",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help=f"Path to kmod directories with stored KMOD measurement files,"
             f"in particular {LSA_RESULTS}{EXT}",
    )
    params.add_parameter(
        name="outputdir",
        type=pathlib.Path,
        help="Output directory where to write the result tfs",
    )
    return params


@entrypoint(get_params(), strict=True)
def merge_kmod_results(opt) -> tfs.TfsDataFrame:
    """ Main function to merge K-Mod output.
    See :mod:`omc3.scripts.merge_kmod_results`.
    """
    # Get the directories we need where the tfs are stored
    ip_dir_names: List[pathlib.Path] = get_ip_dir_names(opt.kmod_dirs)

    # Combine the lsa data
    lsa_tfs = merge_tfs(ip_dir_names, f"{LSA_RESULTS}{EXT}")

    # If the TFS data contains everything we need: get the imbalance
    if _validate_for_imbalance(lsa_tfs):
        imbalance_results = get_lumi_imbalance(lsa_tfs)
        lsa_tfs = _add_imbalance_to_header(lsa_tfs, *imbalance_results)

    if opt.outputdir:
        output_path = opt.outputdir / f"{LSA_RESULTS}{EXT}"
        LOG.info(f"Writing result TFS file to disk at: {str(output_path)}")
        tfs.write(output_path, lsa_tfs)
    return lsa_tfs


def _check_tfs_sanity(data_frame: tfs.TfsDataFrame):
    """
    Checks that the merged `TfsDataFrame` has valid entries,
    i.e. names are unique and max two entries per IP.

    Args:
        data_frame (tfs.TfsDataFrame): a loaded `TfsDataFrame` to validate.
    """
    # Check that both beams are there only once
    multiple_names = [
        name for name in data_frame[NAME] if len(data_frame.loc[data_frame[NAME] == name]) > 1
    ]
    if multiple_names:
        msg = (f"Found entries '{', '.join(set(multiple_names))}' "
               f"several times in merged DataFrame. "
               "Expected only once")
        LOG.error(msg)
        raise KeyError(msg)

    # check that there is no weird additional data
    too_many_entries = [ip for ip in IPS if len(data_frame[NAME].str.startswith(ip)) > 2]
    if too_many_entries:
        msg = ("More than two entries found for ips "
               f"{', '.join(too_many_entries)} in merged DataFrame. "
               "Expected one for each beam.")
        LOG.error(msg)
        raise KeyError(msg)


def _validate_for_imbalance(data_frame: tfs.TfsDataFrame) -> bool:
    """
    Checks that the provided `TfsDataFrame` contains the expected labels and columns for a
    luminosity imbalance calculation.

    Args:
        data_frame (tfs.TfsDataFrame): a loaded `TfsDataFrame` to validate.

    Returns:
        ``True`` if the provided dataframe is valid, ``False`` otherwise.
    """
    # check all required names are there
    not_found_names = [
        name for name in IMBALANCE_NAMES if not data_frame[NAME].str.startswith(name).any()
    ]
    if not_found_names:
        return False

    # Validate the columns we need: BET{X,Y} and ERRBET{X,Y}
    expected_columns = [f"{BETA}{p}" for p in PLANES] + [f"{ERR}{BETA}{p}" for p in PLANES]
    return all([column in data_frame.columns for column in expected_columns])


def get_lumi_imbalance(data_frame: tfs.TfsDataFrame) -> Tuple[UFloat, UFloat, UFloat]:
    """
    Calculate the IP1 / IP5 luminosity imbalance. The luminosity is taken as defined in
    `Concept of luminosity`, Eq(17): https://cds.cern.ch/record/941318/files/p361.pdf

    The calculation of the luminosity imbalance is then:

    .. math::

        \\frac{L_{IP1}}{L_{IP5}}=\\frac{\\sqrt{\\beta_{x1,IP5}+\\beta_{x2,IP5}}\\cdot\\sqrt{\\beta_{y1,IP5}+\\beta_{y2,IP5}}}{\\sqrt{\\beta_{x1,IP1}+\\beta_{x2,IP1}}\\cdot\\sqrt{\\beta_{y1,IP1}+\\beta_{y2,IP1}}}


    Args:
        data_frame (tfs.TfsDataFrame): a `TfsDataFrame` with the results from a kmod analysis.

    Returns:
         Tuple with the imbalance, the betas at IPs as ufloats.
    """

    lumi_coefficient: Dict[str, UFloat] = {}

    for ip in IPS:
        LOG.debug(
            f"Computing lumi contribution from optics for IP {ip}"
        )
        ip_rows = data_frame.loc[data_frame[NAME].str.startswith(ip)]

        beta_sums = [up.uarray(ip_rows[f"{BETA}{plane}"].to_numpy(),
                               ip_rows[f"{ERR}{BETA}{plane}"].to_numpy()
                               ).sum()  # sum over beams
                     for plane in PLANES]
        lumi_coefficient[ip] = 0.5 * up.sqrt(np.prod(beta_sums))

    imbalance = lumi_coefficient[IPS[0]] / lumi_coefficient[IPS[1]]
    LOG.info(f'{"Luminosity Imbalance":22s}: {imbalance:s}')
    LOG.info(f'{"Effective beta IP1":22s}: {lumi_coefficient[IPS[0]]:s}')
    LOG.info(f'{"Effective beta IP5":22s}: {lumi_coefficient[IPS[1]]:s}')

    return imbalance, lumi_coefficient[IPS[0]], lumi_coefficient[IPS[1]]


def _add_imbalance_to_header(tfs_df: tfs.TfsDataFrame,
                             imbalance: UFloat, beta_ip1: UFloat, beta_ip5: UFloat) \
        -> tfs.TfsDataFrame:
    """
    Function to add the calculated imablance and effective betas to the header

    Args:
        tfs_df (tfs.TfsDataFrame): a `TfsDataFrame` with the results from a kmod analysis.
        imbalance (UFloat): uncertain imbalance
        beta_ip1 (UFloat): uncertain effective beta in ip1
        beta_ip5 (UFloat): uncertain effective beta in ip5

    Returns:
        tfs.TfsDataFrame with added headers.

    """
    header_map = [(HEADER_IMBALANCE, HEADER_REL_ERROR, imbalance),
                  (HEADER_EFF_BETA_IP1, HEADER_REL_ERROR_IP1, beta_ip1),
                  (HEADER_EFF_BETA_IP5, HEADER_REL_ERROR_IP5, beta_ip5),
    ]
    for val_key, err_key, val in header_map:
        tfs_df.headers[val_key], tfs_df.headers[err_key] = str(val).split('+/-')

    return tfs_df


def merge_tfs(directories: List[pathlib.Path], filename: str) -> tfs.TfsDataFrame:
    """
    Merge different kmod analysis results from a list of directories into a single `TfsDataFrame`.

    Args:
        directories (List[pathlib.Path]): list of PosixPath objects to directories holding TFS
            files with the results of kmod analysis.
        filename (str): name of the TFS files to look for in the provided directories

    Returns:
        A `TfsDataFrame` combining all the loaded files from the provided directories.
    """
    # Combine the data into one tfs
    new_tfs_df = tfs.TfsDataFrame()
    for d in sorted(directories):
        loaded_tfs = tfs.read_tfs(d / filename)

        # Save old headers before merging so we don't lose them and then add all of them
        old_headers = new_tfs_df.headers
        new_tfs_df = new_tfs_df.append(loaded_tfs, ignore_index=True)

        new_tfs_df.headers.update(old_headers)
        new_tfs_df.headers.update(loaded_tfs.headers)

    _check_tfs_sanity(new_tfs_df)
    return new_tfs_df


def get_ip_dir_names(kmod_dirs: List[pathlib.Path]) -> List[pathlib.Path]:
    # Check directories first
    for d in kmod_dirs:
        if not d.is_dir():
            msg = f"Directory {d} does not exist"
            LOG.error(msg)
            raise NotADirectoryError(msg)

    pattern = re.compile(".*ip[0-9]B[1-2]")
    ip_dir_names = [
        d for kmod in kmod_dirs for d in kmod.glob("**/*") if pattern.match(d.name) and d.is_dir()
    ]

    return ip_dir_names


if __name__ == "__main__":
    merge_kmod_results()
