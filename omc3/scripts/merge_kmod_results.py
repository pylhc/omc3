import pathlib
import re
from functools import reduce
from math import sqrt
from operator import mul
from typing import Dict, List

import tfs
from generic_parser import EntryPointParameters, entrypoint
from tfs.tools import significant_digits

from omc3.definitions.constants import PLANES
from omc3.run_kmod import EXT
from omc3.run_kmod import LSA_FILE_NAME as LSA_RESULTS
from omc3.run_kmod import RESULTS_FILE_NAME as RESULTS
from omc3.utils.logging_tools import get_logger

LOG = get_logger(__name__)

BETASTAR = "BETSTAR"
ERR = "ERR"
LABEL = "LABEL"  # Column containing the IP/Beam names
BEAMS = ("B1", "B2")
IPS = ("ip1", "ip5")
LABELS = [f"{ip}{beam}" for ip in IPS for beam in BEAMS]


def _validate_for_imbalance(data_frame: tfs.TfsDataFrame) -> bool:
    """
    Checks that the provided `TfsDataFrame` contains the expected labels and columns for a
    luminosity imbalance calculation.

    Args:
        data_frame (tfs.TfsDataFrame): a loaded `TfsDataFrame` to validate.

    Returns:
        ``True`` if the provided dataframe is valid, ``False`` otherwise.
    """
    expected_labels = set(LABELS)
    not_found_labels = [
        label for label in expected_labels if not data_frame[LABEL].str.contains(label).any()
    ]
    if any(not_found_labels):
        return False

    for label in LABELS:
        if len(data_frame.loc[data_frame[LABEL] == label]) > 1:
            LOG.error(f"Found label '{label}' several times. Expected only once")
            raise KeyError(f"Duplicate label '{label}' in dataframe's columns")

    # Validate the columns we need: BETSTAR{X,Y} and ERRBETSTAR{X,Y}
    expected_columns = [f"{BETASTAR}{p}" for p in PLANES] + [f"{ERR}{BETASTAR}{p}" for p in PLANES]
    if not all([column in data_frame.columns for column in expected_columns]):
        return False

    return True


def get_lumi_imbalance(data_frame: tfs.TfsDataFrame) -> Dict[str, float]:
    """
    Calculate the IP1 / IP5 luminosity imbalance. The calculation implemented is detailed in
    `Concept of luminosity`, Eq(17): https://cds.cern.ch/record/941318/files/p361.pdf

    Args:
        data_frame (tfs.TfsDataFrame): a `TfsDataFrame` with the results from a kmod analysis.

    Returns:
        A dictionary with the imbalance, the betas at IPs and their errors.
    """
    relative_errors: Dict[str, float] = {}
    effective_betas: Dict[str, float] = {}

    for ip in IPS:
        LOG.debug(
            f"Computing average betastars and absolute errors for IP {ip}, for both planes "
            f"and beams"
        )
        ip_row = data_frame.loc[data_frame[LABEL].str.startswith(ip)]
        average_beta = {}
        error = {}

        for plane in PLANES:
            average_beta[plane] = 0.5 * ip_row[f"{BETASTAR}{plane}"].sum(axis=0)
            error[plane] = 0.5 * sqrt((ip_row[f"{ERR}{BETASTAR}{plane}"] ** 2).sum(axis=0))

        # Compute the relative error and effective beta for this IP
        relative_errors[ip] = 0.5 * sqrt(
            sum([(error[p] ** 2) / (average_beta[p] ** 2) for p in PLANES])
        )
        effective_betas[ip] = sqrt(reduce(mul, average_beta.values()))

    # Compute the whole relative error and get the imbalance
    relative_error = sum(relative_errors.values())
    imbalance = effective_betas[IPS[0]] / effective_betas[IPS[1]]

    return {
        "imbalance": imbalance,
        "relative_error": relative_error,
        "eff_beta_ip1": effective_betas[IPS[0]],
        "rel_error_ip1": relative_errors[IPS[0]],
        "eff_beta_ip5": effective_betas[IPS[1]],
        "rel_error_ip5": relative_errors[IPS[1]],
    }


def print_luminosity(lumi_imbalance_results: Dict[str, float]) -> None:
    """
    Display the analysis results.

    Args:
        lumi_imbalance_results (Dict[str, float]): a dictionary with the luminosity imbalance
            results, as returned by ``get_lumi_imbalance``.
    """
    LOG.info(f'{"Luminosity Imbalance":22s}: {lumi_imbalance_results["imbalance"]}')
    LOG.info(f'{"Relative error":22s}: {lumi_imbalance_results["relative_error"]}')
    LOG.info(f'{"Effective beta IP1":22s}: {lumi_imbalance_results["eff_beta_ip1"]}')
    LOG.info(f'{"Rel error IP1":22s}: {lumi_imbalance_results["rel_error_ip1"]}')
    LOG.info(f'{"Effective beta IP5":22s}: {lumi_imbalance_results["eff_beta_ip5"]}')
    LOG.info(f'{"Rel error IP5":22s}: {lumi_imbalance_results["rel_error_ip5"]}')


def get_significant_digits(lumi_imbalance_res: Dict[str, float]) -> Dict[str, str]:
    """
    Update luminosity imbalance results for significant digits based on their errors.

    Args:
        lumi_imbalance_res (Dict[str, float]): a dictionary with the luminosity imbalance
            results, as returned by ``get_lumi_imbalance``.

    Returns:
        The updated dictionary.
    """
    lumi_imbalance_res["imbalance"], lumi_imbalance_res["relative_error"] = significant_digits(
        lumi_imbalance_res["imbalance"], lumi_imbalance_res["relative_error"]
    )
    lumi_imbalance_res["eff_beta_ip1"], lumi_imbalance_res["rel_error_ip1"] = significant_digits(
        lumi_imbalance_res["eff_beta_ip1"], lumi_imbalance_res["rel_error_ip1"]
    )
    lumi_imbalance_res["eff_beta_ip5"], lumi_imbalance_res["rel_error_ip5"] = significant_digits(
        lumi_imbalance_res["eff_beta_ip5"], lumi_imbalance_res["rel_error_ip5"]
    )
    return lumi_imbalance_res


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


def loader_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="kmod_dirs",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help="Path to kmod directories with stored KMOD measurement files",
    )
    params.add_parameter(
        name="outputdir",
        type=pathlib.Path,
        required=True,
        help="Output directory where to write the result tfs",
    )
    return params


@entrypoint(loader_params(), strict=True)
def merge_and_copy_kmod_output(opt):
    output_path: pathlib.Path = opt.outputdir / f"{LSA_RESULTS}{EXT}"
    # Get the directories we need where the tfs are stored
    ip_dir_names: List[pathlib.Path] = get_ip_dir_names(opt.kmod_dirs)

    # Combine the data into one tfs
    new_data = merge_tfs(ip_dir_names, f"{RESULTS}{EXT}")

    # Combine the lsa data
    lsa_tfs = merge_tfs(ip_dir_names, f"{LSA_RESULTS}{EXT}")

    # If the TFS data contains everything we need: get the imbalance
    if _validate_for_imbalance(new_data):
        res = get_lumi_imbalance(new_data)
        res = get_significant_digits(res)
        print_luminosity(res)

        lsa_tfs.headers.update(
            {
                "LUMINOSITY_IMBALANCE": res["imbalance"],
                "RELATIVE_ERROR": res["relative_error"],
                "EFF_BETA_IP1": res["eff_beta_ip1"],
                "REL_ERROR_IP1": res["rel_error_ip1"],
                "EFF_BETA_IP5": res["eff_beta_ip5"],
                "REL_ERROR_IP5": res["rel_error_ip5"],
            }
        )

    LOG.info(f"Writing result TFS file to disk at: {output_path}")
    tfs.write(output_path, lsa_tfs)


if __name__ == "__main__":
    merge_and_copy_kmod_output()
