import pathlib
import re
from functools import reduce
from operator import mul
from typing import Dict, List
from uncertainties import unumpy as up

import tfs
from generic_parser import EntryPointParameters, entrypoint
from tfs.tools import significant_digits

from omc3.definitions.constants import PLANES
from omc3.kmod.constants import  ERR, BETA
from omc3.run_kmod import EXT
from omc3.run_kmod import LSA_FILE_NAME as LSA_RESULTS
from omc3.utils.logging_tools import get_logger

LOG = get_logger(__name__)

LABEL = "NAME"  # Column containing the IP/Beam names
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

    # Validate the columns we need: BET{X,Y} and ERRBET{X,Y}
    expected_columns = [f"{BETA}{p}" for p in PLANES] + [f"{ERR}{BETA}{p}" for p in PLANES]
    if not all([column in data_frame.columns for column in expected_columns]):
        return False

    return True


def get_lumi_imbalance(data_frame: tfs.TfsDataFrame) -> Dict[str, float]:
    """
    Calculate the IP1 / IP5 luminosity imbalance. The luminosity is taken as defined in
    `Concept of luminosity`, Eq(17): https://cds.cern.ch/record/941318/files/p361.pdf

    The calculation of the luminosity imbalance is then:
    .. math::
    \\frac{L_{IP1}}{L_{IP5}}=\\frac{\\sqrt{\\beta_{x1,IP5}+\\beta_{x2,IP5}}\\cdot\\sqrt{\\beta_{y1,IP5}+\\beta_{y2,IP5}}}{\\sqrt{\\beta_{x1,IP1}+\\beta_{x2,IP1}}\\cdot\\sqrt{\\beta_{y1,IP1}+\\beta_{y2,IP1}}}

    Args:
        data_frame (tfs.TfsDataFrame): a `TfsDataFrame` with the results from a kmod analysis.

    Returns:
        A dictionary with the imbalance, the betas at IPs and their errors.
    """

    lumi_coefficient: Dict[str, float] = {}

    for ip in IPS:
        LOG.debug(
            f"Computing lumi contribution from optics for IP {ip}"
        )
        ip_row = data_frame.loc[data_frame[LABEL].str.startswith(ip)]

        lumi_coefficient[ip] = 0.5*reduce(mul,
                                          [up.sqrt(
                                            up.uarray(ip_row[f"{BETA}{plane}"].values,
                                                      ip_row[f"{ERR}{BETA}{plane}"].values).sum()
                                                   )
                                            for plane in PLANES]
                                           ) # at some point when omc3 is py>=3.8 this can be replaced by prod()

    imbalance = lumi_coefficient[IPS[0]] / lumi_coefficient[IPS[1]]

    return {
        "imbalance": imbalance.nominal_value,
        "relative_error": imbalance.std_dev,
        "eff_beta_ip1": lumi_coefficient[IPS[0]].nominal_value,
        "rel_error_ip1": lumi_coefficient[IPS[0]].std_dev,
        "eff_beta_ip5": lumi_coefficient[IPS[1]].nominal_value,
        "rel_error_ip5": lumi_coefficient[IPS[1]].std_dev,
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
        help=f"Path to kmod directories with stored KMOD measurement files,"
             f"in particular {LSA_RESULTS}{EXT}",
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

    # Combine the lsa data
    lsa_tfs = merge_tfs(ip_dir_names, f"{LSA_RESULTS}{EXT}")

    # If the TFS data contains everything we need: get the imbalance
    if _validate_for_imbalance(lsa_tfs):
        res = get_lumi_imbalance(lsa_tfs)
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
