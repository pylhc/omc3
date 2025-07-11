"""
Corrector Lists Check
---------------------

Quick check of the corrector lists against the LHC sequence.
It checks if all the correctors in the LHC sequence are used in the variables.

These checks are performed against the correction knobs used for global corrections.
"""
from __future__ import annotations

import re
from argparse import ArgumentParser
from pathlib import Path

from omc3.model.accelerators.lhc import Lhc
from omc3.model.constants import AFS_ACCELERATOR_MODEL_REPOSITORY

ACC_MODELS_LHC: Path = AFS_ACCELERATOR_MODEL_REPOSITORY / "lhc"
LHC_SEQ_FILE: str = "lhc.seq"


def parse_lhc_sequence(year: str) -> dict[int, set[str]]:
    """ Find all the correctors in the LHC sequence.
    They follow the pattern 'kq[^fsx][.a-z0-9]+'

    They can be grouped into correctors for beam 1 and beam 2
    depending on their ending 'b1' or 'b2'.

    Args:
        year (str): Year of the optics (or hllhc1.x version).

    Returns:
        dict[int, set[str]]: Sets of corrector names grouped by beam.
    """
    text = (ACC_MODELS_LHC / year / LHC_SEQ_FILE).read_text()
    all_correctors: set[str] = set(re.findall(r"[\s\-\+=](kq[^fsx][.a-z0-9]+)", text))
    return {
        1: {corrector for corrector in all_correctors if corrector.endswith("b1")},
        2: {corrector for corrector in all_correctors if corrector.endswith("b2")},
    }


def get_lhc_correctors(variables: list[str], beam: int, year: str) -> set[str]:
    """ Get the correctors that are used in the variables.

    Args:
        variables (list[str]): List of variable names.
        beam (int): Beam to use.
        year (str): Year of the optics (or hllhc1.x version).
    """
    lhc = Lhc(beam=beam, year=year)
    used_correctors = lhc.get_variables(classes=variables)
    return set(used_correctors)


def check_variables(variables: list[str], year: str):
    """ Main function to check the corrector lists.
    Runs both beams and prints out the remaining correctors.

    Args:
        variables (list[str]): List of variable names.
        year (str): Year of the optics (or hllhc1.x version).
    """
    lhc_correctors = parse_lhc_sequence(year)

    for beam in [1, 2]:
        used_correctors = get_lhc_correctors(
            variables=variables,
            beam=beam,
            year=year,
        )
        unused_correctors = lhc_correctors[beam] - used_correctors
        unknown_correctors = used_correctors - lhc_correctors[beam]

        print(f"Beam {beam} ({year})")
        print(f"Variables: {', '.join(variables)}")
        print(f"Unused correctors: {', '.join(sorted(unused_correctors))}")
        print(f"Unknown correctors: {', '.join(sorted(unknown_correctors))}")
        print()


def main():
    """ Main function with argument parsing. """
    parser = ArgumentParser()
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        help="Year of the optics (or hllhc1.x version)."
    )
    parser.add_argument(
        "--variables",
        type=str,
        required=True,
        nargs="+",
        help="Variables to check."
    )
    args = parser.parse_args()
    check_variables(variables=args.variables, year=args.year)


def example():
    """ Example usage, comparing INJ and TOP correctors for 2024 and 2025."""
    for mqms in ["MQM_INJ_2024", "MQM_TOP_2024"]:
        check_variables(
            variables=[mqms, "MQT", "MQTL", "MQY"],
            year="2024",
        )

    print("--------------------\n")

    for mqms in ["MQM_INJ", "MQM_TOP"]:
        check_variables(
            variables=[mqms, "MQT", "MQTL", "MQY"],
            year="2025",
        )


if __name__ == "__main__":
    main()
    # example()
