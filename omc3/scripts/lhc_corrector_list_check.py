""" 
Corrector Lists Check
---------------------

Quick check of the corrector lists against the LHC sequence.
It checks if all the correctors in the LHC sequence are used in the variables.

These checks are performed against the correction knobs used for global corrections.
"""
from pathlib import Path
import re
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
    correctors = {
        1: set([c for c in all_correctors if c.endswith("b1") ]),
        2: set([c for c in all_correctors if c.endswith("b2") ]),
    }
    return correctors


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


def main(variables: list[str], year: str):
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
        remaining_correctors = lhc_correctors[beam] - used_correctors

        print(f"Beam {beam} ({year})")
        print(f"Variables: {', '.join(variables)}")
        print(f"Unused correctors: {', '.join(sorted(remaining_correctors))}")
        print()



if __name__ == "__main__":
    for mqms in ["MQM_INJ_2024", "MQM_TOP_2024"]:
        main(
            variables=[mqms, "MQT", "MQTL", "MQY"],
            year="2024",
        )
    
    print("--------------------\n")

    for mqms in ["MQM_INJ", "MQM_TOP"]:
        main(
            variables=[mqms, "MQT", "MQTL", "MQY"],
            year="2025",
        )
