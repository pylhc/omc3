"""
Copy reference MQT extraction files for testing.

This script copies the extracted_mqts_b1.str and extracted_mqts_b2.str files
from CERN data paths to the test inputs directory.
"""

import shutil
from pathlib import Path

# Source paths for MQT reference files
B1_SOURCE = Path(
    "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-24/LHCB1/Models/B1_60cm_checks/extracted_mqts.str"
)
B2_SOURCE = Path(
    "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-21/LHCB2/Models/b2_120cm_correct_knobs/extracted_mqts.str"
)

# Output directory for test inputs
OUTPUT_DIR = Path(__file__).parent


def main():
    """Copy MQT reference files for both beams."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Copying MQT reference for Beam 1...")
    shutil.copy(B1_SOURCE, OUTPUT_DIR / "extracted_mqts_b1.str")

    print("Copying MQT reference for Beam 2...")
    shutil.copy(B2_SOURCE, OUTPUT_DIR / "extracted_mqts_b2.str")

    print(f"Reference files copied to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
