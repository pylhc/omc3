"""
Test backwards compatibility for amplitude scaling.

This test verifies that:
1. Old harpy files (without AMPLITUDE_UNIT header) are correctly scaled (multiplied by 2)
2. New harpy files (with AMPLITUDE_UNIT header) are not scaled
3. The InputFiles class handles both correctly
"""

import numpy as np
import pandas as pd
import pytest
import tfs

from omc3.harpy.constants import COL_AMP, COL_NAME, COL_NATAMP, MAINLINE_UNIT
from omc3.optics_measurements.data_models import InputFiles


@pytest.mark.basic
@pytest.mark.parametrize(
    "has_header,expected_multiplier",
    [
        (False, 2.0),
        (True, 1.0),
    ],
)
def test_repair_backwards_compatible_frame(has_header, expected_multiplier):
    """Test that old files are multiplied by 2, new files are not."""
    data = {
        COL_NAME: ["BPM1", "BPM2", "BPM3"],
        f"{COL_AMP}X": [1.0, 2.0, 3.0],
        f"{COL_NATAMP}X": [0.5, 1.0, 1.5],
    }
    if has_header:
        df = tfs.TfsDataFrame(pd.DataFrame(data), headers={MAINLINE_UNIT: "m"})
    else:
        df = pd.DataFrame(data)
    df.index = df[COL_NAME]

    result = InputFiles._repair_backwards_compatible_frame(df, "X")

    np.testing.assert_array_equal(
        result[f"{COL_AMP}X"].to_numpy(), np.array([1.0, 2.0, 3.0]) * expected_multiplier
    )
    np.testing.assert_array_equal(
        result[f"{COL_NATAMP}X"].to_numpy(), np.array([0.5, 1.0, 1.5]) * expected_multiplier
    )


@pytest.mark.basic
def test_repair_backwards_compatible_frame_without_natamp():
    """Test that files without NATAMP column are handled correctly."""
    old_df = pd.DataFrame(
        {
            COL_NAME: ["BPM1", "BPM2"],
            f"{COL_AMP}Y": [1.5, 2.5],
        }
    )
    old_df.index = old_df[COL_NAME]

    result = InputFiles._repair_backwards_compatible_frame(old_df, "Y")

    np.testing.assert_array_equal(result[f"{COL_AMP}Y"].to_numpy(), [3.0, 5.0])
    assert f"{COL_NATAMP}Y" not in result.columns


@pytest.mark.basic
@pytest.mark.parametrize("plane", ["X", "Y"])
def test_backwards_compatibility_both_planes(plane):
    """Test backwards compatibility for both X and Y planes."""
    old_df = pd.DataFrame(
        {
            COL_NAME: ["BPM1", "BPM2"],
            f"{COL_AMP}{plane}": [1.0, 2.0],
            f"{COL_NATAMP}{plane}": [0.1, 0.2],
        }
    )
    old_df.index = old_df[COL_NAME]

    result = InputFiles._repair_backwards_compatible_frame(old_df, plane)

    np.testing.assert_array_equal(result[f"{COL_AMP}{plane}"].to_numpy(), [2.0, 4.0])
    np.testing.assert_array_equal(result[f"{COL_NATAMP}{plane}"].to_numpy(), [0.2, 0.4])


@pytest.mark.basic
def test_mixed_files_scenario():
    """
    Test a realistic scenario where we have both old and new files.
    This simulates what would happen in real usage.
    """
    # Old file (e.g., from older harpy version)
    old_file = pd.DataFrame(
        {
            COL_NAME: ["BPM1", "BPM2"],
            f"{COL_AMP}X": [1.0, 2.0],  # These were divided by 2 in old harpy
        }
    )
    old_file.index = old_file[COL_NAME]

    # New file (from updated harpy with AMPLITUDE_UNIT header)
    new_file = tfs.TfsDataFrame(
        pd.DataFrame(
            {
                COL_NAME: ["BPM1", "BPM2"],
                f"{COL_AMP}X": [2.0, 4.0],  # These are NOT divided by 2 in new harpy
            }
        ),
        headers={MAINLINE_UNIT: "m"},
    )
    new_file.index = new_file[COL_NAME]

    # Process both
    old_result = InputFiles._repair_backwards_compatible_frame(old_file, "X")
    new_result = InputFiles._repair_backwards_compatible_frame(new_file, "X")

    # Both should end up with the same values (correct amplitudes)
    np.testing.assert_array_equal(old_result[f"{COL_AMP}X"].to_numpy(), [2.0, 4.0])
    np.testing.assert_array_equal(new_result[f"{COL_AMP}X"].to_numpy(), [2.0, 4.0])
