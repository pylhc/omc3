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

from omc3.optics_measurements.data_models import InputFiles


@pytest.mark.basic
def test_repair_backwards_compatible_frame_old_file():
    """Test that old files (without AMPLITUDE_UNIT header) are multiplied by 2."""
    # Create an old-style DataFrame (no AMPLITUDE_UNIT header)
    old_df = pd.DataFrame({
        "NAME": ["BPM1", "BPM2", "BPM3"],
        "AMPX": [1.0, 2.0, 3.0],
        "NATAMPX": [0.5, 1.0, 1.5],
    })
    old_df.index = old_df["NAME"]

    # Apply the backwards compatibility repair
    result = InputFiles._repair_backwards_compatible_frame(old_df, "X")

    # Old files should have amplitudes multiplied by 2
    np.testing.assert_array_equal(result["AMPX"].values, [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(result["NATAMPX"].values, [1.0, 2.0, 3.0])


@pytest.mark.basic
def test_repair_backwards_compatible_frame_new_file():
    """Test that new files (with AMPLITUDE_UNIT header) are NOT multiplied by 2."""
    # Create a new-style TfsDataFrame (with AMPLITUDE_UNIT header)
    new_df = tfs.TfsDataFrame(
        pd.DataFrame({
            "NAME": ["BPM1", "BPM2", "BPM3"],
            "AMPX": [2.0, 4.0, 6.0],
            "NATAMPX": [1.0, 2.0, 3.0],
        }),
        headers={"AMPLITUDE_UNIT": "m", "Q1": 0.31}
    )
    new_df.index = new_df["NAME"]

    # Apply the backwards compatibility repair
    result = InputFiles._repair_backwards_compatible_frame(new_df, "X")

    # New files should NOT have amplitudes multiplied (values stay the same)
    np.testing.assert_array_equal(result["AMPX"].values, [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(result["NATAMPX"].values, [1.0, 2.0, 3.0])


@pytest.mark.basic
def test_repair_backwards_compatible_frame_without_natamp():
    """Test that files without NATAMP column are handled correctly."""
    # Old file without NATAMP
    old_df = pd.DataFrame({
        "NAME": ["BPM1", "BPM2"],
        "AMPY": [1.5, 2.5],
    })
    old_df.index = old_df["NAME"]

    result = InputFiles._repair_backwards_compatible_frame(old_df, "Y")

    # AMPY should be multiplied by 2
    np.testing.assert_array_equal(result["AMPY"].values, [3.0, 5.0])
    # No error should occur for missing NATAMPY
    assert "NATAMPY" not in result.columns


@pytest.mark.basic
def test_new_file_with_other_headers():
    """Test that new files with AMPLITUDE_UNIT and other headers work correctly."""
    # New file with multiple headers including AMPLITUDE_UNIT
    new_df = tfs.TfsDataFrame(
        pd.DataFrame({
            "NAME": ["BPM1", "BPM2"],
            "AMPX": [3.0, 4.0],
        }),
        headers={
            "AMPLITUDE_UNIT": "m",
            "Q1": 0.31,
            "Q2": 0.32,
            "TIME": "2025-10-29 00:00:00"
        }
    )
    new_df.index = new_df["NAME"]

    result = InputFiles._repair_backwards_compatible_frame(new_df, "X")

    # Should not be multiplied
    np.testing.assert_array_equal(result["AMPX"].values, [3.0, 4.0])


@pytest.mark.basic
def test_backwards_compatibility_both_planes():
    """Test backwards compatibility for both X and Y planes."""
    # Old file
    old_df_x = pd.DataFrame({
        "NAME": ["BPM1", "BPM2"],
        "AMPX": [1.0, 2.0],
        "NATAMPX": [0.1, 0.2],
    })
    old_df_x.index = old_df_x["NAME"]

    old_df_y = pd.DataFrame({
        "NAME": ["BPM1", "BPM2"],
        "AMPY": [1.5, 2.5],
        "NATAMPY": [0.15, 0.25],
    })
    old_df_y.index = old_df_y["NAME"]

    # Apply repair for both planes
    result_x = InputFiles._repair_backwards_compatible_frame(old_df_x, "X")
    result_y = InputFiles._repair_backwards_compatible_frame(old_df_y, "Y")

    # Both should be multiplied by 2
    np.testing.assert_array_equal(result_x["AMPX"].values, [2.0, 4.0])
    np.testing.assert_array_equal(result_x["NATAMPX"].values, [0.2, 0.4])
    np.testing.assert_array_equal(result_y["AMPY"].values, [3.0, 5.0])
    np.testing.assert_array_equal(result_y["NATAMPY"].values, [0.3, 0.5])


@pytest.mark.basic
def test_mixed_files_scenario():
    """
    Test a realistic scenario where we have both old and new files.
    This simulates what would happen in real usage.
    """
    # Old file (e.g., from older harpy version)
    old_file = pd.DataFrame({
        "NAME": ["BPM1", "BPM2"],
        "AMPX": [1.0, 2.0],  # These were divided by 2 in old harpy
    })
    old_file.index = old_file["NAME"]

    # New file (from updated harpy with AMPLITUDE_UNIT header)
    new_file = tfs.TfsDataFrame(
        pd.DataFrame({
            "NAME": ["BPM1", "BPM2"],
            "AMPX": [2.0, 4.0],  # These are NOT divided by 2 in new harpy
        }),
        headers={"AMPLITUDE_UNIT": "m"}
    )
    new_file.index = new_file["NAME"]

    # Process both
    old_result = InputFiles._repair_backwards_compatible_frame(old_file, "X")
    new_result = InputFiles._repair_backwards_compatible_frame(new_file, "X")

    # Both should end up with the same values (correct amplitudes)
    np.testing.assert_array_equal(old_result["AMPX"].values, [2.0, 4.0])
    np.testing.assert_array_equal(new_result["AMPX"].values, [2.0, 4.0])
