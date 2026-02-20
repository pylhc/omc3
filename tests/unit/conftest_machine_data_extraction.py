"""
Conftest for machine_data_extraction tests.

This module provides shared fixtures and utilities for testing
machine_data_extraction module functions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from omc3.machine_data_extraction.data_classes import (
    BeamProcessInfo,
    FillInfo,
    OpticsInfo,
)
from omc3.machine_data_extraction.nxcals_knobs import NXCALSResult

# Sample Data ##################################################################

@pytest.fixture
def sample_time():
    """Provide a sample datetime for testing."""
    return datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_fill_info(sample_time):
    """Provide a sample FillInfo object."""
    return FillInfo(
        no=12345,
        accelerator="lhc",
        start_time=sample_time,
    )


@pytest.fixture
def sample_beamprocess_info(sample_time):
    """Provide a sample BeamProcessInfo object."""
    return BeamProcessInfo(
        name="RAMP",
        accelerator="lhc",
        context_category="PHYSICS",
        start_time=sample_time,
        category="CYCLE",
        description="LHC ramp cycle",
    )


@pytest.fixture
def sample_optics_info(sample_time):
    """Provide a sample OpticsInfo object."""
    return OpticsInfo(
        name="OPTICSYEAR1",
        id="001",
        start_time=sample_time,
        accelerator="lhc",
    )


@pytest.fixture
def sample_nxcals_results(sample_time):
    """Provide sample NXCALS results for MQT knobs."""
    now_ts = pd.Timestamp(sample_time)
    return [
        NXCALSResult("kqtf.a12b1", 0.5, now_ts, "RPMBB.UA12.RQT12.A12B1"),
        NXCALSResult("kqtd.a12b1", 0.6, now_ts, "RPMBB.UA12.RQT12.A12B1D"),
        NXCALSResult("kqtf.a23b1", 0.7, now_ts, "RPMBB.UA23.RQT23.A23B1"),
        NXCALSResult("kqtd.a23b1", 0.8, now_ts, "RPMBB.UA23.RQT23.A23B1D"),
    ]


@pytest.fixture
def sample_mqt_results_beam1(sample_time):
    """Provide sample MQT results for all 16 beam 1 knobs."""
    now_ts = pd.Timestamp(sample_time)
    results = []
    types = ["f", "d"]
    arcs = [12, 23, 34, 45, 56, 67, 78, 81]

    for t in types:
        for arc in arcs:
            madx_name = f"kqt{t}.a{arc}b1"
            pc_name = f"RPMBB.UA{arc}.RQT{t.upper()}{arc}.A{arc}B1"
            value = 0.5 + (arc + (1 if t == "d" else 0)) * 0.01
            results.append(NXCALSResult(madx_name, value, now_ts, pc_name))

    return results


@pytest.fixture
def sample_mqt_results_beam2(sample_time):
    """Provide sample MQT results for all 16 beam 2 knobs."""
    now_ts = pd.Timestamp(sample_time)
    results = []
    types = ["f", "d"]
    arcs = [12, 23, 34, 45, 56, 67, 78, 81]

    for t in types:
        for arc in arcs:
            madx_name = f"kqt{t}.a{arc}b2"
            pc_name = f"RPMBB.UA{arc}.RQT{t.upper()}{arc}.A{arc}B2"
            value = 0.5 + (arc + (1 if t == "d" else 0)) * 0.01
            results.append(NXCALSResult(madx_name, value, now_ts, pc_name))

    return results


# Mock Factories ###############################################################

@pytest.fixture
def mock_spark():
    """Provide a mock Spark session."""
    return MagicMock()


@pytest.fixture
def mock_lsa_client():
    """Provide a mock LSA client."""
    return MagicMock()


@pytest.fixture
def mock_spark_builder(mock_spark):
    """Provide a mock Spark session builder."""
    builder = MagicMock()
    builder.get_or_create.return_value = mock_spark
    return builder


# Test Data Files ##############################################################

@pytest.fixture
def knob_extractor_inputs_dir():
    """Provide path to knob_extractor test inputs."""
    return Path(__file__).parent.parent / "inputs" / "knob_extractor"


@pytest.fixture
def extracted_mqts_b1_file(knob_extractor_inputs_dir):
    """Provide path to sample extracted MQTs for beam 1."""
    return knob_extractor_inputs_dir / "extracted_mqts_b1.str"


@pytest.fixture
def extracted_mqts_b2_file(knob_extractor_inputs_dir):
    """Provide path to sample extracted MQTs for beam 2."""
    return knob_extractor_inputs_dir / "extracted_mqts_b2.str"


# Markers ######################################################################

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "cern_network: marks tests that require CERN network access"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (e.g., network tests)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
