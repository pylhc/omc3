"""
Unit tests for LSA and NXCALS utility functions.

These tests focus on the core logic of LSA/NXCALS functions,
using mocks to avoid network dependencies.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from omc3.machine_data_extraction.data_classes import TrimHistories


class TestLSAUtilsCalcK:
    """Tests for LSA K-value calculation functions."""

    @patch('omc3.machine_data_extraction.lsa_utils.jpype')
    def test_calc_k_from_iref_basic(self, mock_jpype):
        """Test basic K-value calculation from current reference."""
        from omc3.machine_data_extraction.lsa_utils import calc_k_from_iref

        # Setup mocks for Java interop
        mock_hashmap_class = MagicMock()
        mock_double_class = MagicMock()
        mock_jpype.JClass.side_effect = lambda x: {
            'java.util.HashMap': mock_hashmap_class,
            'java.lang.Double': mock_double_class,
        }.get(x, MagicMock())

        # Create mock LSA client with service
        mock_lsa_client = MagicMock()
        mock_service = MagicMock()
        mock_lsa_client._lhcService = mock_service

        # Create mock result
        mock_result_entry = MagicMock()
        mock_result_entry.getKey.return_value = "RPMBB.UA12.RQT12.A12B1"
        mock_result_entry.getValue.return_value = 0.5

        mock_iterator = MagicMock()
        mock_iterator.hasNext.side_effect = [True, False]
        mock_iterator.next.return_value = mock_result_entry

        mock_result_map = MagicMock()
        mock_result_map.entrySet.return_value.iterator.return_value = mock_iterator

        mock_service.calculateKfromIREF.return_value = mock_result_map

        # Create mock HashMap instance
        mock_map_instance = MagicMock()
        mock_hashmap_class.return_value = mock_map_instance

        # Create mock Double
        mock_double_instance = MagicMock()
        mock_double_class.valueOf.return_value = mock_double_instance

        # Call function
        currents = {"RPMBB.UA12.RQT12.A12B1": 100.0}
        energy = 7000.0

        result = calc_k_from_iref(mock_lsa_client, currents, energy)

        # Verify
        assert "RPMBB.UA12.RQT12.A12B1" in result
        assert result["RPMBB.UA12.RQT12.A12B1"] == 0.5
        mock_service.calculateKfromIREF.assert_called_once()

    @patch('omc3.machine_data_extraction.lsa_utils.jpype')
    def test_calc_k_from_iref_multiple_currents(self, mock_jpype):
        """Test K-value calculation with multiple currents."""
        from omc3.machine_data_extraction.lsa_utils import calc_k_from_iref

        mock_hashmap_class = MagicMock()
        mock_double_class = MagicMock()
        mock_jpype.JClass.side_effect = lambda x: {
            'java.util.HashMap': mock_hashmap_class,
            'java.lang.Double': mock_double_class,
        }.get(x, MagicMock())

        mock_lsa_client = MagicMock()
        mock_service = MagicMock()
        mock_lsa_client._lhcService = mock_service

        # Create multiple result entries
        mock_entry1 = MagicMock()
        mock_entry1.getKey.return_value = "PC1"
        mock_entry1.getValue.return_value = 0.5

        mock_entry2 = MagicMock()
        mock_entry2.getKey.return_value = "PC2"
        mock_entry2.getValue.return_value = 0.6

        mock_iterator = MagicMock()
        mock_iterator.hasNext.side_effect = [True, True, False]
        mock_iterator.next.side_effect = [mock_entry1, mock_entry2]

        mock_result_map = MagicMock()
        mock_result_map.entrySet.return_value.iterator.return_value = mock_iterator

        mock_service.calculateKfromIREF.return_value = mock_result_map

        mock_map_instance = MagicMock()
        mock_hashmap_class.return_value = mock_map_instance

        mock_double_instance = MagicMock()
        mock_double_class.valueOf.return_value = mock_double_instance

        currents = {"PC1": 100.0, "PC2": 110.0}
        energy = 7000.0

        result = calc_k_from_iref(mock_lsa_client, currents, energy)

        assert len(result) == 2
        assert result["PC1"] == 0.5
        assert result["PC2"] == 0.6

    @patch('omc3.machine_data_extraction.lsa_utils.jpype')
    def test_calc_k_from_iref_raises_on_none_current(self, mock_jpype):
        """Test that None current values raise ValueError."""
        from omc3.machine_data_extraction.lsa_utils import calc_k_from_iref

        mock_lsa_client = MagicMock()

        currents = {"PC1": None}  # Invalid: None current
        energy = 7000.0

        with pytest.raises(ValueError, match="Current.*is None"):
            calc_k_from_iref(mock_lsa_client, currents, energy)  # ty:ignore[invalid-argument-type]


@pytest.mark.usefixtures("mock_pjlsa")
class TestLSAKnobsFunctions:
    """Tests for LSA knob extraction functions."""

    @patch('omc3.machine_data_extraction.lsa_knobs.pjlsa')
    def test_find_knob_names_basic(self, mock_pjlsa):
        """Test finding knob names from LSA."""
        from omc3.machine_data_extraction.lsa_knobs import find_knob_names

        # Mock LSA client
        mock_lsa_client = MagicMock()
        mock_accelerator = MagicMock()
        mock_lsa_client._getAccelerator.return_value = mock_accelerator

        # Create mock parameters
        mock_param1 = MagicMock()
        mock_param1.getName.return_value = "KNOB1"

        mock_param2 = MagicMock()
        mock_param2.getName.return_value = "KNOB2"

        # Mock parameter service
        mock_lsa_client._parameterService.findParameters.return_value = [mock_param1, mock_param2]
        mock_lsa_client._ParametersRequestBuilder.return_value.setAccelerator.return_value.setParameterTypeName.return_value.build.return_value = MagicMock()

        result = find_knob_names(mock_lsa_client)

        assert "KNOB1" in result
        assert "KNOB2" in result
        assert result == sorted(result)  # Should be sorted

    @patch('omc3.machine_data_extraction.lsa_knobs.pjlsa')
    def test_find_knob_names_with_regexp_filter(self, mock_pjlsa):
        """Test finding knob names with regexp filtering."""
        from omc3.machine_data_extraction.lsa_knobs import find_knob_names

        mock_lsa_client = MagicMock()
        mock_accelerator = MagicMock()
        mock_lsa_client._getAccelerator.return_value = mock_accelerator

        # Create mock parameters
        mock_param1 = MagicMock()
        mock_param1.getName.return_value = "KNOB_X"

        mock_param2 = MagicMock()
        mock_param2.getName.return_value = "KNOB_Y"

        mock_param3 = MagicMock()
        mock_param3.getName.return_value = "OTHER"

        mock_lsa_client._parameterService.findParameters.return_value = [mock_param1, mock_param2, mock_param3]

        result = find_knob_names(mock_lsa_client, regexp="KNOB")

        assert "KNOB_X" in result
        assert "KNOB_Y" in result
        assert "OTHER" not in result

    @patch('omc3.machine_data_extraction.lsa_knobs.pjlsa')
    def test_filter_non_existing_knobs(self, mock_pjlsa):
        """Test filtering non-existing knobs."""
        from omc3.machine_data_extraction.lsa_knobs import filter_non_existing_knobs

        mock_lsa_client = MagicMock()

        # Mock _getParameter to return None for non-existing knobs
        def get_parameter_side_effect(knob):
            if knob == "KNOB_EXISTS":
                return MagicMock()
            return None

        mock_lsa_client._getParameter.side_effect = get_parameter_side_effect

        knobs = ["KNOB_EXISTS", "KNOB_NOT_EXISTS"]
        result = filter_non_existing_knobs(mock_lsa_client, knobs)

        assert "KNOB_EXISTS" in result
        assert "KNOB_NOT_EXISTS" not in result

    @patch('omc3.machine_data_extraction.lsa_knobs.pjlsa')
    @patch('omc3.machine_data_extraction.lsa_knobs.find_knob_names')
    def test_get_trim_history(self, mock_find_knobs, mock_pjlsa):
        """Test trim history retrieval."""
        from omc3.machine_data_extraction.lsa_knobs import get_trim_history

        mock_lsa_client = MagicMock()
        mock_trim_tuple = MagicMock()

        # Mock getTrims to return trims dict
        mock_lsa_client.getTrims.return_value = {
            "KNOB1": mock_trim_tuple,
            "KNOB2": mock_trim_tuple,
        }

        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=1)

        result = get_trim_history(
            mock_lsa_client,
            beamprocess="RAMP",
            knobs=["KNOB1", "KNOB2"],
            start_time=start_time,
            end_time=now,
            accelerator="lhc",
        )

        assert isinstance(result, TrimHistories)
        assert result.beamprocess == "RAMP"
        assert result.accelerator == "lhc"
        assert len(result.trims) == 2


@pytest.mark.usefixtures("mock_pjlsa")
class TestLSABeamprocessFunctions:
    """Tests for LSA beamprocess extraction functions."""

    @patch('omc3.machine_data_extraction.lsa_beamprocesses.pjlsa')
    def test_get_beamprocess_with_fill_at_time(self, mock_pjlsa):
        """Test retrieving beamprocess and fill information."""
        from omc3.machine_data_extraction.data_classes import BeamProcessInfo, FillInfo
        from omc3.machine_data_extraction.lsa_beamprocesses import get_beamprocess_with_fill_at_time

        mock_spark = MagicMock()
        mock_lsa_client = MagicMock()

        now = datetime.now(timezone.utc)
        bp_start = now - timedelta(minutes=5)

        bp_info = BeamProcessInfo(
            name="RAMP",
            accelerator="lhc",
            context_category="PHYSICS",
            start_time=bp_start,
            category="CYCLE",
            description="Ramp cycle",
        )
        fill_info = FillInfo(
            no=12345,
            accelerator="lhc",
            start_time=bp_start,
            beamprocesses=[(bp_start, "RAMP")],
        )

        with patch(
            'omc3.machine_data_extraction.lsa_beamprocesses.get_active_beamprocess_at_time',
            return_value=bp_info,
        ), patch(
            'omc3.machine_data_extraction.lsa_beamprocesses.get_beamprocesses_for_fills',
            return_value=[fill_info],
        ):
            result_fill, result_bp = get_beamprocess_with_fill_at_time(
                mock_lsa_client,
                mock_spark,
                now,
                accelerator="lhc",
            )

            assert result_bp.name == "RAMP"
            assert result_bp.accelerator == "lhc"
            assert result_fill.no == 12345

    @patch('omc3.machine_data_extraction.lsa_beamprocesses.pjlsa')
    def test_get_beamprocess_with_fill_at_time_missing_bp(self, mock_pjlsa):
        """Test that missing beamprocess in fill raises ValueError."""
        from omc3.machine_data_extraction.data_classes import BeamProcessInfo, FillInfo
        from omc3.machine_data_extraction.lsa_beamprocesses import get_beamprocess_with_fill_at_time

        mock_spark = MagicMock()
        mock_lsa_client = MagicMock()

        now = datetime.now(timezone.utc)
        bp_start = now - timedelta(minutes=5)

        bp_info = BeamProcessInfo(
            name="RAMP",
            accelerator="lhc",
            context_category="PHYSICS",
            start_time=bp_start,
            category="CYCLE",
            description="Ramp cycle",
        )
        fill_info = FillInfo(
            no=12345,
            accelerator="lhc",
            start_time=bp_start,
            beamprocesses=[(bp_start, "OTHER")],
        )

        with patch(
            'omc3.machine_data_extraction.lsa_beamprocesses.get_active_beamprocess_at_time',
            return_value=bp_info,
        ), patch(
            'omc3.machine_data_extraction.lsa_beamprocesses.get_beamprocesses_for_fills',
            return_value=[fill_info],
        ), pytest.raises(ValueError, match="Beamprocess 'RAMP' was not found"):
            get_beamprocess_with_fill_at_time(
                mock_lsa_client,
                mock_spark,
                now,
                accelerator="lhc",
            )


@pytest.mark.usefixtures("mock_pjlsa")
class TestLSAOpticsFunctions:
    """Tests for LSA optics extraction functions."""

    @patch('omc3.machine_data_extraction.lsa_optics.pjlsa')
    def test_get_optics_for_beamprocess_at_time(self, mock_pjlsa):
        """Test retrieving optics information for a beamprocess."""
        from omc3.machine_data_extraction.lsa_optics import get_optics_for_beamprocess_at_time

        mock_lsa_client = MagicMock()

        # Create mock beamprocess info
        mock_bp_info = MagicMock()
        mock_bp_info.name = "RAMP"
        mock_bp_info.accelerator = "lhc"
        mock_bp_info.start_time = datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc)

        mock_item = MagicMock()
        mock_item.time = 0
        mock_item.name = "OPTICSYEAR1"
        mock_item.id = "001"

        # Mock LSA method
        mock_lsa_client.getOpticTable.return_value = [mock_item]

        now = datetime.now(timezone.utc)

        result = get_optics_for_beamprocess_at_time(
            mock_lsa_client,
            now,
            mock_bp_info,
        )

        assert result.name == "OPTICSYEAR1"
        assert result.id == "001"

    @patch('omc3.machine_data_extraction.lsa_optics.pjlsa')
    def test_get_optics_for_beamprocess_at_time_no_match(self, mock_pjlsa):
        """Test that missing optics entry raises ValueError."""
        from omc3.machine_data_extraction.lsa_optics import get_optics_for_beamprocess_at_time

        mock_lsa_client = MagicMock()

        mock_bp_info = MagicMock()
        mock_bp_info.name = "RAMP"
        mock_bp_info.accelerator = "lhc"
        mock_bp_info.start_time = datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc)

        mock_item = MagicMock()
        mock_item.time = 999999
        mock_item.name = "OPTICSYEAR1"
        mock_item.id = "001"

        mock_lsa_client.getOpticTable.return_value = [mock_item]

        now = datetime(2025, 1, 1, 11, 0, 1, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="No optics found"):
            get_optics_for_beamprocess_at_time(
                mock_lsa_client,
                now,
                mock_bp_info,
            )


@pytest.mark.usefixtures("mock_pjlsa")
class TestKnobDefinitions:
    """Tests for knob definition extraction."""

    @patch('omc3.machine_data_extraction.lsa_knobs.pjlsa')
    def test_get_knob_definition(self, mock_pjlsa):
        """Test retrieving knob definition."""
        from omc3.machine_data_extraction.lsa_knobs import get_knob_definition

        mock_lsa_client = MagicMock()

        # Create mock Java knob definition
        mock_definition = MagicMock()
        mock_definition.getName.return_value = "KNOB1"
        mock_definition.getDescription.return_value = "Example knob"

        # Mock LSA method (would retrieve actual definition)
        mock_lsa_client.getKnobDefinition.return_value = mock_definition

        result = get_knob_definition(mock_lsa_client, "KNOB1", "OPTICSYEAR1")

        # Result should be a KnobDefinition object
        from omc3.machine_data_extraction.data_classes import KnobDefinition
        assert isinstance(result, KnobDefinition)
        assert result.name == "KNOB1"


class TestLastTrimExtraction:
    """Tests for extracting last trim values."""

    def test_get_last_trim_from_history(self):
        """Test extracting latest trim value from trim history."""
        from omc3.machine_data_extraction.lsa_knobs import get_last_trim

        # Create mock TrimTuple objects (named tuples with times and data)
        mock_trim1 = MagicMock()
        mock_trim1.times = [1, 2, 3]
        mock_trim1.data = [0.1, 0.2, 0.3]

        mock_trim2 = MagicMock()
        mock_trim2.times = [10, 20, 30]
        mock_trim2.data = [1.0, 2.0, 3.0]

        trims = {"KNOB1": mock_trim1, "KNOB2": mock_trim2}

        result = get_last_trim(trims)

        # Should return the last value for each knob
        assert "KNOB1" in result
        assert "KNOB2" in result
        assert result["KNOB1"] == 0.3  # Last data value for KNOB1
        assert result["KNOB2"] == 3.0  # Last data value for KNOB2
