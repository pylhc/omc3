"""
Tests for the definition files of the global correction knobs.
"""
import json

import numpy as np
import pytest

from omc3.model.accelerators.lhc import Lhc
from tests.conftest import INPUTS

MODELS_DIR = INPUTS / "models"

class TestLHCKnobs:
    @staticmethod
    def load_knobs_file(name: str, beam: int = None):
        correctors_dir = Lhc.DEFAULT_CORRECTORS_DIR
        if beam is not None:
            correctors_dir = correctors_dir / f"correctors_b{beam}"

        with open(correctors_dir / f"{name}_correctors.json") as f:
            return json.load(f)


    def test_all_json_files_are_readable(self):
        """ Check if all json files are readable. """
        for file in ["beta", "coupling"]:
            for beam in [1, 2]:
                knobs = self.load_knobs_file(file, beam=beam)
                assert knobs

        for file in ["triplet"]:
            knobs = self.load_knobs_file(file)
            assert knobs

    def test_both_beams_have_all_knobs(self):
        """ Check if all knobs are present in both beams. """
        for file in ["beta", "coupling"]:
            knobs_b1 = self.load_knobs_file(file, beam=1)
            knobs_b2 = self.load_knobs_file(file, beam=2)
            assert set(knobs_b1.keys()) == set(knobs_b2.keys())

    def test_variables_logic(self, accel_lhcb1: Lhc):
        """ Tests that the variables getting logic works. """
        vars_none = accel_lhcb1.get_variables(classes=[])
        assert not len(vars_none)

        vars_all = accel_lhcb1.get_variables(classes=None)
        assert len(vars_all)

        vars_mqy = accel_lhcb1.get_variables(classes=["MQY"])
        assert len(vars_mqy) < len(vars_all)
        assert all(mqy in vars_all for mqy in vars_mqy)

        vars_q = accel_lhcb1.get_variables(classes=["Q"])
        vars_mqy_q = accel_lhcb1.get_variables(classes=["MQY", "Q"])
        assert all(mqy in vars_mqy_q for mqy in vars_mqy)
        assert all(q in vars_mqy_q for q in vars_q)

        vars_mqy_extra = accel_lhcb1.get_variables(classes=["MQY", "test1", "test2"])
        assert all(mqy in vars_mqy_extra for mqy in vars_mqy)
        assert "test1" in vars_mqy_extra
        assert "test2" in vars_mqy_extra

        kq4_name = "kq4.l8b1"
        vars_mqy_extra_and_minus = accel_lhcb1.get_variables(classes=["MQY", "test1", "test2", "-test2", f"-{kq4_name}"])
        assert kq4_name in vars_mqy
        assert kq4_name in vars_mqy_extra
        assert kq4_name not in vars_mqy_extra_and_minus
        assert "test1" in vars_mqy_extra_and_minus
        assert "test2" not in vars_mqy_extra_and_minus

    def test_default_and_specific_variables(self, tmp_path, accel_lhcb1: Lhc, accel_lhcb2: Lhc):
        """ Tests if both json files are loaded correctly. This specific test only works
        with 2022-lhc models, as only here the MQM_TOP_2024 and MQM_INJ_2024 classes are implemented."""

        my_class = "MY_CLASS"
        my_dict = {my_class: ["A", "B", "C"]}
        user_json = tmp_path / "beta_correctors.json"
        user_json.write_text(json.dumps(my_dict))

        for accel in [accel_lhcb1, accel_lhcb2]:
            mqm_all = accel.get_variables(classes=["MQM_ALL"])  # from default json
            for mqm_class in ["MQM_TOP_2024", "MQM_INJ_2024"]:
                mqm_2024 = accel.get_variables(classes=[mqm_class])
                assert all(mqm in mqm_all for mqm in mqm_2024)
                assert any(mqm not in mqm_2024 for mqm in mqm_all)

            my_vars = accel.get_variables(classes=["MY_CLASS"])  # not present
            assert len(my_vars) == 1
            assert my_vars[0] == "MY_CLASS"

            accel.model_dir = tmp_path
            my_vars = accel.get_variables(classes=["MY_CLASS"])  # from user json
            assert len(my_vars) == len(my_dict[my_class])
            assert np.all(np.array(my_vars) == np.array(my_dict[my_class]))


# Helpers ------------------------------------------------------------------------------------------

@pytest.fixture
def accel_lhcb1():
    return Lhc(
        year="2024",
        beam=1,
        model_dir=MODELS_DIR / "2022_inj_b1_acd"
    )


@pytest.fixture
def accel_lhcb2():
    return Lhc(
        year="2024",
        beam=2,
        model_dir=MODELS_DIR / "2022_inj_b1_acd"
    )
