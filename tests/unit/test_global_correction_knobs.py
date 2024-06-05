""" 
Tests for the definition files of the global correction knobs.
"""
import json

from omc3.model.accelerators.lhc import LHC_DIR

CORRECTORS_DIR = LHC_DIR / "2012" / "correctors"


class TestLHCKnobs:
    @staticmethod
    def load_knobs_file(name: str, beam: int = None):
        correctors_dir = CORRECTORS_DIR
        if beam is not None:
            correctors_dir = correctors_dir / f"correctors_b{beam}"

        with open(correctors_dir / f"{name}_correctors.json", "r") as f:
            knobs = json.load(f)
        return knobs
    
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
