"""
PETRA
-----

Accelerator-Class for the ``PETRA`` machine.
Model creation not implemented yet.
"""
from omc3.model.accelerators.accelerator import Accelerator, AcceleratorDefinitionError
from omc3.model.constants import PLANE_TO_HV

EXCITER_BPM = "BPM_SOR_13"


class Petra(Accelerator):
    NAME = "petra"

    # Public Methods ##########################################################

    def verify_object(self):  # TODO: Maybe more checks?
        if self.model_dir is None:  # is the class is used to create full response?
            raise AcceleratorDefinitionError("PETRA doesn't have a model creation yet, calling it this "
                                             "way is most probably wrong.")

    def get_exciter_bpm(self, plane, commonbpms):
        return [list(commonbpms).index(EXCITER_BPM), EXCITER_BPM], f"KIFB{PLANE_TO_HV[plane]}N"
