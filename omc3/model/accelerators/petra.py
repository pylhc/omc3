"""
PETRA
-------------------
"""
import os
import datetime as dt
from model.accelerators.accelerator import Accelerator, AcceleratorDefinitionError
import logging

LOGGER = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(__file__)
#PS_DIR = os.path.join(CURRENT_DIR, "ps")


class Petra(Accelerator):
    NAME = "petra"

    # Public Methods ##########################################################

    def verify_object(self):  # TODO: Maybe more checks?
        if self.model_dir is None:  # is the class is used to create full response?
            raise AcceleratorDefinitionError("PETRA doesn't have a model creation yet, calling it this "
                                             "way is most probably wrong.")

