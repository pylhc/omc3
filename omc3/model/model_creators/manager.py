""" 
Model Creator Manager
---------------------

A manager that helps you find the optimal model creator of your favorite accelerator.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from omc3.model.model_creators.lhc_model_creator import (
    LhcBestKnowledgeCreator,
    LhcModelCreator,
    LhcCorrectionModelCreator,
    LhcSegmentCreator
)
from omc3.model.model_creators.ps_model_creator import PsModelCreator
from omc3.model.model_creators.psbooster_model_creator import PsboosterModelCreator
from omc3.utils.misc import StrEnum
from omc3.model.accelerators.lhc import Lhc
from omc3.model.accelerators.psbooster import Psbooster
from omc3.model.accelerators.ps import Ps

if TYPE_CHECKING:
    from omc3.model.accelerators.accelerator import Accelerator
    from omc3.model.model_creators.abstract_model_creator import ModelCreator


class CreatorType(StrEnum):
    NOMINAL: str = "nominal"
    BEST_KNOWLEDGE: str = "best_knowledge"
    SEGMENT: str = "segment"
    CORRECTION: str = "correction"


CREATORS: dict[str, dict[CreatorType, type]] = {
    Lhc.NAME: {
        CreatorType.NOMINAL: LhcModelCreator,
        CreatorType.BEST_KNOWLEDGE: LhcBestKnowledgeCreator,
        CreatorType.CORRECTION: LhcCorrectionModelCreator,
        CreatorType.SEGMENT: LhcSegmentCreator
    },
    Psbooster.NAME: {
        CreatorType.NOMINAL: PsboosterModelCreator
    },
    Ps.NAME: {
        CreatorType.NOMINAL: PsModelCreator
    },
}


def get_model_creator_class(
    accel: type[Accelerator] | Accelerator | str, 
    creator_type: CreatorType
    ) -> type[ModelCreator]:
    """ Returns the model creator for the given accelerator and creator type.
    This function will raise a ValueError if the accelerator or creator type is unknown.
    
    Args:
        accel: The accelerator to use, can be class, instance or name.
        creator_type: The type of model creator to use.

    Returns:
        The model creator class.
    """
    name = accel if isinstance(accel, str) else accel.NAME
    
    try:
        CREATORS[name]
    except KeyError:
        raise ValueError(f"Unknown accelerator '{name}' for a model creation.")
    
    try:
        return CREATORS[name][creator_type]
    except KeyError:
        raise ValueError(f"Unknown model creator type '{creator_type}' for accelerator '{name}'.")
