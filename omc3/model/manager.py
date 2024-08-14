"""
Manager
-------

This module provides high-level functions to manage most functionality of ``model``.
It contains entrypoint wrappers to get accelerator classes or their instances.
"""
from generic_parser.entrypoint_parser import entrypoint, EntryPoint, EntryPointParameters
from omc3.model.accelerators import lhc, ps, esrf, psbooster, skekb, petra, iota

from omc3.model.accelerators.accelerator import Accelerator

ACCELS = {
    lhc.Lhc.NAME: lhc.Lhc,
    ps.Ps.NAME: ps.Ps,
    esrf.Esrf.NAME: esrf.Esrf,
    psbooster.Psbooster.NAME: psbooster.Psbooster,
    skekb.SKekB.NAME: skekb.SKekB,
    "JPARC": skekb.SKekB,
    petra.Petra.NAME: petra.Petra,
    iota.Iota.NAME: iota.Iota
}


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(name="accel", required=True, choices=list(ACCELS.keys()),
                         help="Choose the accelerator to use.Can be the class already.")
    return params


@entrypoint(_get_params())
def get_accelerator(opt, other_opt) -> Accelerator:
    """
    Returns (opt.accel, help_requested):
        `opt.accel` is the `Accelerator` instance of the desired accelerator, as given at the commandline.
        `help_requested` is a boolean stating if help was requested at any point

    """

    if not isinstance(opt.accel, str):
        # if it's the class already, we just return it
        return opt.accel

    return ACCELS[opt.accel](other_opt)


@entrypoint(_get_params())
def get_accelerator_class(opt, other):
    """
    """

    return ACCELS[opt.accel]

@entrypoint(_get_params())
def get_parsed_opt(opt, other_opt):
    """Get all accelerator related options as a `dict`."""
    accel = ACCELS[opt.accel]
    parser = EntryPoint(accel.get_parameters(), strict=True)
    accel_opt = parser.parse(other_opt)
    return {**opt, **accel_opt}
