"""
Manager
-------

This module provides high-level functions to manage most functionality of ``model``.
It contains entrypoint wrappers to get accelerator classes or their instances.
"""
from generic_parser.entrypoint_parser import entrypoint, EntryPoint, EntryPointParameters
from omc3.model.accelerators import lhc, ps, esrf, psbooster, skekb, petra, iota
from generic_parser.dict_parser import ArgumentError
from generic_parser.tools import silence
from omc3.utils.parsertools import print_help

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
    params.add_parameter(name="show_help", action="store_true", help="instructs the subsequent modules to print a help message")
    return params


@entrypoint(_get_params())
def get_accelerator(opt, other_opt):
    """
    Returns the `Accelerator` instance of the desired accelerator, as given at the commandline.
    """
    if not isinstance(opt.accel, str):
        # assume it's the class
        return opt.accel, False
    if not opt.show_help:
        return ACCELS[opt.accel](other_opt), False

    accelclass = ACCELS[opt.accel]
    print(f"--- {accelclass.NAME.upper()} Accelerator Class. Parameters:")
    print_help(accelclass.get_parameters())

    try :
        with silence():
            return accelclass(other_opt), True
    except SystemExit:
        return None, True


@entrypoint(_get_params())
def get_parsed_opt(opt, other_opt):
    """Get all accelerator related options as a `dict`."""
    accel = ACCELS[opt.accel]
    parser = EntryPoint(accel.get_parameters(), strict=True)
    accel_opt = parser.parse(other_opt)
    return {**opt, **accel_opt}