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

from omc3.model.model_creators.abstract_model_creator import CreatedModel
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
    Returns (accel, help_requested):
        `accel` is the `Accelerator` instance of the desired accelerator, as given at the commandline.
        `help_requested` is a boolean stating if help was requested at any point

    """
    if not isinstance(opt.accel, str):
        # if it's the class already, we just return it
        return CreatedModel(opt.accel)

    if not opt.show_help:
        # if no help is requested, return the accelerator instance and fall through
        return CreatedModel(ACCELS[opt.accel](other_opt))

    # ----------------------------------------------------------------------------------------------
    # if we are still here, print the help
    accelclass = ACCELS[opt.accel]
    print(f"--- {accelclass.NAME.upper()} Accelerator Class. Parameters:")
    print_help(accelclass.get_parameters())

    # try creating the accelclass from the options
    try:
        with silence():
            return CreatedModel.help()
    except SystemExit:
        # if this fails, the accelclass options where incomplete, so we DON'T return an accel instance
        # but only the flag help_requested=True
        return CreatedModel.help()


@entrypoint(_get_params())
def get_parsed_opt(opt, other_opt):
    """Get all accelerator related options as a `dict`."""
    accel = ACCELS[opt.accel]
    parser = EntryPoint(accel.get_parameters(), strict=True)
    accel_opt = parser.parse(other_opt)
    return {**opt, **accel_opt}
