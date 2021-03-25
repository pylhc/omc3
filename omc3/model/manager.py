"""
Manager
-------

This module provides high-level functions to manage most functionality of ``model``.
It contains entrypoint wrappers to get accelerator classes or their instances.
"""
from generic_parser.entrypoint_parser import entrypoint, EntryPoint, EntryPointParameters
from omc3.model.accelerators import lhc, ps, esrf, psbooster, skekb, petra, iota

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
def get_accelerator(opt, other_opt):
    """
    Returns the `Accelerator` instance of the desired accelerator, as given at the commandline.
    """
    if not isinstance(opt.accel, str):
        # assume it's the class
        return opt.accel
    return ACCELS[opt.accel](other_opt)


@entrypoint(_get_params())
def get_parsed_opt(opt, other_opt):
    """Get all accelerator related options as a `dict`."""
    accel = ACCELS[opt.accel]
    parser = EntryPoint(accel.get_parameters(), strict=True)
    accel_opt = parser.parse(other_opt)
    return {**opt, **accel_opt}
