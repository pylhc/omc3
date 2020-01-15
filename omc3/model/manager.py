"""
Manager
-------------------

Contains entrypoint wrappers to get accelerator classes or their instances
"""
from generic_parser.entrypoint_parser import (EntryPoint, EntryPointParameters,
                                              entrypoint)

from omc3.model.accelerators import esrf, iota, lhc, ps, psbooster, skekb

ACCELS = {
    lhc.Lhc.NAME: lhc.Lhc,
    ps.Ps.NAME: ps.Ps,
    esrf.Esrf.NAME: esrf.Esrf,
    psbooster.Psbooster.NAME: psbooster.Psbooster,
    skekb.SKekB.NAME: skekb.SKekB,
    "JPARC": skekb.SKekB,
    iota.Iota.NAME: iota.Iota
}


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(name="accel", required=True, choices=list(ACCELS.keys()),
                         help="Choose the accelerator to use.Can be the class already.")
    return params


@entrypoint(_get_params())
def get_accelerator(opt, other_opt):
    """ Returns accelerator instance. """
    if not isinstance(opt.accel, str):
        # assume it's the class
        return opt.accel
    return ACCELS[opt.accel](other_opt)


@entrypoint(_get_params())
def get_parsed_opt(opt, other_opt):
    """ Get all accelerator related options as a nice dict. """
    accel = ACCELS[opt.accel]
    parser = EntryPoint(accel.get_parameters(), strict=True)
    accel_opt = parser.parse(other_opt)
    return {**opt, **accel_opt}
