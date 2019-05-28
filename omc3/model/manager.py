"""
Manager
-------------------

Contains entrypoint wrappers to get accelerator classes or their instances
"""
from parser.entrypoint import entrypoint, EntryPoint, EntryPointParameters, split_arguments
from model.accelerators import lhc, ps, esrf, psbooster, skekb


ACCELS = {
    lhc.Lhc.NAME: lhc.Lhc,
    ps.Ps.NAME: ps.Ps,
    esrf.Esrf.NAME: esrf.Esrf,
    psbooster.Psbooster.NAME: psbooster.Psbooster,
    skekb.SKekB.NAME: skekb.SKekB,
    "JPARC": skekb.SKekB
}


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(flags=["--accel"], name="accel", required=True, choices=list(ACCELS.keys()),
                         help="Choose the accelerator to use.Can be the class already.")
    return params


@entrypoint(_get_params())
def get_accel_class(opt, cls_opt):
    """Returns accelerator class

    Keyword Args:
        accel: Choose the accelerator to use. Can be the class already, which is then returned.
    """
    if not isinstance(opt.accel, str):
        # assume it's the class
        return opt.accel

    accel = _get_parent_class(opt.accel)
    accel_cls = accel.get_class(cls_opt)
    return accel_cls


@entrypoint(_get_params())
def get_accel_instance(opt, other_opt):
    """Returns accelerator instance."""
    if not isinstance(opt.accel, str):
        accel_cls = opt.accel
    else:
        accel = _get_parent_class(opt.accel)
        accel_cls, other_opt = accel.get_class_and_unknown(other_opt)
    return accel_cls(other_opt)


@entrypoint(_get_params())
def get_accel_class_and_unkown(opt, cls_opt):
    """Returns accelerator class

    Keyword Args:
        accel: Choose the accelerator to use. Can be the class already, which is then returned.
    """
    if not isinstance(opt.accel, str):
        # assume it's the class
        return opt.accel

    accel = _get_parent_class(opt.accel)
    accel_cls, unknown_opt = accel.get_class_and_unknown(cls_opt)
    return accel_cls, unknown_opt


def get_accel_class_from_args(args=None):
    """ LEGACY-FUNCTION SHOULD BE REPLACED BY USING get_accel_class """
    parser = EntryPoint(_get_params())
    opt, class_args = parser.parse(args)

    accel = _get_parent_class(opt.accel)

    accel_args, rest_args = split_arguments(class_args, accel.get_class_parameters())
    accel_cls = accel.get_class(accel_args)
    return accel_cls, rest_args


def _get_parent_class(name):
    try:
        return ACCELS[name]
    except KeyError:
        raise ValueError(f"name should be one of: {ACCELS.keys()}")
