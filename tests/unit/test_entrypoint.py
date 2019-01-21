import pytest
from .context import omc3
from omc3.utils.entrypoint import EntryPointParameters, entrypoint, EntryPoint, ArgumentError
from omc3.utils.dict_tools import print_dict_tree
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)

# Example Parameter Definitions ################################################


def _get_params():
    """ Parameters defined with EntryPointArguments (which is a dict *cough*) """
    args = EntryPointParameters()
    args.add_parameter(name="accel",
                       flags=["-a", "--accel"],
                       help="Which accelerator: LHCB1 LHCB2 LHCB4? SPS RHIC TEVATRON",
                       choices=["LHCB1","LHCB2","LHCB5"],
                       default="LHCB1")
    args.add_parameter(name="dict",
                       flags=["-d", "--dictionary"],
                       help="File with the BPM dictionary",
                       default="/test.notafile",
                       type=str)
    args.add_parameter(name="anumber",
                       flags=["-num", "--anum"],
                       help="Just a number.",
                       type=float,
                       default=19.,
                       )
    args.add_parameter(name="anint",
                       flags=["-i", "--int"],
                       help="Just a number.",
                       type=int,
                       required=True,
                       )
    args.add_parameter(name="alist",
                       flags=["-l", "--lint"],
                       help="Just a number.",
                       type=int,
                       nargs="+",
                       required=True,
                       )
    args.add_parameter(name="anotherlist",
                       flags=["-k", "--alint"],
                       help="list.",
                       type=str,
                       nargs=3,
                       default=["a", "c", "f"],
                       choices=["a", "b", "c", "d", "e", "f"]
                       ),
    return args

# Example Wrapped Functions ####################################################


@entrypoint(_get_params())
def some_function(options, unknown_options):
    LOG.debug("Some Function")
    print_dict_tree(options, print_fun=LOG.debug)
    LOG.debug("Unknown Options: \n {:s}".format(str(unknown_options)))
    LOG.debug("\n")


@entrypoint(_get_params(), strict=True)
def strict_function(options):
    LOG.debug("Strict Function")
    print_dict_tree(options, print_fun=LOG.debug)
    LOG.debug("\n")


class TestClass(object):
    @entrypoint(_get_params())
    def instance_function(self, options, unknowns):
        LOG.debug("Instance Function")
        print_dict_tree(options, print_fun=LOG.debug)
        LOG.debug("Unknown Options: \n {:s}".format(str(unknowns)))
        LOG.debug("\n")

    @classmethod
    @entrypoint(_get_params())
    def class_function(cls, options, unknowns):
        LOG.debug("Class Function")
        print_dict_tree(options, print_fun=LOG.debug)
        LOG.debug("Unknown Options: \n {:s}".format(str(unknowns)))
        LOG.debug("\n")


# Tests ########################################################################


def test_strict_pass():
    pass


def test_strict_fail():
    with pytest.raises(ArgumentError):
        pass


def test_normal_fail():
    with pytest.raises(ArgumentError):
        pass


def test_class_functions():
    pass


def test_as_kwargs():
    pass


def test_as_string():
    pass
