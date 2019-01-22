import os
import sys

import pytest
from .context import omc3
from omc3.utils.entrypoint import (EntryPointParameters,
                                   entrypoint, EntryPoint,
                                   ArgumentError, ParameterError, OptionsError,
                                   )
from omc3.utils.dict_tools import print_dict_tree
from omc3.utils import logging_tools
from omc3.utils.contexts import silence

LOG = logging_tools.get_logger(__name__)


# Tests ########################################################################


# Options Tests


def test_strict_wrapper_fail():
    with pytest.raises(OptionsError):
        @entrypoint(get_simple_params(), strict=True)
        def strict(opt, unknown):  # too many option-structures
            pass


def test_class_wrapper_fail():
    with pytest.raises(OptionsError):
        class MyClass(object):
            @entrypoint(get_simple_params())
            def fun(self, opt):  # too few option-structures
                pass


def test_normal_wrapper_fail():
    with pytest.raises(OptionsError):
        @entrypoint(get_simple_params())
        def fun(opt, unknown, more):  # too many option-structures
            pass


def test_class_functions():
    class MyClass(object):
        @classmethod
        @entrypoint(get_simple_params())
        def fun(cls, opt, unknown):
            pass


def test_instance_functions():
    class MyClass(object):
        @entrypoint(get_simple_params())
        def fun(self, opt, unknown):
            pass


# Parameter Tests


def test_wrong_param_creation_name():
    with pytest.raises(ParameterError):
        EntryPoint([{"flags": "--flag"}])


def test_wrong_param_creation_flags():
    with pytest.raises(ParameterError):
        EntryPoint([{"name": "test"}])


def test_wrong_param_creation_other():
    with pytest.raises(TypeError):
        EntryPoint([{"name": "test", "flags": "--flag", "other": "unknown"}])


def test_default_not_in_choices():
    with pytest.raises(ParameterError):
        EntryPoint([{"name": "test", "flags": "--flag", "default": "a", "choices": ["b", "c"]}])


def test_default_not_in_choices_list():
    with pytest.raises(ParameterError):
        EntryPoint([{"name": "test", "flags": "--flag",
                     "default": ["a", "b"], "choices": ["b", "c"],
                     "nargs": "+",
                     }])


def test_default_not_of_type():
    with pytest.raises(ParameterError):
        EntryPoint([{"name": "test", "flags": "--flag",
                     "default": 3, "type": str,
                     }])


def test_default_not_a_list_with_nargs():
    with pytest.raises(ParameterError):
        EntryPoint([{"name": "test", "flags": "--flag",
                     "default": "a", "nargs": "+",
                     }])


def test_choices_not_iterable():
    with pytest.raises((ParameterError, ValueError)):
        # Value error comes from argparse (would be caught in dict_parser as well)
        EntryPoint([{"name": "test", "flags": "--flag",
                     "choices": 3,
                     }])


def test_choices_not_of_type():
    with pytest.raises(ParameterError):
        EntryPoint([{"name": "test", "flags": "--flag",
                     "choices": ["b", 3], "type": str,
                     }])


def test_name_not_string():
    with pytest.raises(ParameterError):
        EntryPoint([{"name": 5, "flags": "--flag",
                     }])


# Argument Tests


def test_strict_pass():
    strict_function(accel="LHCB1", anint=3)


def test_strict_fail():
    with pytest.raises(ArgumentError):
        strict_function(accel="LHCB1", anint=3, unkown="not_found")


def test_as_kwargs():
    pass


def test_as_string():
    pass


def test_all_missing():
    with pytest.raises(SystemExit):
        with silence():
            some_function()


def test_required_missing():
    with pytest.raises(ArgumentError):
        some_function(accel="LHCB1")


def test_wrong_choice():
    with pytest.raises(ArgumentError):
        some_function(accel="accel", anint=3)


def test_wrong_type():
    with pytest.raises(ArgumentError):
        some_function(accel="LHCB1", anint=3.)


def test_wrong_type_in_list():
    with pytest.raises(ArgumentError):
        some_function(accel="LHCB1", anint=3, alist=["a", "b"])


def test_not_enough_length():
    with pytest.raises(ArgumentError):
        some_function(accel="LHCB1", anint=3, alist=[])


# Example Parameter Definitions ################################################


def get_simple_params():
    """ Parameters as a list of dicts, to test this behaviour as well."""
    return [
        {"name": "arg1",
         "flags": "--a1",
         },
        {"name": "arg2",
         "flags": "--a2",
         }
    ]


def get_params():
    """ Parameters defined with EntryPointArguments (which is a dict *cough*) """
    args = EntryPointParameters()
    args.add_parameter(name="accel",
                       flags=["-a", "--accel"],
                       help="Which accelerator: LHCB1 LHCB2 LHCB4? SPS RHIC TEVATRON",
                       choices=["LHCB1", "LHCB2", "LHCB5"],
                       required=True,
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
                       )
    args.add_parameter(name="anotherlist",
                       flags=["-k", "--alint"],
                       help="list.",
                       type=str,
                       nargs=3,
                       default=["a", "c", "f"],
                       choices=["a", "b", "c", "d", "e", "f"],
                       ),
    return args


# Example Wrapped Functions ####################################################


@entrypoint(get_params())
def some_function(options, unknown_options):
    LOG.debug("Some Function")
    print_dict_tree(options, print_fun=LOG.debug)
    LOG.debug("Unknown Options: \n {:s}".format(str(unknown_options)))
    LOG.debug("\n")


@entrypoint(get_params(), strict=True)
def strict_function(options):
    LOG.debug("Strict Function")
    print_dict_tree(options, print_fun=LOG.debug)
    LOG.debug("\n")

