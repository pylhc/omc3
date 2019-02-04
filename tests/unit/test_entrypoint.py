import os
import pytest
import tempfile

from .context import omc3
from omc3.utils.entrypoint import (EntryPointParameters,
                                   entrypoint, EntryPoint,
                                   ArgumentError, ParameterError, OptionsError,
                                   )
from omc3.utils.entry_datatypes import get_multi_class, DictAsString, BoolOrString, BoolOrList
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
    opt, unknown = paramtest_function(
        name="myname",
        int=3,
        list=[4, 5, 6],
        unknown="myfinalargument"
    )
    assert(opt.name == "myname")
    assert(opt.int == 3)
    assert(len(opt.list) == 3)
    assert(opt.list[1] == 5)
    assert(len(unknown) > 0)


def test_as_string():
    opt, unknown = paramtest_function(
        ["--name", "myname",
         "--int", "3",
         "--list", "4", "5", "6",
         "--other"]
    )
    assert(opt.name == "myname")
    assert(opt.int == 3)
    assert(len(opt.list) == 3)
    assert(opt.list[1] == 5)
    assert(len(unknown) > 0)


def test_as_config():
    with tempfile.TemporaryDirectory() as cwd:
        cfg_file = os.path.join(cwd, "config.ini")
        with open(cfg_file, "w") as f:
            f.write("\n".join([
                "[Section]",
                "name = 'myname'",
                "int = 3",
                "list = 4, 5, 6",
                "unknown = 'other'",
            ]))


        # test config as kwarg
        opt1, unknown1 = paramtest_function(
            entry_cfg=cfg_file, section="Section"
        )

        # test config as commandline args
        opt2, unknown2 = paramtest_function(
            ["--entry_cfg", cfg_file, "--section", "Section"]
        )

    assert opt1.name == "myname"
    assert opt1.int == 3
    assert len(opt1.list) == 3
    assert opt1.list[1] == 5
    assert len(unknown1) > 0

    assert opt2.name == "myname"
    assert opt2.int == 3
    assert len(opt2.list) == 3
    assert opt2.list[1] == 5
    assert len(unknown2) > 0


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


# Test Special Datatypes

def test_multiclass():
    IntOrStr = get_multi_class(int, str)

    @entrypoint([dict(flags="--ios", name="ios", type=IntOrStr)], strict=True)
    def fun(opt):
        return opt

    opt = fun(ios=3)
    assert(opt.ios == 3)

    opt = fun(ios='3')
    assert(opt.ios == '3')

    opt = fun(["--ios", "3"])
    assert(opt.ios == 3)

    opt = fun(["--ios", "'3'"])
    assert(opt.ios == "'3'")


def test_dict_as_string():
    @entrypoint([dict(flags="--dict", name="dict", type=DictAsString)], strict=True)
    def fun(opt):
        return opt

    opt = fun(dict={'int': 5, 'str': 'hello'})
    assert(opt.dict['int'] == 5)
    assert(opt.dict['str'] == 'hello')

    opt = fun(["--dict", "{'int': 5, 'str': 'hello'}"])
    assert(opt.dict['int'] == 5)
    assert(opt.dict['str'] == 'hello')


def test_bool_or_str():
    @entrypoint([dict(flags="--bos", name="bos", type=BoolOrString)], strict=True)
    def fun(opt):
        return opt

    opt = fun(bos=True)
    assert(opt.bos == True)

    opt = fun(bos='myString')
    assert(opt.bos == 'myString')

    opt = fun(["--bos", "False"])
    assert(opt.bos == False)

    opt = fun(["--bos", "1"])
    assert(opt.bos == True)

    opt = fun(["--bos", "myString"])
    assert(opt.bos == "myString")

    with tempfile.TemporaryDirectory() as cwd:
        cfg_file = os.path.join(cwd, "bos.ini")
        with open(cfg_file, "w") as f:
            f.write("[Section]\nbos = 'myString'")
        opt = fun(entry_cfg=cfg_file)
    assert(opt.bos == "myString")


def test_bool_or_str_cfg():
    @entrypoint([dict(flags="--bos1", name="bos1", type=BoolOrString),
                 dict(flags="--bos2", name="bos2", type=BoolOrString)], strict=True)
    def fun(opt):
        return opt

    with tempfile.TemporaryDirectory() as cwd:
        cfg_file = os.path.join(cwd, "bos.ini")
        with open(cfg_file, "w") as f:
            f.write("[Section]\nbos1 = 'myString'\nbos2 = True")
        opt = fun(entry_cfg=cfg_file)
    assert(opt.bos1 == 'myString')
    assert(opt.bos2 == True)


def test_bool_or_list():
    @entrypoint([dict(flags="--bol", name="bol", type=BoolOrList)], strict=True)
    def fun(opt):
        return opt

    opt = fun(bol=True)
    assert(opt.bol == True)

    opt = fun(bol=[1, 2])
    assert(opt.bol == [1, 2])

    opt = fun(["--bol", "[1, 2]"])
    assert(opt.bol == [1, 2])

    opt = fun(["--bol", "0"])
    assert(opt.bol == False)

    opt = fun(["--bol", "True"])
    assert(opt.bol == True)


def test_bool_or_list_cfg():
    @entrypoint([dict(flags="--bol1", name="bol1", type=BoolOrList),
                 dict(flags="--bol2", name="bol2", type=BoolOrList)], strict=True)
    def fun(opt):
        return opt

    with tempfile.TemporaryDirectory() as cwd:
        cfg_file = os.path.join(cwd, "bol.ini")
        with open(cfg_file, "w") as f:
            f.write("[Section]\nbol1 = 1,2\nbol2 = True")
        opt = fun(entry_cfg=cfg_file)
    assert(opt.bol1 == [1, 2])
    assert(opt.bol2 == True)


# Example Parameter Definitions ################################################


def get_simple_params():
    """ Parameters as a list of dicts, to test this behaviour as well."""
    return [{"name": "arg1", "flags": "--a1", },
            {"name": "arg2", "flags": "--a2", },]


def get_testing_params():
    """ Parameters as a dict of dicts, to test this behaviour as well."""
    return {
        "name": dict(flags="--name", type=str),
        "int": dict(flags="--int", type=int),
        "list": dict(flags="--list", type=int, nargs="+")
    }


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


@entrypoint(get_testing_params())
def paramtest_function(opt, unknown):
    return opt, unknown
