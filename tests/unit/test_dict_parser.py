import pytest
import sys

from . import context
from parser.dict_parser import ParameterError, Parameter, DictParser
from utils.logging_tools import TempStringLogger


def test_deep_dict():
    parser = DictParser({
        'sub': {'param': Parameter('param', type=int)},
        'sub2': {'suub': {'param': Parameter("param", type=str)}}
    }, strict=True)

    # parser.tree()
    opt = {
        'sub': {'param': 4},
        'sub2': {'suub': {'param': "myString"}}
    }

    opt = parser.parse_arguments(opt)
    assert opt.sub.param == 4
    assert opt.sub2.suub.param == "myString"


def test_add_param_loc():
    parser = DictParser()
    parser.add_parameter(Parameter("test", default="def"), loc="sub.suub.suuub")
    assert parser.dictionary["sub"]["suub"]["suuub"]["test"].name == "test"
    assert parser.dictionary["sub"]["suub"]["suuub"]["test"].default == "def"


def test_add_param_loc2():
    parser = DictParser()
    parser.add_parameter("test", loc="sub.suub.suuub", default="def")
    assert parser.dictionary["sub"]["suub"]["suuub"]["test"].name == "test"
    assert parser.dictionary["sub"]["suub"]["suuub"]["test"].default == "def"


def test_add_param_already_exist_loc():
    parser = DictParser()
    parser.add_parameter("test", loc="sub.suub")
    with pytest.raises(ParameterError):
        parser.add_parameter("suub", loc="sub")


def test_add_parameter_dict():
    parser = DictParser()
    parser.add_parameter("param1", loc="sub.suub")
    parser2 = DictParser()
    parser2.add_parameter("param2")
    parser2.add_parameter_dict(parser.dictionary, loc="sb")
    assert parser2.dictionary["sb"]["sub"]["suub"]["param1"].name == "param1"
    assert parser2.dictionary["param2"].name == "param2"


def test_most_basic_init():
    p = Parameter(name='test')
    assert p.name == 'test'


def test_missing_name():
    with pytest.raises(TypeError):
        Parameter(default="a")


def test_default_not_a_list_with_nargs():
    with pytest.raises(ParameterError):
        Parameter(name="test", default="a", nargs="+")


def test_default_not_in_choices():
    with pytest.raises(ParameterError):
        Parameter(name="test", default="a", choices=["b", "c"])


def test_default_not_in_choices_list():
    with pytest.raises(ParameterError):
        Parameter(name="test", default=["a", "b"], choices=["b", "c"], nargs="+")


def test_default_not_of_type():
    with pytest.raises(ParameterError):
        Parameter(name="test", default=3, type=str)


def test_choices_not_iterable():
    with pytest.raises(ParameterError):
        Parameter(name="test", choices=3)


def test_choices_not_of_type():
    with pytest.raises(ParameterError):
        Parameter(name="test", choices=["a", 3], type=str)


def test_name_not_string():
    with pytest.raises(ParameterError):
        Parameter(name=5)

    with pytest.raises(ParameterError):
        DictParser({5: dict()})


def test_name_not_key():
    with pytest.raises(ParameterError):
        DictParser({"test": Parameter(name="nottest")})


def test_print_tree():
    parser = DictParser()
    param = Parameter("test", default="c", required=False, choices=["a", "b", "c"], help="help")
    loc = "sub.suub.suuub"
    parser.add_parameter(param, loc=loc)
    parser_module = sys.modules[parser.tree.__module__].__name__
    with TempStringLogger(parser_module) as log:
        parser.tree()

    text = log.get_log()
    for l in loc.split("."):
        assert l in text
    for attr in ["name", "default", "required", "choices", "help"]:
        assert str(getattr(param, attr)) in text
