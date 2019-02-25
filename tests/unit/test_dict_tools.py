from io import StringIO
import pytest


from . import context
from utils.dict_tools import DotDict, print_dict_tree


def test_dot_dict(simple_dict):
    dd = DotDict(simple_dict)
    assert dd.a == 1
    assert dd.b == 'str'
    assert dd.c.e == [1, 2, 3]


def test_get_subdict(simple_dict):
    dd = DotDict(simple_dict)
    sub = dd.get_subdict(["a", "b"])
    assert sub.a == 1
    assert sub.b == 'str'
    assert "c" not in sub


def test_print_tree(simple_dict):
    name = "A Dict"
    stream = StringIO()
    fun = lambda s: stream.write("\n" + s)

    print_dict_tree(simple_dict, name, print_fun=fun)

    text = stream.getvalue()
    # print(text)  # I'll leave it here, in case you want to see the dict
    assert name in text
    for l in simple_dict.keys():
        assert l in text


# Fixtures #####################################################################


@pytest.fixture()
def simple_dict():
    return dict(a=1, b='str', c=dict(e=[1,2,3]))
