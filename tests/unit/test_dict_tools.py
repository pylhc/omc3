import pytest

from . import context
from utils.dict_tools import DotDict


def test_dot_dict():
    dd = DotDict(dict(a=1, b='str', c=dict(e=[1,2,3])))
    assert dd.a == 1
    assert dd.b == 'str'
    assert dd.c.e == [1, 2, 3]

