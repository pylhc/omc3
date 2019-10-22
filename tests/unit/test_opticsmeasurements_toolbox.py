import pytest
import numpy as np
import pandas as pd
from . import context
from optics_measurements import toolbox as tb

ARRAY_LENGTH = 10


def test_df_diff_zero(random, zeros):
    assert all(random == tb.df_diff(*_df(random, zeros)))


def test_df_sum_zero(random, zeros):
    assert all(random == tb.df_sum(*_df(random, zeros)))


def test_df_sum_diff():
    a, b = _arand(), _arand()
    sum = tb.df_sum(*_df(a, b))
    diff = tb.df_diff(*_df(sum, b))
    assert _numerically_equal(a, diff)


def test_df_ratio_one(random, ones):
    assert all(random == tb.df_ratio(*_df(random, ones)))


def test_df_ratio_zero(random, zeros):
    assert not sum(tb.df_ratio(*_df(zeros, random)))
    with pytest.warns(RuntimeWarning):
        tb.df_ratio(*_df(random, zeros))


def test_df_prod_zero(random, zeros):
    assert not sum(tb.df_prod(*_df(random, zeros)))


def test_df_prod_one(random, ones):
    assert all(random == tb.df_prod(*_df(random, ones)))


def test_df_prod_ratio():
    a, b = _arand(), _arand()
    prod = tb.df_prod(*_df(a, b))
    ratio = tb.df_ratio(*_df(prod, b))
    assert _numerically_equal(a, ratio)


def test_df_rel_diff(random, zeros, ones):
    assert all(-tb.df_rel_diff(*_df(zeros, random)) == ones)
    with pytest.warns(RuntimeWarning):
        tb.df_rel_diff(*_df(random, zeros))


# Test with errors ---

def test_df_err_sum():
    a, b = _erand(), _erand()
    err_sum = tb.df_err_sum(*_df(a, b))
    sum = tb.df_sum(*_df(a, b))
    assert all(err_sum > 0)
    assert all(sum >= err_sum)


def test_df_rel_err_sum():
    a, b, aerr, berr = _arand(), _arand(), _erand(), _erand()
    err_sum = tb.df_rel_err_sum(*_df(a, b, aerr, berr))
    assert all(err_sum > 0)


def test_df_other():
    # basic functionality is tested, just check that these run
    a, b, aerr, berr = _arand(), _arand(), _erand(), _erand()
    df_and_cols = _df(a, b, aerr, berr)
    sum = tb.df_sum_with_err(*df_and_cols)
    diff = tb.df_diff_with_err(*df_and_cols)
    rel_diff = tb.df_rel_diff_with_err(*df_and_cols)
    ratio = tb.df_ratio_with_err(*df_and_cols)
    prod = tb.df_prod_with_err(*df_and_cols)

    for res in (sum, diff, rel_diff, ratio, prod):
        assert len(res) == 2
        assert len(res[0]) == len(res[1])
        assert len(res[0]) == len(a)


# Angular tests


def test_ang():
    a, b = _arand(), _arand()
    diff = tb.df_ang_diff(*_df(a, b))
    sum = tb.ang_sum(a, b)
    for res in (diff, sum):
        assert len(res) == len(a)
        assert all(-0.5 <= res)
        assert all(res <= 0.5)


# Helper ---


def _df(*cols):
    return (pd.DataFrame(data=cols).T, *range(len(cols)))


def _numerically_equal(a, b):
    return all(np.isclose(a, b, rtol=np.finfo(float).eps))


def _arand():
    array = np.zeros(ARRAY_LENGTH)
    while any(array == 0):
        array = np.random.rand(ARRAY_LENGTH) - .5
    return array


def _erand():
    return np.abs(_arand())


# Fixtures ---


@pytest.fixture
def random():
    return _arand()


@pytest.fixture
def zeros():
    return np.zeros(ARRAY_LENGTH)


@pytest.fixture
def ones():
    return np.ones(ARRAY_LENGTH)
