import numpy as np
import pandas as pd
import pytest

from omc3.optics_measurements import toolbox

ARRAY_LENGTH = 10


@pytest.mark.basic
def test_df_diff_zero(random, zeros):
    assert all(random == toolbox.df_diff(*_df(random, zeros)))


@pytest.mark.basic
def test_df_sum_zero(random, zeros):
    assert all(random == toolbox.df_sum(*_df(random, zeros)))


@pytest.mark.basic
def test_df_sum_diff():
    a, b = _arand(), _arand()
    sum_of_columns = toolbox.df_sum(*_df(a, b))
    diff = toolbox.df_diff(*_df(sum_of_columns, b))
    assert _numerically_equal(a, diff)


@pytest.mark.basic
def test_df_ratio_one(random, ones):
    assert all(random == toolbox.df_ratio(*_df(random, ones)))


@pytest.mark.basic
def test_df_ratio_zero(random, zeros):
    assert not sum(toolbox.df_ratio(*_df(zeros, random)))
    with pytest.warns(RuntimeWarning):
        toolbox.df_ratio(*_df(random, zeros))


@pytest.mark.basic
def test_df_prod_zero(random, zeros):
    assert not sum(toolbox.df_prod(*_df(random, zeros)))


@pytest.mark.basic
def test_df_prod_one(random, ones):
    assert all(random == toolbox.df_prod(*_df(random, ones)))


@pytest.mark.basic
def test_df_prod_ratio():
    a, b = _arand(), _arand()
    prod = toolbox.df_prod(*_df(a, b))
    ratio = toolbox.df_ratio(*_df(prod, b))
    assert _numerically_equal(a, ratio)


@pytest.mark.basic
def test_df_rel_diff(random, zeros, ones):
    assert all(-toolbox.df_rel_diff(*_df(zeros, random)) == ones)
    with pytest.warns(RuntimeWarning):
        toolbox.df_rel_diff(*_df(random, zeros))


# Test with errors ---
@pytest.mark.basic
def test_df_err_sum():
    a, b = _erand(), _erand()
    err_sum = toolbox.df_err_sum(*_df(a, b))
    sum_of_columns = toolbox.df_sum(*_df(a, b))
    assert all(err_sum > 0)
    assert all(sum_of_columns >= err_sum)


@pytest.mark.basic
def test_df_rel_err_sum():
    a, b, aerr, berr = _arand(), _arand(), _erand(), _erand()
    err_sum = toolbox.df_rel_err_sum(*_df(a, b, aerr, berr))
    assert all(err_sum > 0)


@pytest.mark.basic
def test_df_other():
    # basic functionality is tested, just check that these run
    a, b, aerr, berr = _arand(), _arand(), _erand(), _erand()
    df_and_cols = _df(a, b, aerr, berr)
    tuple_sum_columns_and_sum_errors = toolbox.df_sum_with_err(*df_and_cols)
    diff = toolbox.df_diff_with_err(*df_and_cols)
    rel_diff = toolbox.df_rel_diff_with_err(*df_and_cols)
    ratio = toolbox.df_ratio_with_err(*df_and_cols)
    prod = toolbox.df_prod_with_err(*df_and_cols)

    for res in (tuple_sum_columns_and_sum_errors, diff, rel_diff, ratio, prod):
        assert len(res) == 2
        assert len(res[0]) == len(res[1])
        assert len(res[0]) == len(a)


# Angular tests
@pytest.mark.basic
def test_ang():
    a, b = _arand(), _arand()
    diff = toolbox.df_ang_diff(*_df(a, b))
    sum_angles = toolbox.ang_sum(a, b)
    for res in (diff, sum_angles):
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
