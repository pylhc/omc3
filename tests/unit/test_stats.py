import pytest
import numpy as np
from . import context
from utils import stats


def test_circular_zeros(zeros):
    assert stats.circular_mean(zeros) == 0
    assert stats.circular_error(zeros) == 0


def test_circular_nans(a_nan):
    assert np.isnan(stats.circular_mean(a_nan))


def test_circular_empties():
    empty = np.array([])
    with pytest.warns(RuntimeWarning):
        stats.circular_mean(empty)
    with pytest.warns(RuntimeWarning):
        stats.circular_error(empty)


def test_mean_ones_errors_is_like_no_errors(some_numbers):
    assert stats.circular_mean(some_numbers,
                               errors=np.ones(len(some_numbers))) ==\
           stats.circular_mean(some_numbers)


def test_opposite_values_cancel_out():
    vector = np.array([355., 0., 5.])
    assert stats.circular_mean(vector, period=360.) == pytest.approx(0.)


def test_one_important_number(some_numbers):
    errors = np.ones(10)
    errors[1:] = np.inf
    assert stats.circular_mean(some_numbers, errors=errors) == some_numbers[0]
    assert stats.circular_error(some_numbers, errors=errors, t_value_corr=False) ==\
           stats.circular_error(errors[0], errors=1., t_value_corr=False)


def test_mean_fall_back_to_linear():
    vector = np.arange(10)
    assert stats.circular_mean(vector, period=100000) ==\
           pytest.approx(np.mean(vector))


def test_two_important_number(some_numbers):
    errors = np.ones(10)
    errors[2:] = np.inf
    assert stats.circular_mean(some_numbers, errors=errors) == stats.circular_mean(some_numbers[:2])
    assert stats.circular_error(some_numbers, errors=errors, t_value_corr=False) ==\
           stats.circular_error(some_numbers[:2], errors=np.ones(2), t_value_corr=False)


# Utilities ###################################################################

@pytest.fixture
def zeros():
    return np.zeros(10)


@pytest.fixture
def a_nan():
    a_nan = np.zeros(10)
    a_nan[3] = np.nan
    return a_nan


@pytest.fixture
def some_numbers():
    return np.arange(10)
