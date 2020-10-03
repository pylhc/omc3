import numpy as np
import pytest

from omc3.utils import stats


@pytest.mark.basic
def test_circular_zeros(zeros):
    assert stats.circular_mean(zeros) == 0
    assert stats.circular_error(zeros) == 0


@pytest.mark.basic
def test_circular_nans(a_nan):
    assert np.isnan(stats.circular_mean(a_nan))
    assert stats.circular_nanmean(a_nan) == 0.0


@pytest.mark.basic
def test_nanhandling():
    vector = np.array([355.0, 0.0, 5.0, np.nan])
    assert stats.circular_nanmean(vector) == stats.circular_mean(vector[:-1])
    assert stats.weighted_nanmean(vector) == stats.weighted_mean(vector[:-1])
    assert stats.weighted_nanrms(vector) == stats.weighted_rms(vector[:-1])
    vector = np.array([[355.0, 0.0, 5.0, 0.0], [355.0, 0.0, 5.0, 0.0], [355.0, 0.0, 5.0, np.nan]])
    assert np.all(
        stats.circular_nanerror(vector, axis=1) == stats.circular_error(vector[:, :-1], axis=1)
    )


@pytest.mark.basic
def test_circular_empties():
    empty = np.array([])
    with pytest.warns(RuntimeWarning):
        stats.circular_mean(empty)
    with pytest.warns(RuntimeWarning):
        stats.circular_error(empty)


@pytest.mark.basic
def test_mean_ones_errors_is_like_no_errors(some_numbers):
    assert stats.circular_mean(
        some_numbers, errors=np.ones(len(some_numbers))
    ) == stats.circular_mean(some_numbers)


@pytest.mark.basic
def test_opposite_values_cancel_out():
    vector = np.array([355.0, 0.0, 5.0])
    assert stats.circular_mean(vector, period=360.0) == pytest.approx(0.0)


@pytest.mark.basic
def test_one_important_number(some_numbers):
    errors = np.ones(10)
    errors[1:] = np.inf
    assert stats.circular_mean(some_numbers, errors=errors) == some_numbers[0]
    assert stats.circular_error(
        some_numbers, errors=errors, t_value_corr=False
    ) == stats.circular_error(errors[0], errors=1.0, t_value_corr=False)


@pytest.mark.basic
def test_mean_fall_back_to_linear():
    vector = np.arange(10)
    assert stats.circular_mean(vector, period=100000) == pytest.approx(np.mean(vector))


@pytest.mark.basic
def test_two_important_number(some_numbers):
    errors = np.ones(10)
    errors[2:] = np.inf
    assert stats.circular_mean(some_numbers, errors=errors) == stats.circular_mean(some_numbers[:2])
    assert stats.circular_error(
        some_numbers, errors=errors, t_value_corr=False
    ) == stats.circular_error(some_numbers[:2], errors=np.ones(2), t_value_corr=False)


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
