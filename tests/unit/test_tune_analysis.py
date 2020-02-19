from omc3.tune_analysis.bbq_tools import get_moving_average
from omc3.tune_analysis.detuning_tools import get_poly_fun


import pandas as pd
import numpy as np
np.random.seed(2020)


def test_moving_average():
    n_samples = 1000
    sin_data = pd.Series(np.sin(np.linspace(0, 2*np.pi, n_samples)))
    data = sin_data + (2*np.random.rand(n_samples) - 1)
    high_int = np.random.randint(low=0, high=n_samples, size=100)
    data[high_int[:50]] += 1.
    data[high_int[50:]] -= 1.
    kwargs = dict(
        min_val=-2,
        max_val=2,
        length=int(n_samples/10),
        fine_length=int(n_samples/10),
        fine_cut=1,
    )
    mav, std, mask = get_moving_average(data, **kwargs)
    assert sum(np.abs(mav) > 1.2) == 0
    assert (sin_data - mav).std() < (sin_data - data).std()/5  # 5 is handwavingly choosen
    # _plot_helper(sin_data, data, mav)


def test_get_poly_fun():
    x_arr = np.linspace(0, 100, 101)
    p0 = get_poly_fun(0)
    assert all(p0([1.43], x_arr) == 1.43)

    p1 = get_poly_fun(1)
    assert all(p1([0, 1], x_arr) == x_arr)

    p2 = get_poly_fun(2)
    assert all(p2([0, 2, 3.5], x_arr) == 2*x_arr + 3.5*(x_arr**2))


# Helper -----------------------------------------------------------------------

def _plot_helper(*series):
    ax = series[0].plot()
    if len(series) > 1:
        for s in series[1:]:
            s.plot(ax=ax)