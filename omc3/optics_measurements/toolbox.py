import numpy as np


def df_diff(df, a_col, b_col):
    return _diff(df.loc[:, a_col].values, df.loc[:, b_col].values)


def _diff(a, b):
    return a - b


def df_ratio(df, a_col, b_col):
    return _ratio(df.loc[:, a_col].values, df.loc[:, b_col].values)


def _ratio(a, b):
    return a / b


def df_rel_diff(df, a_col, b_col):
    return _rel_diff(df.loc[:, a_col].values, df.loc[:, b_col].values)


def _rel_diff(a, b):
    return (a / b) - 1


def df_ang_diff(df, a_col, b_col):
    return _ang_diff(df.loc[:, a_col].values, df.loc[:, b_col].values)


def _ang_diff(a, b):
    return _interval_check(_interval_check(a) - _interval_check(b))


def ang_sum(a, b):
    return _interval_check(_interval_check(a) + _interval_check(b))


def _interval_check(ang):
    return np.where(np.abs(ang) > 0.5, ang - np.sign(ang), ang)


def df_prod(df, a_col, b_col):
    return df.loc[:, a_col].values * df.loc[:, b_col].values