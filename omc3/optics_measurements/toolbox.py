"""
Toolbox
-------

This module contains helper functionality for ``optics_measurements``.
It provides functions to perform regularly used simple calculations.
"""
import numpy as np


def df_diff(df, a_col, b_col):
    """ Returns a column containing the difference between a_col and b_col """
    return df.loc[:, a_col].to_numpy() - df.loc[:, b_col].to_numpy()


def df_sum(df, a_col, b_col):
    """ Returns a column containing the sum of a_col and b_col """
    return df.loc[:, a_col].to_numpy() + df.loc[:, b_col].to_numpy()


def df_ratio(df, a_col, b_col):
    """ Returns a column containing the ratio between a_col and b_col """
    return df.loc[:, a_col].to_numpy() / df.loc[:, b_col].to_numpy()


def df_prod(df, a_col, b_col):
    """ Returns a column containing the product of a_col and b_col """
    return df.loc[:, a_col].to_numpy() * df.loc[:, b_col].to_numpy()


def df_rel_diff(df, a_col, b_col):
    """ Returns a column containing the difference between a_col and b_col relative to b_col """
    return df_ratio(df, a_col, b_col) - 1


# with Errors ---


def df_err_sum(df, a_err_col, b_err_col):
    """ Returns a column containing the root of the sum-of-squares of a_err_col and b_err_col """
    return np.sqrt(np.square(df.loc[:, a_err_col].to_numpy()) + np.square(df.loc[:, b_err_col].to_numpy()))


def df_rel_err_sum(df, a_col, b_col, a_err_col, b_err_col):
    """ Returns a column containing the root of the relative sum-of-square of a_col/a_err_col and b_col/b_err_col """
    return np.sqrt(np.square(df_ratio(df, a_err_col, a_col)) + np.square(df_ratio(df, b_err_col, b_col)))


def df_sum_with_err(df, a_col, b_col, a_err_col, b_err_col):
    """ Returns two columns containing the sum and the total errors of the given columns."""
    return df_sum(df, a_col, b_col), df_err_sum(df, a_err_col, b_err_col)


def df_diff_with_err(df, a_col, b_col, a_err_col, b_err_col):
    """ Returns two columns containing the difference and the total errors of the given columns."""
    return df_diff(df, a_col, b_col), df_err_sum(df, a_err_col, b_err_col)


def df_rel_diff_with_err(df, a_col, b_col, a_err_col, b_err_col):
    """ Returns two columns containing the relative difference and the total errors of the given columns."""
    return (df_rel_diff(df, a_col, b_col),
            np.abs(df_ratio(df, a_col, b_col)) * df_rel_err_sum(df, a_col, b_col, a_err_col, b_err_col))


def df_ratio_with_err(df, a_col, b_col, a_err_col, b_err_col):
    """ Returns two columns containing the ratio and the total errors of the given columns."""
    ratio = df_ratio(df, a_col, b_col)
    return ratio, np.abs(ratio) * df_rel_err_sum(df, a_col, b_col, a_err_col, b_err_col)


def df_prod_with_err(df, a_col, b_col, a_err_col, b_err_col):
    """ Returns two columns containing the product and the total errors of the given columns."""
    prod = df_prod(df, a_col, b_col)
    return prod, np.abs(prod) * df_rel_err_sum(df, a_col, b_col, a_err_col, b_err_col)


# Angular Calculations ---


def df_ang_diff(df, a_col, b_col):
    """ Returns a column containing the angular difference between angles a and b in [-0.5 , 0.5] """
    return ang_diff(df.loc[:, a_col].to_numpy(), df.loc[:, b_col].to_numpy())


def ang_diff(a, b):
    return ang_interval_check(ang_interval_check(a) - ang_interval_check(b))


def ang_sum(a, b):
    """ Returns a column containing the angular sum between angles a and b in [-0.5 , 0.5] """
    return ang_interval_check(ang_interval_check(a) + ang_interval_check(b))


def ang_interval_check(ang):
    """ Returns ang wrapped into [-0.5, 0.5] """
    return np.where(np.abs(ang) > 0.5, ang - np.sign(ang), ang)
