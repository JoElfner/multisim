# -*- coding: utf-8 -*-
"""
Created on 30 Aug 2021

@author: Johannes Elfner
"""

import numpy as np
import pandas as pd


def r_squared(y, y_hat):
    """
    Calculate coefficient of determination :math:`R^2`.

    Will be calculated between the measurement/oberserved data `y` and the
    predicted/simulated data `y_hat`.

    `y_hat` may contain multiple columns. In this case, the coefficient with be
    calculated for **each** column separately.

    Parameters
    ----------
    y : np.array, pd.Series, pd.DataFrame
        Measurement/oberserved data.
    y_hat : np.array, pd.Series, pd.DataFrame
        Predicted/forecast/simulated data.
    """

    assert (
        not np.isnan(y).any().any()
    ), (  # double any for ndim >= 2
        '`y` contains NaNs. Please drop NaNs in y before calculating R^2.'
    )
    # broadcast y to 2D, if input is 2D to correctly calculate r^2 col wise
    if y_hat.ndim == 2:
        y = y[:, None] if y.ndim == 1 else y
    elif y_hat.ndim == 1 and y.ndim == 2:
        y_hat = y_hat[:, None]
    elif y_hat.ndim > 2:
        raise ValueError
    ss_res = ((y - y_hat) ** 2).sum(axis=0)  # residuals
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum()  # total sum of squares
    return 1 - ss_res / ss_tot


def r_squared_adj(p, r_sqrd=None, n=None, y=None, y_hat=None):
    r"""
    Calculate adjusted coefficient of determination :math:`R_{adj}^2`.

    Often also notated as :math:`\bar{R}^2`.

    Either pass the measurement/oberserved data `y` and the predicted/simulated
    data `y_hat` **OR** an already calculated value for `r_sqrd` and the number
    of samples `n`. The total number of explanatory variables `p` must be given
    in **any case**.

    `y_hat` may contain multiple columns. In this case, the coefficient with be
    calculated for **each** column separately.

    Parameters
    ----------
    p : int
        Total number of explanatory variables in the model, including
        higher order and interaction terms.
    r_sqrd : float, optional
        Non-adjusted R-squared value. Only required if `y` and `y_hat` are not
        given.
    n : int, optional
        Number of samples. Only required if `r_sqrd` is given.
    y : np.array, pd.Series, pd.DataFrame, optional
        Measurement/oberserved data. Only required, if :math:`R^2` (`r_sqrd`)
        is not given as argument.
    y_hat : np.array, pd.Series, pd.DataFrame, optional
        Predicted/forecast/simulated data. Only required, if :math:`R^2`
        (`r_sqrd`) is not given as argument.

    """

    if r_sqrd is not None:
        assert (n is not None) and (y is None) and (y_hat is None), (
            'If `r_sqrd` is given, `n` must be given and `y` and `y_hat` '
            'must not be given.'
        )
    else:  # if r_sqrd not given
        assert (n is None) and (y is not None) and (y_hat is not None), (
            'If `r_sqrd` is not given, `y` and `y_hat` must be given and'
            '`n` must not be given.'
        )
        breakpoint()
        print('false! n must be of the training data set!!!')
        print('also avoid printing R for p>n! raise error!')
        r_sqrd = r_squared(y, y_hat)  # get R^2
        n = y_hat.shape[0]  # get number of samples
    # set r2 range correctly (-inf -> 1. or -inf ->100.)
    max_r2 = 1.0 if r_sqrd <= 1.0 else 100.0
    # get adjusted r_sqrd
    r_sqrd_adj = max_r2 - (max_r2 - r_sqrd) * (n - 1) / (n - p - 1)
    return r_sqrd_adj


def mean_squared_err(y, y_hat, roll=False, window=None, min_periods=None):
    """
    Calculate the mean squared error (MSE).

    Parameters
    ----------
    y : np.array, pd.Series, pd.DataFrame
        Measurement/oberserved data.
    y_hat : np.array, pd.Series, pd.DataFrame
        Predicted/forecast/simulated data.
    roll : bool, optional
        Calculate a rolling MSE. The default is False.
    window : None, int, optional
        Window size to use for rolling MSE. The default is None.
    min_periods : None, int, optional
        Minimum number of periods to use for rolling MSE. The default is None.

    Raises
    ------
    ValueError
        Raises ValueError if `y_hat` has more than 2 dimensions.

    Returns
    -------
    float
        Mean squared error.

    """
    # broadcast y to 2D, if input is 2D to correctly calculate r^2 col wise
    if y_hat.ndim == 2:
        y = y[:, None] if y.ndim == 1 else y
    elif y_hat.ndim == 1 and y.ndim == 2:
        y_hat = y_hat[:, None]
    elif y_hat.ndim > 2:
        raise ValueError
    if not roll:
        return ((y - y_hat) ** 2).mean(axis=0)
    else:
        return (
            pd.DataFrame((y - y_hat) ** 2)
            .rolling(window=window, min_periods=min_periods, center=True)
            .mean()
        )


def mean_abs_err(y, y_hat, roll=False, window=None, min_periods=None):
    """
    Calculate the mean absolute error (MAE).

    Also called mean absolute deviation (MAD).

    Parameters
    ----------
    y : np.array, pd.Series, pd.DataFrame
        Measurement/oberserved data.
    y_hat : np.array, pd.Series, pd.DataFrame
        Predicted/forecast/simulated data.
    roll : bool, optional
        Calculate a rolling MSE. The default is False.
    window : None, int, optional
        Window size to use for rolling MSE. The default is None.
    min_periods : None, int, optional
        Minimum number of periods to use for rolling MSE. The default is None.

    Raises
    ------
    ValueError
        Raises ValueError if `y_hat` has more than 2 dimensions.

    Returns
    -------
    float
        Mean absolute error.

    """
    # broadcast y to 2D, if input is 2D to correctly calculate r^2 col wise
    if y_hat.ndim == 2:
        y = y[:, None] if y.ndim == 1 else y
    elif y_hat.ndim == 1 and y.ndim == 2:
        y_hat = y_hat[:, None]
    elif y_hat.ndim > 2:
        raise ValueError
    if not roll:
        return np.abs(y - y_hat).mean(axis=0)
    else:
        return (
            pd.DataFrame(np.abs(y - y_hat))
            .rolling(window=window, min_periods=min_periods, center=True)
            .mean()
        )


def mean_signed_deviation(y, y_hat, roll=False, window=None, min_periods=None):
    """
    Calculate the mean signed deviation (MSD).

    Also called mean biased error (MBE).

    Parameters
    ----------
    y : np.array, pd.Series, pd.DataFrame
        Measurement/oberserved data.
    y_hat : np.array, pd.Series, pd.DataFrame
        Predicted/forecast/simulated data.
    roll : bool, optional
        Calculate a rolling MSE. The default is False.
    window : None, int, optional
        Window size to use for rolling MSE. The default is None.
    min_periods : None, int, optional
        Minimum number of periods to use for rolling MSE. The default is None.

    Raises
    ------
    ValueError
        Raises ValueError if `y_hat` has more than 2 dimensions.

    Returns
    -------
    float
        Mean signed deviation.

    """
    # broadcast y to 2D, if input is 2D to correctly calculate r^2 col wise
    if y_hat.ndim == 2:
        y = y[:, None] if y.ndim == 1 else y
    elif y_hat.ndim == 1 and y.ndim == 2:
        y_hat = y_hat[:, None]
    elif y_hat.ndim > 2:
        raise ValueError
    if not roll:
        return (y_hat - y).mean(axis=0)
    else:
        return (
            pd.DataFrame(y_hat - y)
            .rolling(window=window, min_periods=min_periods, center=True)
            .mean()
        )


def rmse(y, y_hat, roll=False, window=None, min_periods=None):
    """
    Calculate the root mean squared error (RMSE).

    Parameters
    ----------
    y : np.array, pd.Series, pd.DataFrame
        Measurement/oberserved data.
    y_hat : np.array, pd.Series, pd.DataFrame
        Predicted/forecast/simulated data.
    roll : bool, optional
        Calculate a rolling MSE. The default is False.
    window : None, int, optional
        Window size to use for rolling MSE. The default is None.
    min_periods : None, int, optional
        Minimum number of periods to use for rolling MSE. The default is None.

    Returns
    -------
    float
        Root mean squared error.

    """
    return np.sqrt(
        mean_squared_err(
            y, y_hat, roll=roll, window=window, min_periods=min_periods
        )
    )


def cv_rmse(y, y_hat, roll=False, window=None, min_periods=None):
    """
    Calculate the coefficient of variation of the RMSE (CV(RMSE)).

    Parameters
    ----------
    y : np.array, pd.Series, pd.DataFrame
        Measurement/oberserved data.
    y_hat : np.array, pd.Series, pd.DataFrame
        Predicted/forecast/simulated data.
    roll : bool, optional
        Calculate a rolling MSE. The default is False.
    window : None, int, optional
        Window size to use for rolling MSE. The default is None.
    min_periods : None, int, optional
        Minimum number of periods to use for rolling MSE. The default is None.

    Returns
    -------
    float
        Mean squared error.

    """
    return rmse(
        y, y_hat, roll=roll, window=window, min_periods=min_periods
    ) / np.mean(y)


def normalized_err(y, y_hat, err_method='MSE', norm='IQR'):
    """
    Calculate normalized error.

    Any of MSE, MAE, MSD or RMSE can be normalized.

    Normalization can be calculated by IQR, range, mean or median.

    Parameters
    ----------
    y : np.array, pd.Series, pd.DataFrame
        Measurement/oberserved data.
    y_hat : np.array, pd.Series, pd.DataFrame
        Predicted/forecast/simulated data.
    err_method : str, optional
        Error measure to normalize. The default is 'MSE'.
    norm : str, optional
        Normalization range to use. The default is 'IQR'.

    Returns
    -------
    float
        Normalized error measure.

    """
    assert err_method in ('MSE', 'RMSE', 'MSD', 'MAE')
    assert norm in ('IQR', 'range', 'mean', 'median')

    err_methods = {  # supported error methods:
        'MSE': mean_squared_err,
        'RMSE': rmse,
        'MSD': mean_signed_deviation,
        'MAE': mean_abs_err,
    }
    err_fun = err_methods[err_method]

    if norm == 'mean':
        return err_fun(y, y_hat) / np.mean(y)
    elif norm == 'median':
        return err_fun(y, y_hat) / np.median(y)
    elif norm == 'IQR':
        q3, q1 = np.percentile(y, [75, 25])
        return err_fun(y, y_hat) / (q3 - q1)
    elif norm == 'range':
        desc = pd.Series(y).describe()
        return err_fun(y, y_hat) / (desc['max'] - desc['min'])
