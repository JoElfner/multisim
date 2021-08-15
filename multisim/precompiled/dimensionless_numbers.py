# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Jan 2021

Precompiled functions for dimensionless numbers, such as Reynolds, Rayleigh,
Peclet, Nusselt, etc...
"""

import numba as nb
# from numba import jit, njit, float64, int32
import numpy as np

nb.NUMBA_DISABLE_JIT = 0
GLOB_NOGIL = True
GLOB_PARALLEL = True



@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def get_Re_water(v, L, ny, Re):
    """
    Calculate Reynolds number. Result is written to referenced variable Re.

    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    L : TYPE
        DESCRIPTION.
    ny : TYPE
        DESCRIPTION.
    Re : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    Re[:] = np.abs(v) * L / ny


@nb.njit(nogil=GLOB_NOGIL, cache=True)
def Re_water_return(v, L, ny):
    """
    Calculate Reynolds number. Result is returned.

    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    L : TYPE
        DESCRIPTION.
    ny : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.abs(v) * L / ny


@nb.njit(nogil=GLOB_NOGIL, cache=True)
def rayleigh_number(T_s, T_inf, Pr, ny, Kelvin, flow_length):
    """
    Calculate the Rayleigh number for the given parameters [1]_.

    Parameters:
    -----------
    T_s : float, int, np.ndarray
        Surface temperature in [°C] or [K].
    T_inf : float, int, np.ndarray
        Surrounding fluid temperature in [°C] or [K].
    Pr : float, int, np.ndarray
        Prandtl number of the surrounding fluid at the mean temperature:
        $$(T_s + T_{inf}) / 2$$
        For (dry) air this can be set to a constant value of ca.:
        $$Pr = 0.708$$
    ny : float, int, np.ndarray
        Kinematic viscosity in [m^2 / s] of the surrounding fluid at the mean
        temperature: $$(T_s + T_{inf}) / 2$$
    Kelvin : float, int
        If temperatures `T_s` and `T_inf` are given in [°C], Kelvin has to be
        set to `Kelvin=273.15`. If `T_s` and `T_inf` are given in [K], Kelvin
        has to be set to `Kelvin=0`
    flow_length : float, int
        Specific flow length in [m]. Has to be calculated depending on the part
        geometry. See function calc_flow_length() for further information.

    Notes:
    ------
    .. [1] VDI Wärmeatlas 2013, VDI-Gesellschaft Verfahrenstechnik und
       Chemieingenieurwesen, Düsseldorf, Deutschland, p. 754
    """
    # Rayleigh number according to VDI Wärmeatlas 2013 chapter F1
    # eq (7), replacing kappa with kappa = ny/Pr (F1 eq (8)) and beta
    # with 1/T_inf (F1 eq (2)):
    return (
        np.abs(T_s - T_inf)
        * 9.81
        * flow_length ** 3
        * Pr
        / ((T_inf + Kelvin) * ny ** 2)
    )