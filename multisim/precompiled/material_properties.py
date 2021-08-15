# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Jan 2021

Polynomials for material properties.
Most underlying data for material properties is taken from VDI Wärmeatlas.
Polynomials were calculated using scipy and scikit-learn.
"""

import numba as nb

# from numba import jit, njit, float64, int32
import numpy as np

nb.NUMBA_DISABLE_JIT = 0
GLOB_NOGIL = True
GLOB_PARALLEL = True


# ---> water
# calc density from celsius temperature:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def get_rho_water(T, rho):
    # 4th degree
    # Tclp = T.copy()
    # Tclp[Tclp > 100.] = 100.
    rho[:] = (
        999.88785713136213
        + 4.9604454990529602e-02 * T
        - 7.4722666453460717e-03 * T ** 2
        + 4.1094484438154484e-05 * T ** 3
        - 1.2915789546323614e-07 * T ** 4
    )
    # 3rd degree
    # rho[:] = (1000.0614995891804 + 1.3246507417626112e-02*T
    #           - 5.8171082149854319e-03*T**2 + 1.5262905345518088e-05*T**3)


# calc density from celsius temperature AND RETURN the result:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def rho_water(T):
    # 4th degree
    # Tclp = T.copy()
    # Tclp[Tclp > 100.] = 100.
    return (
        999.88785713136213
        + 4.9604454990529602e-02 * T
        - 7.4722666453460717e-03 * T ** 2
        + 4.1094484438154484e-05 * T ** 3
        - 1.2915789546323614e-07 * T ** 4
    )


# calc heat conduction from celsius temperature AND RETURN IT:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def lambda_water(T):
    # 3rd degree (4th degree not sufficiently better)
    # Tclp = T.copy()
    # Tclp[Tclp > 100.] = 100.
    return (
        5.6987912853229539e-01
        + 1.7878370402545738e-03 * T
        - 5.9998217273879795e-06 * T ** 2
        - 8.6964577115093407e-09 * T ** 3
    )


# calc heat conduction from celsius temperature:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def get_lambda_water(T, lam):
    # 3rd degree (4th degree not sufficiently better)
    # Tclp = T.copy()
    # Tclp[Tclp > 100.] = 100.
    lam[:] = (
        5.6987912853229539e-01
        + 1.7878370402545738e-03 * T
        - 5.9998217273879795e-06 * T ** 2
        - 8.6964577115093407e-09 * T ** 3
    )


# calc specific heat capacity from celsius temperature (4th degree, about 10%
# slower but a tiny bit more accurate):
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def get_cp_water(T, cp):
    # 4th degree
    # Tclp = T.copy()
    # Tclp[Tclp > 100.] = 100.
    cp[:] = (
        4215.4023574179992
        - 2.8853943283519348 * T
        + 7.490580684801168e-02 * T ** 2
        - 7.7807143441700321e-04 * T ** 3
        + 3.2109328970410432e-06 * T ** 4
    )
    # 3rd degree
    # cp[:] = (4211.0855150125581 - 1.9815167178349438*T
    #          + 3.375770177242976e-02*T**2 - 1.3588485500876595e-04*T**3)


# calc specific heat capacity from celsius temperature AND RETURN IT
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def cp_water(T):
    # 4th degree
    # Tclp = T.copy()
    # Tclp[Tclp > 100.] = 100.
    return (
        4215.4023574179992
        - 2.8853943283519348 * T
        + 7.490580684801168e-02 * T ** 2
        - 7.7807143441700321e-04 * T ** 3
        + 3.2109328970410432e-06 * T ** 4
    )


# calc kinematic viscosity from celsius temperature after VDI Wärmeatlas 2013
# table D2.1:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def get_ny_water(T, ny):
    # 4th degree:
    # Tclp = T.copy()
    # Tclp[Tclp > 100.] = 100.
    ny[:] = (
        1.7764473380494155e-06
        - 5.5640275781265404e-08 * T
        + 1.0243072887494426e-09 * T ** 2
        - 9.7954460136981165e-12 * T ** 3
        + 3.6460468745062724e-14 * T ** 4
    )


# calc kinematic viscosity from celsius temperature AND RETURN IT, VDI
# Wärmeatlas 2013 table D2.1:
@nb.njit(nogil=GLOB_NOGIL, cache=True)
def ny_water(T):
    # 4th degree:
    # Tclp = T.copy()
    # Tclp[Tclp > 100.] = 100.
    return (
        1.7764473380494155e-06
        - 5.5640275781265404e-08 * T
        + 1.0243072887494426e-09 * T ** 2
        - 9.7954460136981165e-12 * T ** 3
        + 3.6460468745062724e-14 * T ** 4
    )


# calc Prandtl number from celsius temperature:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def get_Pr_water(T, Pr):
    # 4th degree:
    # Pr[:] = (12.909891117064289 - 0.4207372206483363*T
    #          + 7.4860282126284405e-03*T**2 - 6.854571430021334e-05*T**3
    #          + 2.4685760188512201e-07*T**4)
    # 3rd degree:
    # Tclp = T.copy()
    # Tclp[Tclp > 100.] = 100.
    Pr[:] = (
        12.5780108199379058
        - 0.35124680571767508 * T
        + 4.3225480444706085e-03 * T ** 2
        - 1.9174193923188898e-05 * T ** 3
    )


# calc Prandtl number from celsius temperature AND RETURN IT
# (alot faster for single values):
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def Pr_water_return(T):
    # 4th degree:
    # Pr[:] = (12.909891117064289 - 0.4207372206483363*T
    #          + 7.4860282126284405e-03*T**2 - 6.854571430021334e-05*T**3
    #          + 2.4685760188512201e-07*T**4)
    # 3rd degree:
    # Tclp = T.copy()
    # Tclp[Tclp > 100.] = 100.
    return (
        12.5780108199379058
        - 0.35124680571767508 * T
        + 4.3225480444706085e-03 * T ** 2
        - 1.9174193923188898e-05 * T ** 3
    )


# calc isobaric expansion coefficient in [1/K] from celsius temperature:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def get_beta_water(T, beta):
    # 3rd degree:
    # Tclp = T.copy()
    # Tclp[Tclp > 100.] = 100.
    beta[:] = (
        -5.87985364766666e-05
        + 1.5641955219950547e-05 * T
        - 1.3587684743777981e-07 * T ** 2
        + 6.1220503308149086e-10 * T ** 3
    )


# calc isobaric expansion coefficient in [1/K] from celsius temperature
# AND RETURN IT:
@nb.njit(nogil=GLOB_NOGIL, cache=True)
def beta_water_return(T):
    # 3rd degree:
    # Tclp = T.copy()
    # Tclp[Tclp > 100.] = 100.
    return (
        -5.87985364766666e-05
        + 1.5641955219950547e-05 * T
        - 1.3587684743777981e-07 * T ** 2
        + 6.1220503308149086e-10 * T ** 3
    )


# ---> dry air:
# calc density from celsius temperature:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def get_rho_dryair(T, rho):
    # 2nd degree
    rho[:] = (
        1.2767987012987012
        - 0.0046968614718614701 * T
        + 1.4296536796536256e-05 * T ** 2
    )


# calc heat conductivity from celsius temperature:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def get_lam_dryair(T, lam):
    # 2nd degree
    lam[:] = (
        0.024358670995670989
        + 7.6533982683982561e-05 * T
        - 4.2099567099572201e-08 * T ** 2
    )


# calc heat conductivity from celsius temperature and return it:
@nb.njit(nogil=GLOB_NOGIL, cache=True)
def lam_dryair_return(T):
    # 2nd degree
    return (
        0.024358670995670989
        + 7.6533982683982561e-05 * T
        - 4.2099567099572201e-08 * T ** 2
    )


# calc kinematic viscosity from celsius temperature:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def get_ny_dryair(T, ny):
    # 2nd degree
    ny[:] = (
        1.3500069264069257e-05
        + 8.8810389610389459e-08 * T
        + 1.0974025974025443e-10 * T ** 2
    )


# calc kinematic viscosity from celsius temperature and return it:
@nb.njit(nogil=GLOB_NOGIL, cache=True)
def ny_dryair_return(T):
    # 2nd degree
    return (
        1.3500069264069257e-05
        + 8.8810389610389459e-08 * T
        + 1.0974025974025443e-10 * T ** 2
    )


# ---> humid air:
# saturation pressure in [Pa] of humid air for total pressures < 2MPa
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def humid_air_saturation_pressure(T):
    # 6th degree
    return (
        +1.56927617e-09 * T ** 6
        + 2.32760367e-06 * T ** 5
        + 3.19028425e-04 * T ** 4
        + 2.51824584e-02 * T ** 3
        + 1.42489091e00 * T ** 2
        + 4.55277840e01 * T ** 1
        + 5.99770272e02
    )
    # 10th degree


#    return (- 1.30339138e-16*T**10 + 7.49527386e-14*T**9 - 1.59881730e-11*T**8
#            + 1.54764869e-09*T**7 - 5.56609536e-08*T**6 + 1.46597641e-06*T**5
#            + 4.21883898e-04*T**4 + 2.43290034e-02*T**3 + 1.38204573e+00*T**2
#            + 4.58581434e+01*T + 6.02909924e+02)


# mass of water in fully saturated air in [kg H2O / kg Air] for a pressure of
# 0.1 MPa, only valid for -30 <= T <= 80 !
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def humid_air_sat_water_mass(T):
    r"""
    Calculate the mass of water in fully saturated air (at 100% relative
    humidity) in :math:`[f]= \mathtt{kg_{H_2O}}/\mathtt{kg_{Luft}}`,
    valid for a pressure of :math:`0.1\mathtt{\,MPa}` and a temperature range
    of :math:`-30\mathtt{\,°C}\leq T \leq 80\mathtt{\,°C}`.

    """

    #    assert np.all(-30 <= T) and np.all(T <= 80)
    # 6th degree
    #    return (1.56927617e-09*T**6 + 2.32760367e-06*T**5 + 3.19028425e-04*T**4
    #            + 2.51824584e-02*T**3 + 1.42489091e+00*T**2 + 4.55277840e+01*T
    #            + 5.99770272e+02)
    # 10th degree
    return (
        +3.47491188e-19 * T ** 10
        - 6.50956001e-17 * T ** 9
        + 3.68271647e-15 * T ** 8
        + 2.06252891e-14 * T ** 7
        - 7.11474217e-12 * T ** 6
        + 1.29052920e-10 * T ** 5
        + 6.62755505e-09 * T ** 4
        + 8.79652019e-08 * T ** 3
        + 8.16034548e-06 * T ** 2
        + 2.98380899e-04 * T
        + 3.79413965e-03
    )
