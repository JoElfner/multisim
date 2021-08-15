# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Oct 2017
"""

import numba as nb
from numba import jit, njit, float64, int32
import numpy as np

nb.NUMBA_DISABLE_JIT = 0
GLOB_NOGIL = True
GLOB_PARALLEL = True


# %% Simulation environment processing
# f.i. port and part flow processing, material properties etc..
# %%% Simulation Env. port updating
@jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def upd_p_arr(ports_all, port_ids, values, _port_own_idx):
    """
    Updates the array which stores the values of all ports. This only updates
    the port values of a single part per call!
    """
    # get values from part result arrays at ports and pass them to the model
    # environment ports array (using flattened array is about 6% slower than
    # direct indexing, but allows 1D and 2D indexing):
    for i in range(_port_own_idx.shape[0]):
        ports_all[port_ids[i]] = values.flat[_port_own_idx[i]]


@njit(nogil=GLOB_NOGIL, cache=True)
def _upddate_ports_interm(ports_all, trgt_indices, ports_src, source=0):
    """
    This function updates the array which stores the port values of all parts
    with the intermediate result values of the current step stored in
    `ports_src`. If more than one intermediate step is calculated during the
    solver run, these can be update by passing the number of the intermediate
    result to `source=X`, where X is an integer value starting with 0 for the
    first intermediate step.
    """
    values = ports_src[source]
    i = 0
    for val in values:
        ports_all[trgt_indices[i]] = val[0]
        i += 1


@njit(nogil=GLOB_NOGIL, cache=True)
def _upddate_ports_result(
    ports_all, trgt_indices, ports_src, stepnum, src_list
):
    """
    This function updates the array which stores the port values of all parts
    with the final result values of the current step stored in `ports_src`.
    """
    i = 0
    for val in ports_src:
        ports_all[trgt_indices[i]] = val[
            (stepnum * src_list[i][0]) + src_list[i][1]
        ]
        i += 1


@njit(nogil=GLOB_NOGIL, cache=True)  # parallel=GLOB_PARALLEL useful
def _port_values_to_idx(ports_all, port_link_idx, port_own_idx, out):
    """
    Values of requested ports are saved to a non-contiguous array (port values
    are only stored at specific locations).
    """

    for i in range(port_link_idx.size):
        out.flat[port_own_idx[i]] = ports_all[port_link_idx[i]]


@nb.njit(nogil=GLOB_NOGIL, cache=True)  # parallel=GLOB_PARALLEL useful
def _port_values_to_cont(ports_all, port_link_idx, out):
    """
    Values of requested ports are saved to a contiguous array.
    """

    for i in range(port_link_idx.size):
        out.flat[i] = ports_all[port_link_idx[i]]


# %%% Simulation Env. in-part flow processing:
@njit(nogil=GLOB_NOGIL, cache=True)  # parallel=GLOB_PARALLEL useful
def _process_flow_invar(
    process_flows, dm_io, dm_top, dm_bot, dm_port, stepnum, res_dm
):
    """
    Process massflows.

    Massflows are being processed for parts where the massflow is defined as
    invariant.
    """
    if process_flows[0]:
        if dm_io[0] >= 0.0:  # massflow from the top
            # get massflow though top cell-cell border:
            dm_top[1:] = dm_io[0]
            dm_bot[:-1] = 0.0  # set massflow from the bottom to zero
        else:  # massflow from the bottom:
            # get massflow though bottom cell-cell border (and *-1!):
            dm_bot[:-1] = -dm_io[0]
            dm_top[1:] = 0.0  # set massflow from the top to zero

        # get ports massflow (only for the positive inflow):
        if dm_io[0] >= 0.0:
            dm_port[0] = dm_io[0]
            dm_port[-1] = 0.0
        else:
            dm_port[-1] = -dm_io[0]
            dm_port[0] = 0.0

        res_dm[stepnum[0]] = dm_io[0]

    # return process flows bool to disable processing flows until next step
    return False


@njit(nogil=GLOB_NOGIL, cache=True)  # parallel=GLOB_PARALLEL useful
def _process_flow_invar_fast(
    process_flows, dm_io, dm_top, dm_bot, stepnum, res_dm
):
    """
    Process massflows for parts with invariant flows.

    Massflows are being processed for parts where the massflow is defined as
    invariant.
    """
    if process_flows[0]:
        # preallocate arrays which need assignment:
        dm_port = np.empty(2)

        if dm_io[0] >= 0.0:  # massflow from the top
            # get massflow though top cell-cell border:
            dm_top[1:] = dm_io[0]
            dm_bot[:-1] = 0.0  # set massflow from the bottom to zero
        else:  # massflow from the bottom:
            # get massflow though bottom cell-cell border (and *-1!):
            dm_bot[:-1] = -dm_io[0]
            dm_top[1:] = 0.0  # set massflow from the top to zero

        # get ports massflow (only for the positive inflow):
        if dm_io[0] >= 0.0:
            dm_port[0] = dm_io[0]
            dm_port[-1] = 0.0
        else:
            dm_port[-1] = -dm_io[0]
            dm_port[0] = 0.0

        res_dm[stepnum[0]] = dm_io[0]

    # return process flows bool to disable processing flows until next step
    return False, dm_port


@njit(nogil=GLOB_NOGIL, cache=True)  # parallel=GLOB_PARALLEL useful
def _process_flow_var(
    process_flows,
    dm_io,
    dm,
    dm_top,
    dm_bot,
    dm_port,
    port_own_idx,
    stepnum,
    res_dm,
):
    """
    Process massflows for parts with variant flows.

    Massflows are being processed for parts where the massflow is defined as
    variant.
    """
    if process_flows[0]:
        # massflow through ports is aquired by update_FlowNet
        # get massflow through each cell (sum up in/out dm of ports and then
        # run cumulative sum over all cells)
        # copy I/O flows to NOT alter the I/O array during calculations:

        # this is the procedure for collapsed and flattened dm_io.
        dm[:] = 0.0
        cs = np.cumsum(dm_io)
        for i in range(port_own_idx.size - 1):
            dm[port_own_idx[i] : port_own_idx[i + 1]] += cs[i]
        # get port values
        dm_port[:] = dm_io
        # get massflow though top cell-cell border:
        dm_top[1:] = dm[:-1]
        # get massflow though bottom cell-cell border (and *-1!):
        dm_bot[:-1] = -dm[:-1]
        # remove negative values:
        dm_top[dm_top < 0] = 0.0
        dm_bot[dm_bot < 0] = 0.0
        dp = dm_port.ravel()  # flattened view to ports for 2D indexing
        dp[dp < 0.0] = 0.0
        # set last value of dm to be the same as the value of the previous cell
        # to avoid having 0-massflow in it due to cumsum:
        dm[-1] = dm[-2]

        res_dm[stepnum[0]] = dm

    # return process flows bool to disable processing flows until next step
    return False


@njit(nogil=GLOB_NOGIL, cache=True)  # parallel=GLOB_PARALLEL useful
def _process_flow_multi_flow(
    process_flows, dm_io, dm_top, dm_bot, dm_port, stepnum, res_dm
):
    """
    Process masssflows for parts with multiple flow channels.

    Massflows are being processed for parts which have multiple separated flow
    channels. The massflow in each flow channel must be invariant.
    The massflow through ports in `dm_io` is aquired by update_FlowNet.
    """
    # if flows were not yet processed in this timestep
    if process_flows[0]:
        # loop over channels and get each channel's massflow
        for i in range(dm_io.size):
            if dm_io[i] >= 0.0:  # massflow from in port (top)
                # get massflow though top cell-cell border:
                dm_top[1:, i] = dm_io[i]
                dm_bot[:-1, i] = 0.0  # set massflow from the bottom to zero
                # set port massflows. dm_port has 2 cells per flow channel,
                # first is in, second is out. Thus if flow from in port, set
                # flow to in and out to 0.
                dm_port[i * 2] = dm_io[i]
                dm_port[i * 2 + 1] = 0.0
            else:  # massflow from out port (bottom):
                # get massflow though bottom cell-cell border (and *-1!):
                dm_bot[:-1, i] = -dm_io[i]  # -1 makes this a pos. massflow!
                dm_top[1:, i] = 0.0  # set massflow from the top to zero
                # set port massflows. dm_port has 2 cells per flow channel,
                # first is in, second is out. Thus if flow from out port, set
                # flow to out (make it positive!) and in to 0.
                dm_port[i * 2] = 0.0
                dm_port[i * 2 + 1] = -dm_io[i]  # dm port is ALWAYS positive!
        # set current steps flow to result
        res_dm[stepnum[0]] = dm_io

    # return process flows bool to disable processing flows until next step
    return False


# %%% Simulation Env. in-part port temperatures processing:
@nb.njit(cache=True, nogil=GLOB_NOGIL)
def _process_ports_collapsed(
    ports_all,
    port_link_idx,
    port_own_idx,
    T,
    mcp,
    UA_port,
    UA_port_wll,
    A_p_fld_mean,
    port_gsp,
    grid_spacing,
    lam_T,
    cp_port,
    lam_port_fld,
    T_port,
):
    """
    Values of requested ports are saved to results array. Only works for parts
    which use collapsed port arrays.
    """

    dT_cond_port = np.zeros(port_own_idx.shape)

    for i in range(port_link_idx.size):
        p_val = ports_all[port_link_idx[i]]
        idx = port_own_idx[i]

        # collapsed arrays only take index i:
        T_port.flat[i] = p_val
        cp_port.flat[i] = cp_water(p_val)
        lam_port_fld.flat[i] = lambda_water(p_val)

        #        lam_fld_own_p[i] =
        # get total port heat conduction:
        UA_port.flat[i] = (
            A_p_fld_mean[i]
            / (
                +(port_gsp[i] / (2 * lam_port_fld[i]))
                + (grid_spacing / (2 * lam_T.flat[idx]))
            )
            + UA_port_wll[i]
        )

        dT_cond_port.flat[i] = (
            UA_port.flat[i] * (p_val - T.flat[idx]) / mcp.flat[idx]
        )

    return dT_cond_port


@njit(nogil=GLOB_NOGIL, cache=True)  # parallel=GLOB_PARALLEL useful
def _process_ports(
    ports_all,
    port_link_idx,
    port_own_idx,
    T,
    mcp,
    UA_port,
    UA_port_wll,
    A_p_fld_mean,
    port_gsp,
    grid_spacing,
    lam_T,
    cp_port,
    lam_port_fld,
    T_port,
):
    """
    Values of requested ports are saved to results array.
    """

    dT_cond_port = np.zeros(port_own_idx.shape)

    for i in range(port_link_idx.size):
        p_val = ports_all[port_link_idx[i]]
        idx = port_own_idx[i]

        T_port.flat[idx] = p_val
        cp_port.flat[idx] = cp_water(p_val)
        # collapsed arrays only take index i:
        lam_port_fld.flat[idx] = lambda_water(p_val)

        #        lam_fld_own_p[i] =
        # get total port heat conduction:
        UA_port.flat[idx] = (
            A_p_fld_mean.flat[idx]
            / (
                +(port_gsp.flat[idx] / (2 * lam_port_fld.flat[idx]))
                + (grid_spacing / (2 * lam_T.flat[idx]))
            )
            + UA_port_wll.flat[idx]
        )

        dT_cond_port.flat[i] = UA_port.flat[idx] * (p_val - T[idx]) / mcp[idx]

    return dT_cond_port


# %%% Simulation Env. in-part material properties processing:
@njit(nogil=GLOB_NOGIL, cache=True)
def water_mat_props_ext_view(T_ext, cp_T, lam_T, rho_T, ny_T):
    """
    Get the relevant temperature dependent material properties of water for
    parts which use the extended array format:
        cp: Specific heat capacity in [J/(kg K)]
        lam: Heat conductivity in [W/(m K)]
        rho: Density in [kg/m^3]
        ny: Kinematic viscosity in [Pa/m^2]
    """

    get_cp_water(T_ext, cp_T)  # extended array for top/bot views in adv.
    get_lambda_water(T_ext[1:-1], lam_T)  # non-ext. array for other mat.
    get_rho_water(T_ext[1:-1], rho_T)  # props. since no views needed here
    get_ny_water(T_ext[1:-1], ny_T)


@njit(nogil=GLOB_NOGIL, cache=True)
def water_mat_props_ext(T_ext):
    """
    Get the relevant temperature dependent material properties of water for
    parts which use the extended array format:
        cp: Specific heat capacity in [J/(kg K)]
        lam: Heat conductivity in [W/(m K)]
        rho: Density in [kg/m^3]
        ny: Kinematic viscosity in [Pa/m^2]
    """
    # cp: extended array for top/bot views in adv.
    # non-ext. array for other mat.  props. since no views needed here
    return (
        cp_water(T_ext),
        lambda_water(T_ext[1:-1]),
        rho_water(T_ext[1:-1]),
        ny_water(T_ext[1:-1]),
    )


@njit(nogil=GLOB_NOGIL, cache=True)
def water_mat_props(T, cp_T, lam_T, rho_T, ny_T):
    """
    Get the relevant temperature dependent material properties of water:
        cp: Specific heat capacity in [J/(kg K)]
        lam: Heat conductivity in [W/(m K)]
        rho: Density in [kg/m^3]
        ny: Kinematic viscosity in [Pa/m^2]
    """

    get_cp_water(T, cp_T)
    get_lambda_water(T, lam_T)  # non-ext. array for mat.
    get_rho_water(T, rho_T)  # props. since no views needed here
    get_ny_water(T, ny_T)


@njit(nogil=GLOB_NOGIL, cache=True)
def cell_temp_props_ext(T_ext, V_cell, cp_T, rho_T, mcp_wll, rhocp, mcp, ui):
    """
    Calculate the each cells specific temperature dependent properties for
    parts which use the extended array format:
        rho*cp: Volume specific heat capacity in [J / (K m^3)]
        m*cp: heat capacity (of fluid AND wall) in [J / K]
        u_i: mass specific inner energy in [J / kg]
    """
    # precalculate values which are needed multiple times:
    # volume specific heat capacity:
    rhocp[:] = rho_T * cp_T[1:-1]
    # heat capacity of fluid AND wall:
    mcp[:] = V_cell * rhocp + mcp_wll
    # mass specific inner energy:
    ui[:] = cp_T[1:-1] * T_ext[1:-1]


@njit(nogil=GLOB_NOGIL, cache=True)
def cell_temp_props_fld(
    T_ext_fld, V_cell, cp_T, rho_T, rhocp_fld, mcp_fld, ui_fld
):
    """
    Calculate the each cells fluid specific temperature dependent properties
    for parts which use the extended array format:
        rho*cp: Volume specific heat capacity in [J / (K m^3)]
        m*cp: heat capacity (of fluid) in [J / K]
        u_i: mass specific inner energy in [J / kg]
    """
    # precalculate values which are needed multiple times:
    # volume specific heat capacity:
    rhocp_fld[:] = rho_T * cp_T[1:-1]
    # heat capacity of fluid AND wall:
    mcp_fld[:] = V_cell * rhocp_fld
    # mass specific inner energy:
    ui_fld[:] = cp_T[1:-1] * T_ext_fld[1:-1]


@njit(nogil=GLOB_NOGIL, cache=True)
def specific_inner_energy_wll(T_wll, cp_wll, ui):
    """
    Calculate the each cells specific temperature dependent properties for
    parts which use the extended array format:
        rho*cp: Volume specific heat capacity in [J / (K m^3)]
        m*cp: heat capacity (of fluid) in [J / K]
        u_i: mass specific inner energy in [J / kg]
    """
    # mass specific inner energy:
    ui[:] = cp_wll * T_wll


@njit(nogil=GLOB_NOGIL, cache=True)
def cell_temp_props(T, V_cell, cp_T, rho_T, mcp_wll, rhocp, mcp, ui):
    """
    Calculate the each cells specific temperature dependent properties:
        rho*cp: Volume specific heat capacity in [J / (K m^3)]
        m*cp: heat capacity (of fluid AND wall) in [J / K]
        u_i: mass specific inner energy in [J / kg]
    """
    # precalculate values which are needed multiple times:
    # volume specific heat capacity:
    rhocp[:] = rho_T * cp_T
    # heat capacity of fluid AND wall:
    mcp[:] = V_cell * rhocp + mcp_wll
    # mass specific inner energy:
    ui[:] = cp_T * T


@njit(nogil=GLOB_NOGIL, cache=True)
def _lambda_mean_view(lam_T, out):
    """
    Get mean lambda of two neighbouring cells for the first axis of an
    n-dimensional grid.
    This is **NOT** the arithmetic mean, since the mean value of two heat
    conductivities in series circuit is calculated by adding the inverse of
    the heat conductivities.
    For example for two heat conductivites `lam_1=40` and `lam_2=80`, each over
    a length of `L=0.2`, the mean value is:
        eq:: $$lam_{mean} = 2*L / (L/lam_1 + L/lam_2) = 2 / (1/lam_1 + 1/lam_2)$$
        where the second right hand side of the equation is only true for
        equally spaced grids.
    """
    out[:] = 2 * lam_T[:-1] * lam_T[1:] / (lam_T[:-1] + lam_T[1:])


@njit(nogil=GLOB_NOGIL, cache=True)
def _lambda_mean(lam_T):
    """
    Get mean lambda of two neighbouring cells for the first axis of an
    n-dimensional grid.
    This is **NOT** the arithmetic mean, since the mean value of two heat
    conductivities in series circuit is calculated by adding the inverse of
    the heat conductivities.
    For example for two heat conductivites `lam_1=40` and `lam_2=80`, each over
    a length of `L=0.2`, the mean value is:
        eq:: $$lam_{mean} = 2*L / (L/lam_1 + L/lam_2) = 2 / (1/lam_1 + 1/lam_2)$$
        where the second right hand side of the equation is only true for
        equally spaced grids.
    """
    return 2 * lam_T[:-1] * lam_T[1:] / (lam_T[:-1] + lam_T[1:])


# %% U*A values calculation:
@njit(nogil=GLOB_NOGIL, cache=True)
def UA_plate_tb(A_cell, grid_spacing, lam_mean, UA_tb_wll, out):
    """
    Get the UA-value for plate-like geometry, for example in a pipe or TES,
    between neighboring cells.
    """

    # get UA value between cells. UA value of walls added (parallel circuit).
    # UA is extended array to enable using views for calculation:
    out[1:-1] = A_cell / grid_spacing * lam_mean + UA_tb_wll


@njit(nogil=GLOB_NOGIL, cache=True)
def UA_plate_tb_fld(A_cell, grid_spacing, lam_mean, out):
    """
    Get the UA-value for plate-like geometry, for example in a pipe or TES,
    between neighboring cells.
    """

    # get UA value between cells. UA value of walls added (parallel circuit).
    # UA is extended array to enable using views for calculation:
    out[1:-1] = A_cell / grid_spacing * lam_mean


@njit(nogil=GLOB_NOGIL, cache=True)
def UA_plate_tb_wll(UA_tb_wll, out):
    """
    Get the UA-value for plate-like geometry, for example in a pipe or TES,
    between neighboring cells.
    """

    # get UA value between cells. UA value of walls added (parallel circuit).
    # UA is extended array to enable using views for calculation:
    out[1:-1] = UA_tb_wll


@nb.njit(nogil=GLOB_NOGIL, cache=True)
def buoyancy_byNusselt(T, ny, d_i, lam_mean):
    """
    Calculate the buoyancy driven heat flow by Nusselt approximation.

    Calculate the buoyancy driven heat flow inside a vertically stratified
    thermal energy storage tank by using Nusselt approximations to calculate a
    correction factor for the heat conductivity.
    """
    # get temperature difference for all cells (temperature below last cell
    # is 0, thus don't use the last cell):
    #    T_diff = T_bot[:-1] - T[:-1]  # replaced with stencil operation below:
    T_diff = T[1:] - T[:-1]
    # if there is no temperature inversion, skip this function:
    if np.all(T_diff <= 0):
        return
    # only use the positive difference (inverted cells):
    T_diff[T_diff < 0] = 0
    # buoyancy correction factor to get the buoyant flow from fluid-fluid
    # instead of a solid-fluid horizontal plate:
    corr_f = 20
    # preallocate arrays:
    shape = T.shape[0] - 1
    Nu = np.zeros(shape)

    # free convection over a horizontal plate, VDI F2 3.1:
    # get material properties for all bottom cells:
    Pr = Pr_water_return(T[1:])
    beta = beta_water_return(T[1:])
    # to deal with the minimum in water density at 4°C, just set negative
    # values to pos.
    beta[beta < 0] *= -1
    # get characteristic length:
    L = d_i / 4
    # get Rayleigh number
    Ra = Pr * 9.81 * L ** 3 * beta * T_diff / ny[1:] ** 2
    # get Rayleigh number with Prandtl function, VDI F2 eq (9):
    Ra_f2 = Ra * (1 + (0.322 / Pr) ** (11 / 20)) ** (-20 / 11)
    # get bool index for laminar or turbulent convection:
    turb = Ra_f2 > 7e4
    # get Nusselt number, following VDI Wärmeatlas 2013 F2 eq (7) and (8):
    Nu[~turb] = 0.766 * (Ra_f2[~turb]) ** 0.2
    Nu[turb] = 0.15 * (Ra_f2[turb]) ** (1 / 3)
    # get bool index for Nusselt number > 1 to ignore lower values
    Nu_idx = Nu >= 1
    # multiplicate lambda value between cells with the Nusselt number. The
    # calculation of the alpha value is implemented in the calculation of
    # the UA value.
    lam_mean[Nu_idx] *= Nu[Nu_idx] * corr_f


@nb.njit(nogil=GLOB_NOGIL, cache=True)
def buoyancy_AixLib(T, cp, rho, ny, grid_spacing, lam_mean):
    """
    Calculate the buoyancy driven heat flow by conductivity plus.

    Calculate the buoyancy driven heat flow inside a vertically stratified
    thermal energy storage tank by using AixLib based epmirical relations for
    an additional heat conductivity [1]_.

    Sources:
        [1] : https://github.com/RWTH-EBC/AixLib/blob/master/AixLib/Fluid/Storage/BaseClasses/Bouyancy.mo
    """
    # get temperature difference for all cells (temperature below last cell
    # is 0, thus don't use the last cell):
    #    T_diff = T_bot[:-1] - T[:-1]  # replaced with stencil operation below:
    T_diff = T[1:] - T[:-1]
    # if there is no temperature inversion, skip this function:
    if np.all(T_diff <= 0):
        return
    # only use the positive difference (inverted cells):
    T_diff[T_diff < 0] = 0
    # kappa is assumed to be constant at 0.4, g at 9.81
    kappa = 0.4
    g = 9.81

    # get material properties for all bottom cells:
    beta = beta_water_return(T[1:])
    # to deal with the minimum in water density at 4°C, just set negative
    # values to pos.
    beta[beta < 0] *= -1

    # calculate lambda surplus due to buoyancy
    lambda_plus = (
        2
        / 3
        * rho
        * cp
        * kappa
        * grid_spacing ** 2
        * np.sqrt(np.abs(-g * beta * T_diff / grid_spacing))
    )

    # add up to lambda mean
    lam_mean += lambda_plus


# %% Simulation Env. in-part von Neumann stability calculation:
@njit(nogil=GLOB_NOGIL, cache=True)
def _vonNeumann_stability_invar(
    part_id,
    stability_breaches,
    UA_tb,
    UA_port,
    UA_amb_shell,
    dm_io,
    rho_T,
    rhocp,
    grid_spacing,
    port_subs_gsp,
    A_cell,
    A_port,
    A_shell,  # areas to backcalc diffusivity from UA
    r_total,
    V_cell,
    step_stable,  # check_vN, ,  # system wide bools
    vN_max_step,
    max_factor,
    stepnum,
    timestep,  # system wide vars
):
    r"""
    Check for L2/von Neumann stability for diffusion and massflows.

    Massflows are checked for parts where the massflow is defined NOT
    invariant, that means where all cells in the part share the same massflow!

    Notes
    -----
    Von Neumann stability for conduction:

    .. math::
        r = \frac{\alpha \Delta t}{(\Delta x)^2} \leq \frac{1}{2} \\
        \text{With the thermal diffusivity: } \alpha = \frac{
        \lambda}{\rho c_{p}}\\
        \text{and } \lambda = \frac{U\cdot A}{A} \cdot \Delta x \\
        \text{yields } r = \frac{(UA)}{\rho c_{p}} \frac{\Delta t}{A \Delta x}
    Von Neumann stability for advection:
    """

    # save von Neumann stability values for cells by multiplying the cells
    # relevant total x-gridspacing with the maximum UA-value (this gives a
    # substitue heat conduction to get a total diffusion coefficient) and
    # the inverse maximum rho*cp value (of all cells! this may result in a
    # worst-case-result with a security factor of up to about 4.2%) to get
    # the substitute diffusion coefficient and then mult. with step and
    # div. by gridspacing**2 (not **2 since this is cut out with mult. by
    # it to get substitute diffusion from UA) and save to array:
    vN_diff = np.empty(3)

    # rhocpmax = rhocp.max()
    # For calculation see docstring
    # replaced old and faulty calculations with missing Area
    # vN_diff[0] = (UA_tb.max() / rhocpmax) * timestep / grid_spacing
    vN_diff[0] = (
        np.max(UA_tb[1:-1] / rhocp[1:]) * timestep / (A_cell * grid_spacing)
    )
    # for the next two with non-constant gridspacing, find max of UA/gsp:
    # vN_diff[1] = (UA_port / port_subs_gsp).max() / rhocpmax * timestep
    vN_diff[1] = (
        np.max(UA_port / (A_port * port_subs_gsp)) * timestep / rhocp.max()
    )
    # vN_diff[2] = UA_amb_shell.max() / r_total / rhocpmax * timestep
    vN_diff[2] = np.max(UA_amb_shell / rhocp) * timestep / (A_shell * r_total)
    # get maximum:
    vN_diff_max = vN_diff.max()

    # for massflow:
    # get maximum cfl number (this is the von Neumann stability condition
    # for massflow through cells), again with total max. of rho to get a
    # small security factor for worst case:
    Vcellrhomax = V_cell * rho_T.max()
    vN_dm_max = np.abs(dm_io).max() * timestep / Vcellrhomax

    # get maximum von Neumann stability condition values:
    # get dividers for maximum stable timestep to increase or decrease
    # stepsize:
    vN_diff_mult = vN_diff_max / 0.5
    vN_dm_mult = vN_dm_max / 1
    # get biggest divider:
    vN_div_max = max(vN_diff_mult, vN_dm_mult)
    # check if any L2 stability conditions are violated:
    if vN_div_max > 1:
        # only do something if von Neumann checking is active, else just
        # print an error but go on with the calculation:
        #        if check_vN:
        if True:
            # if not stable, set stable step bool to False
            step_stable[0] = False
            stability_breaches += 1  # increase breaches counter for this part
            # calculate required timestep to make this part stable with a
            # security factor of 0.95:
            local_vN_max_step = timestep / vN_div_max * 0.95
            # if this is the smallest step of all parts needed to make all
            # parts stable save it to maximum von Neumann step:
            if vN_max_step[0] > local_vN_max_step:
                vN_max_step[0] = local_vN_max_step
            #                # increase error weight of this part by the factor of 1.1 to
            #                # avoid running into this error again:
            #                self._trnc_err_cell_weight *= 1.1  # NOT good
            # adjust max factor if vN was violated:
            if max_factor[0] > 1.05:
                max_factor[0] = max_factor[0] ** 0.99
                if max_factor[0] < 1.05:  # clip to 1.05
                    max_factor[0] = 1.05
        else:
            print(
                '\nVon Neumann stability violated at step',
                stepnum,
                'and part with id',
                part_id,
                '!',
            )
            raise ValueError

    # return new values (or old values if unchanged):
    return step_stable, vN_max_step, max_factor


@njit(nogil=GLOB_NOGIL, cache=True)
def _vonNeumann_stability_invar_hexnum(
    part_id,
    stability_breaches,
    UA_dim1,
    UA_dim2,
    UA_port,
    dm_io,
    rho_T,
    rhocp,
    grid_spacing,
    port_subs_gsp,
    A_channel,
    A_plate_eff,
    A_port,
    V_cell,
    step_stable,  # check_vN, ,  # system wide bools
    vN_max_step,
    max_factor,
    stepnum,
    timestep,  # system wide vars
):
    r"""
    Check for L2/von Neumann stability for diffusion and massflows.

    Special method for numeric Heat Exchanger calculation with two-dimensional
    heat flow and two seeparated flow regimes.

    Notes
    -----
    Von Neumann stability for conduction:

    .. math::
        r = \frac{\alpha \Delta t}{(\Delta x)^2} \leq \frac{1}{2} \\
        \text{With the thermal diffusivity: } \alpha = \frac{
        \lambda}{\rho c_{p}}\\
        \text{and } \lambda = \frac{U\cdot A}{A} \cdot \Delta x \\
        \text{yields } r = \frac{(UA)}{\rho c_{p}} \frac{\Delta t}{A \Delta x}
    Von Neumann stability for advection:
    """

    # save von Neumann stability values for cells by multiplying the cells
    # relevant total x-gridspacing with the maximum UA-value (this gives a
    # substitue heat conduction to get a total diffusion coefficient) and
    # the inverse maximum rho*cp value (of all cells! this may result in a
    # worst-case-result with a security factor of up to about 4.2%) to get
    # the substitute diffusion coefficient and then mult. with step and
    # div. by gridspacing**2 (not **2 since this is cut out with mult. by
    # it to get substitute diffusion from UA) and save to array:
    vN_diff = np.empty(3)

    rhocpmax = rhocp.max()
    # heat conduction in flow direction:
    vN_diff[0] = (  # intermediate solution with Area but not detailed max.
        (UA_dim1.max() / rhocpmax) * timestep / (A_channel * grid_spacing)
    )
    # old version without area
    # vN_diff[0] = (UA_dim1.max() / rhocpmax) * timestep / grid_spacing
    # new version with detailed checks, but not validated yet, thus
    # replaced with the intermediate solution
    # vN_diff[0] = (
    #     np.max(UA_dim1[1:-1] / rhocp[1:])
    #     * timestep / (A_channel * grid_spacing))
    # heat conduction perpendicular to flow direction (fluid-plate-fuild):
    vN_diff[1] = (
        (UA_dim2.max() / rhocpmax) * timestep / (A_plate_eff * grid_spacing)
    )
    # vN_diff[1] = (UA_dim2.max() / rhocpmax) * timestep / grid_spacing
    # vN_diff[1] = (
    #     np.max(UA_dim2[1:-1] / rhocp[1:])
    #     * timestep / (A_plate_eff * grid_spacing))
    # for the next two with non-constant gridspacing, find max of UA/gsp:
    vN_diff[2] = (
        (UA_port / (A_port * port_subs_gsp)).max() / rhocpmax * timestep
    )
    # vN_diff[2] = (UA_port / port_subs_gsp).max() / rhocpmax * timestep
    # vN_diff[2] = (
    #     np.max(UA_port / (A_port * port_subs_gsp)) * timestep / rhocp.max())
    # get maximum:
    vN_diff_max = vN_diff.max()
    # for massflow:
    # get maximum cfl number (this is the von Neumann stability condition
    # for massflow through cells), again with total max. of rho to get a
    # small security factor for worst case:
    Vcellrhomax = V_cell * rho_T.max()
    vN_dm_max = np.abs(dm_io).max() * timestep / Vcellrhomax

    # get maximum von Neumann stability condition values:
    # get dividers for maximum stable timestep to increase or decrease
    # stepsize:
    vN_diff_mult = vN_diff_max / 0.5
    vN_dm_mult = vN_dm_max / 1
    # get biggest divider:
    vN_div_max = max(vN_diff_mult, vN_dm_mult)
    # check if any L2 stability conditions are violated:
    if vN_div_max > 1:
        # only do something if von Neumann checking is active, else just
        # print an error but go on with the calculation:
        #        if check_vN:
        if True:
            # if not stable, set stable step bool to False
            step_stable[0] = False
            stability_breaches += 1  # increase breaches counter for this part
            # calculate required timestep to make this part stable with a
            # security factor of 0.95:
            local_vN_max_step = timestep / vN_div_max * 0.95
            # if this is the smallest step of all parts needed to make all
            # parts stable save it to maximum von Neumann step:
            if vN_max_step[0] > local_vN_max_step:
                vN_max_step[0] = local_vN_max_step
            #                # increase error weight of this part by the factor of 1.1 to
            #                # avoid running into this error again:
            #                self._trnc_err_cell_weight *= 1.1  # NOT good
            # adjust max factor if vN was violated:
            if max_factor[0] > 1.05:
                max_factor[0] = max_factor[0] ** 0.99
                if max_factor[0] < 1.05:  # clip to 1.05
                    max_factor[0] = 1.05
        else:
            print(
                '\nVon Neumann stability violated at step',
                stepnum,
                'and part with id',
                part_id,
                '!',
            )
            raise ValueError

    # return new values (or old values if unchanged):
    return step_stable, vN_max_step, max_factor


@njit(nogil=GLOB_NOGIL, cache=True)
def _vonNeumann_stability_var(
    part_id,
    stability_breaches,
    UA_tb,
    UA_port,
    UA_amb_shell,
    dm_top,
    dm_bot,
    dm_port,
    rho_T,
    rhocp,
    grid_spacing,
    port_subs_gsp,
    A_cell,
    A_port,
    A_shell,  # areas to backcalc diffusivity from UA
    r_total,
    V_cell,
    step_stable,  # check_vN, ,  # system wide bools
    vN_max_step,
    max_factor,
    stepnum,
    timestep,  # system wide vars
):
    r"""
    Check for L2/von Neumann stability for diffusion and massflows.

    Massflows are checked for parts where the massflow is defined as NOT
    invariant, that means where all cells in the part may have different
    massflow!

    Notes
    -----
    Von Neumann stability for conduction:

    .. math::
        r = \frac{\alpha \Delta t}{(\Delta x)^2} \leq \frac{1}{2} \\
        \text{With the thermal diffusivity: } \alpha = \frac{
        \lambda}{\rho c_{p}}\\
        \text{and } \lambda = \frac{U\cdot A}{A} \cdot \Delta x \\
        \text{yields } r = \frac{(UA)}{\rho c_{p}} \frac{\Delta t}{A \Delta x}
    Von Neumann stability for advection:
    """
    # save von Neumann stability values for cells by multiplying the cells
    # relevant total x-gridspacing with the maximum UA-value (this gives a
    # substitue heat conduction to get a total diffusion coefficient) and
    # the inverse maximum rho*cp value (of all cells! this may result in a
    # worst-case-result with a security factor of up to about 4.2%) to get
    # the substitute diffusion coefficient and then mult. with step and
    # div. by gridspacing**2 (not **2 since this is cut out with mult. by
    # it to get substitute diffusion from UA) and save to array:
    vN_diff = np.empty(3)

    # rhocpmax = rhocp.max()
    # For calculation see docstring
    # replaced old and faulty calculations with missing Area
    # vN_diff[0] = (UA_tb.max() / rhocpmax) * timestep / grid_spacing
    vN_diff[0] = (
        np.max(UA_tb[1:-1] / rhocp[1:]) * timestep / (A_cell * grid_spacing)
    )
    # for the next two with non-constant gridspacing, find max of UA/gsp:
    # vN_diff[1] = (UA_port / port_subs_gsp).max() / rhocpmax * timestep
    vN_diff[1] = (
        np.max(UA_port / (A_port * port_subs_gsp)) * timestep / rhocp.max()
    )
    # vN_diff[2] = UA_amb_shell.max() / r_total / rhocpmax * timestep
    vN_diff[2] = np.max(UA_amb_shell / rhocp) * timestep / (A_shell * r_total)
    # get maximum:
    vN_diff_max = vN_diff.max()
    # for massflow:
    # get maximum cfl number (this is the von Neumann stability condition
    # for massflow through cells), again with total max. of rho to get a
    # small security factor for worst case:
    Vcellrhomax = V_cell * rho_T.max()
    # NO checking for dims, since error probability of only having a critical
    # massflow sum at the port inflow cell and NOT at the next cell border is
    # extremely low AND this calculation would require either complicated
    # generated_jit functions OR keepdims support in sum! Thus just simple
    # check.
    #    if UA_port.ndim == 1:
    vN_dm_max = (
        max(dm_top.max(), dm_bot.max(), np.abs(dm_port).max())
        * timestep
        / Vcellrhomax
    )
    #    else:
    #        vN_dm = (
    #                max(dm_top.max(), dm_bot.max(),
    #                    np.abs(dm_port.sum(axis=0, keepdims=True)).max())
    #                * timestep / Vcellrhomax)

    # get maximum von Neumann stability condition values:
    # get dividers for maximum stable timestep to increase or decrease
    # stepsize:
    vN_diff_mult = vN_diff_max / 0.5
    vN_dm_mult = vN_dm_max / 1
    # get biggest divider:
    vN_div_max = max(vN_diff_mult, vN_dm_mult)
    # check if any L2 stability conditions are violated:
    if vN_div_max > 1.0:
        # only do something if von Neumann checking is active, else just
        # print an error but go on with the calculation:
        #        if check_vN:
        if True:
            # if not stable, set stable step bool to False
            step_stable[0] = False
            stability_breaches += 1  # increase breaches counter for this part
            # calculate required timestep to make this part stable with a
            # security factor of 0.95:
            local_vN_max_step = timestep / vN_div_max * 0.95
            # if this is the smallest step of all parts needed to make all
            # parts stable save it to maximum von Neumann step:
            if vN_max_step[0] > local_vN_max_step:
                vN_max_step[0] = local_vN_max_step
            #                # increase error weight of this part by the factor of 1.1 to
            #                # avoid running into this error again:
            #                self._trnc_err_cell_weight *= 1.1  # NOT good
            # adjust max factor if vN was violated:
            if max_factor[0] > 1.05:
                max_factor[0] = max_factor[0] ** 0.99
                if max_factor[0] < 1.05:  # clip to 1.05
                    max_factor[0] = 1.05
        else:
            print(
                '\nVon Neumann stability violated at step',
                stepnum,
                'and part with id',
                part_id,
                '!',
            )
            raise ValueError

    # return new values (or old values if unchanged):
    return step_stable, vN_max_step, max_factor


# %% Simulation Env. part specific differential functions:
@njit(nogil=GLOB_NOGIL, cache=True)
def pipe1D_diff(
    T_ext,
    T_port,
    T_s,
    T_amb,
    ports_all,  # temperatures
    dm_io,
    dm_top,
    dm_bot,
    dm_port,
    res_dm,  # flows
    cp_T,
    lam_T,
    rho_T,
    ny_T,
    lam_mean,
    cp_port,
    lam_port_fld,
    mcp,
    rhocp,
    lam_wll,
    lam_ins,
    mcp_wll,
    ui,  # material properties.
    alpha_i,
    alpha_inf,  # alpha values
    UA_tb,
    UA_tb_wll,
    UA_amb_shell,
    UA_port,
    UA_port_wll,  # UA values
    port_own_idx,
    port_link_idx,  # indices
    grid_spacing,
    port_gsp,
    port_subs_gsp,
    d_i,
    cell_dist,  # lengths
    flow_length,
    r_total,
    r_ln_wll,
    r_ln_ins,
    r_rins,  # lengths
    A_cell,
    V_cell,
    A_shell_i,
    A_shell_ins,
    A_p_fld_mean,  # areas and vols
    process_flows,
    vertical,
    step_stable,  # bools
    part_id,
    stability_breaches,
    vN_max_step,
    max_factor,  # misc.
    stepnum,  # step information
    dT_cond,
    dT_adv,
    dT_total,  # differentials
    timestep,
):

    process_flows[0] = _process_flow_invar(
        process_flows=process_flows,
        dm_io=dm_io,
        dm_top=dm_top,
        dm_bot=dm_bot,
        dm_port=dm_port,
        stepnum=stepnum,
        res_dm=res_dm,
    )

    water_mat_props_ext_view(
        T_ext=T_ext, cp_T=cp_T, lam_T=lam_T, rho_T=rho_T, ny_T=ny_T
    )

    # get mean lambda value between cells:
    _lambda_mean_view(lam_T=lam_T, out=lam_mean)

    UA_plate_tb(
        A_cell=A_cell,
        grid_spacing=grid_spacing,
        lam_mean=lam_mean,
        UA_tb_wll=UA_tb_wll,
        out=UA_tb,
    )

    # for conduction between current cell and ambient:
    # get outer pipe (insulation) surface temperature using a linearized
    # approach assuming steady state (assuming surface temperature = const.
    # for t -> infinity) and for cylinder shell (lids are omitted)
    surface_temp_steady_state_inplace(
        T=T_ext[1:-1],
        T_inf=T_amb[0],
        A_s=A_shell_ins,
        alpha_inf=alpha_inf,
        UA=UA_amb_shell,
        T_s=T_s,
    )
    # get inner alpha value between fluid and wall from nusselt equations:
    pipe_alpha_i(
        dm_io, T_ext[1:-1], rho_T, ny_T, lam_T, A_cell, d_i, cell_dist, alpha_i
    )
    # get outer alpha value between insulation and surrounding air:
    cylinder_alpha_inf(  # for a cylinder
        T_s=T_s,
        T_inf=T_amb[0],
        flow_length=flow_length,
        vertical=vertical,
        r_total=r_total,
        alpha_inf=alpha_inf,
    )
    # get resulting UA to ambient:
    UA_fld_wll_ins_amb_cyl(
        A_i=A_shell_i,
        r_ln_wll=r_ln_wll,
        r_ln_ins=r_ln_ins,
        r_rins=r_rins,
        alpha_i=alpha_i,
        alpha_inf=alpha_inf,
        lam_wll=lam_wll,
        lam_ins=lam_ins,
        out=UA_amb_shell,
    )

    # precalculate values which are needed multiple times:
    cell_temp_props_ext(
        T_ext=T_ext,
        V_cell=V_cell,
        cp_T=cp_T,
        rho_T=rho_T,
        mcp_wll=mcp_wll,
        rhocp=rhocp,
        mcp=mcp,
        ui=ui,
    )

    dT_cond_port = _process_ports_collapsed(
        ports_all=ports_all,
        port_link_idx=port_link_idx,
        port_own_idx=port_own_idx,
        T=T_ext[1:-1],
        mcp=mcp,
        UA_port=UA_port,
        UA_port_wll=UA_port_wll,
        A_p_fld_mean=A_p_fld_mean,
        port_gsp=port_gsp,
        grid_spacing=grid_spacing,
        lam_T=lam_T,
        cp_port=cp_port,
        lam_port_fld=lam_port_fld,
        T_port=T_port,
    )

    step_stable, vN_max_step, max_factor = _vonNeumann_stability_invar(
        part_id=part_id,
        stability_breaches=stability_breaches,
        UA_tb=UA_tb,
        UA_port=UA_port,
        UA_amb_shell=UA_amb_shell,
        dm_io=dm_io,
        rho_T=rho_T,
        rhocp=rhocp,
        grid_spacing=grid_spacing,
        port_subs_gsp=port_subs_gsp,
        A_cell=A_cell,
        A_port=A_p_fld_mean,
        A_shell=A_shell_ins,
        r_total=r_total,
        V_cell=V_cell,
        step_stable=step_stable,
        vN_max_step=vN_max_step,
        max_factor=max_factor,
        stepnum=stepnum,
        timestep=timestep,
    )

    # CALCULATE DIFFERENTIALS
    # calculate heat transfer by conduction
    dT_cond[:] = (
        +UA_tb[:-1] * (T_ext[:-2] - T_ext[1:-1])
        + UA_tb[1:] * (T_ext[2:] - T_ext[1:-1])
        #  + UA_port * (T_port - T_ext[1:-1])
        + UA_amb_shell * (T_amb[0] - T_ext[1:-1])
    ) / mcp
    # calculate heat transfer by advection
    dT_adv[:] = (
        +dm_top * (cp_T[:-2] * T_ext[:-2] - ui)
        + dm_bot * (cp_T[2:] * T_ext[2:] - ui)
    ) / mcp

    # sum up heat conduction and advection for port values:
    for i in range(port_own_idx.size):
        idx = port_own_idx[i]  # idx of port values at temperature/diff array
        # conduction
        dT_cond[idx] += dT_cond_port[i]
        # advection
        dT_adv[idx] += (
            dm_port[i] * (cp_port[i] * T_port[i] - ui[idx]) / mcp[idx]
        )

    dT_total[:] = dT_cond + dT_adv

    return dT_total


@njit(nogil=GLOB_NOGIL, cache=True)
def pipe1D_diff_fullstructarr(
    T_ext,
    ports_all,  # temperatures
    dm_io,
    res_dm,
    cp_T,
    lam_mean,
    UA_tb,
    port_link_idx,
    port_subs_gsp,
    step_stable,  # bools
    vN_max_step,
    max_factor,
    process_flows,
    vertical,  # misc.
    stepnum,
    ra1,
    ra2,
    ra5,
    timestep,
):
    """
    This function uses as many structured arrays as possible to reduce the
    time needed for calls to typeof_pyval. ra1, ra2, and ra5 are the
    structured arrays. ra1 contains single floats, ra2 all values of the shape
    of the port arrays and ra5 all non-extended value array sized arrays.
    The current speedup over using a list of args is so small, if any, that
    the easy approach if lists is preferred.
    As soon as structured arrays of variable shaped sub arrays is supported,
    this may become interesting.
    """

    process_flows[0] = _process_flow_invar(
        process_flows=process_flows,
        dm_io=dm_io,
        dm_top=ra5['dm_top'],
        dm_bot=ra5['dm_bot'],
        dm_port=ra2['dm_port'],
        stepnum=stepnum,
        res_dm=res_dm,
    )

    water_mat_props_ext_view(
        T_ext=T_ext,
        cp_T=cp_T,
        lam_T=ra5['lam_T'],
        rho_T=ra5['rho_T'],
        ny_T=ra5['ny_T'],
    )

    # get mean lambda value between cells:
    _lambda_mean_view(lam_T=ra5['lam_T'], out=lam_mean)

    UA_plate_tb(
        A_cell=ra1['A_cell'],
        grid_spacing=ra1['grid_spacing'],
        lam_mean=lam_mean,
        UA_tb_wll=ra1['UA_tb_wll'],
        out=UA_tb,
    )

    # for conduction between current cell and ambient:
    # get outer pipe (insulation) surface temperature using a linearized
    # approach assuming steady state (assuming surface temperature = const.
    # for t -> infinity) and for cylinder shell (lids are omitted)
    surface_temp_steady_state_inplace(
        T=T_ext[1:-1],
        T_inf=ra1['T_amb'][0],
        A_s=ra1['A_shell_ins'],
        alpha_inf=ra5['alpha_inf'],
        UA=ra5['UA_amb_shell'],
        T_s=ra5['T_s'],
    )
    # get inner alpha value between fluid and wall from nusselt equations:
    pipe_alpha_i(
        dm_io,
        T_ext[1:-1],
        ra5['rho_T'],
        ra5['ny_T'],
        ra5['lam_T'],
        ra1['A_cell'],
        ra1['d_i'],
        ra5['cell_dist'],
        ra5['alpha_i'],
    )
    # get outer alpha value between insulation and surrounding air:
    cylinder_alpha_inf(  # for a cylinder
        T_s=ra5['T_s'],
        T_inf=ra1['T_amb'][0],
        flow_length=ra1['flow_length'],
        vertical=vertical,
        r_total=ra1['r_total'],
        alpha_inf=ra5['alpha_inf'],
    )
    # get resulting UA to ambient:
    UA_fld_wll_ins_amb_cyl(
        A_i=ra1['A_shell_i'],
        r_ln_wll=ra1['r_ln_wll'],
        r_ln_ins=ra1['r_ln_ins'],
        r_rins=ra1['r_rins'],
        alpha_i=ra5['alpha_i'],
        alpha_inf=ra5['alpha_inf'],
        lam_wll=ra1['lam_wll'],
        lam_ins=ra1['lam_ins'],
        out=ra5['UA_amb_shell'],
    )

    # precalculate values which are needed multiple times:
    cell_temp_props_ext(
        T_ext=T_ext,
        V_cell=ra1['V_cell'],
        cp_T=cp_T,
        rho_T=ra5['rho_T'],
        mcp_wll=ra1['mcp_wll'],
        rhocp=ra5['rhocp'],
        mcp=ra5['mcp'],
        ui=ra5['ui'],
    )

    dT_cond_port = _process_ports_collapsed(
        ports_all=ports_all,
        port_link_idx=port_link_idx,
        port_own_idx=ra2['port_own_idx'],
        T=T_ext[1:-1],
        mcp=ra5['mcp'],
        UA_port=ra2['UA_port'],
        UA_port_wll=ra2['UA_port_wll'],
        A_p_fld_mean=ra2['A_p_fld_mean'],
        port_gsp=ra2['port_gsp'],
        grid_spacing=ra1['grid_spacing'][0],
        lam_T=ra5['lam_T'],
        cp_port=ra2['cp_port'],
        lam_port_fld=ra2['lam_port_fld'],
        T_port=ra2['T_port'],
    )

    step_stable, vN_max_step, max_factor = _vonNeumann_stability_invar(
        part_id=ra1['part_id'],
        stability_breaches=ra1['stability_breaches'],
        UA_tb=UA_tb,
        UA_port=ra2['UA_port'],
        UA_amb_shell=ra5['UA_amb_shell'],
        dm_io=dm_io,
        rho_T=ra5['rho_T'],
        rhocp=ra5['rhocp'],
        grid_spacing=ra1['grid_spacing'][0],
        port_subs_gsp=port_subs_gsp,
        A_cell=ra1['A_cell'],
        A_port=ra2['A_p_fld_mean'],
        A_shell=ra1['A_shell_ins'],
        r_total=ra1['r_total'][0],
        V_cell=ra1['V_cell'][0],
        step_stable=step_stable,
        vN_max_step=vN_max_step,
        max_factor=max_factor,
        stepnum=stepnum,
        timestep=timestep,
    )

    # CALCULATE DIFFERENTIALS
    # calculate heat transfer by conduction
    ra5['dT_cond'][:] = (
        +UA_tb[:-1] * (T_ext[:-2] - T_ext[1:-1])
        + UA_tb[1:] * (T_ext[2:] - T_ext[1:-1])
        #  + UA_port * (T_port - T_ext[1:-1])
        + ra5['UA_amb_shell'] * (ra1['T_amb'][0] - T_ext[1:-1])
    ) / ra5['mcp']
    #    dT_cond[0] += dT_cond_port[0]
    #    dT_cond[-1] += dT_cond_port[-1]
    # calculate heat transfer by advection
    ra5['dT_adv'][:] = (
        (
            +ra5['dm_top'] * (cp_T[:-2] * T_ext[:-2] - ra5['ui'])
            + ra5['dm_bot'] * (cp_T[2:] * T_ext[2:] - ra5['ui'])
        )
        #             + dm_port * (cp_port * T_port - ui))
        / ra5['mcp']
    )
    #    dT_adv[0] += dm_port[0] * (cp_port[0] * T_port[0] - ui[0]) / mcp[0]
    #    dT_adv[-1] += dm_port[-1] * (cp_port[-1] * T_port[-1] - ui[-1]) / mcp[-1]

    # T_port and cp_port NOT collapsed
    #    for i in range(port_own_idx.size):
    #        idx = port_own_idx[i]
    #        dT_cond[idx] += dT_cond_port[i]
    #        dT_adv[idx] += (
    #                dm_port[idx] * (cp_port[idx] * T_port[idx] - ui[idx])
    #                / mcp[idx])
    # all (except dm_port) collapsed:
    for i in range(ra2['port_own_idx'].size):
        idx = ra2['port_own_idx'][i]
        ra5['dT_cond'][idx] += dT_cond_port[i]
        #        dT_adv[idx] += (  # dm_port like T
        #                dm_port[idx] * (cp_port[i] * T_port[i] - ui[idx])
        #                / mcp[idx])
        ra5['dT_adv'][idx] += (  # dm port only 2 cells
            ra2['dm_port'][i]
            * (ra2['cp_port'][i] * ra2['T_port'][i] - ra5['ui'][idx])
            / ra5['mcp'][idx]
        )

    ra5['dT_total'][:] = ra5['dT_cond'] + ra5['dT_adv']

    return ra5['dT_total']


@njit(nogil=GLOB_NOGIL, cache=True)
def pipe1D_diff_structarr(
    T_ext,
    T_port,
    T_s,
    T_amb,
    ports_all,  # temperatures
    dm_io,
    dm_top,
    dm_bot,
    dm_port,
    res_dm,  # flows
    cp_T,
    lam_T,
    rho_T,
    ny_T,
    lam_mean,
    cp_port,
    lam_port_fld,  # mat. props.
    mcp,
    rhocp,
    ui,  # material properties.
    alpha_i,
    alpha_inf,  # alpha values
    UA_tb,
    UA_amb_shell,
    UA_port,
    UA_port_wll,  # UA values
    port_own_idx,
    port_link_idx,  # indices
    port_gsp,
    port_subs_gsp,
    cell_dist,  # lengths
    A_p_fld_mean,  # areas and vols
    process_flows,
    step_stable,  # bools
    stability_breaches,
    vN_max_step,
    max_factor,  # misc.
    stepnum,  # step information
    dT_cond,
    dT_adv,
    dT_total,  # differentials
    sar,  # structarr
    vertical,
    part_id,
    timestep,
):

    process_flows[0] = _process_flow_invar(
        process_flows=process_flows,
        dm_io=dm_io,
        dm_top=dm_top,
        dm_bot=dm_bot,
        dm_port=dm_port,
        stepnum=stepnum,
        res_dm=res_dm,
    )

    water_mat_props_ext_view(
        T_ext=T_ext, cp_T=cp_T, lam_T=lam_T, rho_T=rho_T, ny_T=ny_T
    )

    # get mean lambda value between cells:
    _lambda_mean_view(lam_T=lam_T, out=lam_mean)

    UA_plate_tb(
        A_cell=sar['A_cell'][0],
        grid_spacing=sar['grid_spacing'][0],
        lam_mean=lam_mean,
        UA_tb_wll=sar['UA_tb_wll'][0],
        out=UA_tb,
    )

    # for conduction between current cell and ambient:
    # get outer pipe (insulation) surface temperature using a linearized
    # approach assuming steady state (assuming surface temperature = const.
    # for t -> infinity) and for cylinder shell (lids are omitted)
    surface_temp_steady_state_inplace(
        T=T_ext[1:-1],
        T_inf=T_amb[0],
        A_s=sar['A_shell_ins'][0],
        alpha_inf=alpha_inf,
        UA=UA_amb_shell,
        T_s=T_s,
    )
    # get inner alpha value between fluid and wall from nusselt equations:
    pipe_alpha_i(
        dm_io,
        T_ext[1:-1],
        rho_T,
        ny_T,
        lam_T,
        sar['A_cell'][0],
        sar['d_i'][0],
        cell_dist,
        alpha_i,
    )
    # get outer alpha value between insulation and surrounding air:
    cylinder_alpha_inf(  # for a cylinder
        T_s=T_s,
        T_inf=T_amb[0],
        flow_length=sar['flow_length'][0],
        vertical=vertical,
        r_total=sar['r_total'][0],
        alpha_inf=alpha_inf,
    )
    # get resulting UA to ambient:
    UA_fld_wll_ins_amb_cyl(
        A_i=sar['A_shell_i'][0],
        r_ln_wll=sar['r_ln_wll'][0],
        r_ln_ins=sar['r_ln_ins'][0],
        r_rins=sar['r_rins'][0],
        alpha_i=alpha_i,
        alpha_inf=alpha_inf,
        lam_wll=sar['lam_wll'][0],
        lam_ins=sar['lam_ins'][0],
        out=UA_amb_shell,
    )

    # precalculate values which are needed multiple times:
    cell_temp_props_ext(
        T_ext=T_ext,
        V_cell=sar['V_cell'][0],
        cp_T=cp_T,
        rho_T=rho_T,
        mcp_wll=sar['mcp_wll'][0],
        rhocp=rhocp,
        mcp=mcp,
        ui=ui,
    )

    dT_cond_port = _process_ports_collapsed(
        ports_all=ports_all,
        port_link_idx=port_link_idx,
        port_own_idx=port_own_idx,
        T=T_ext[1:-1],
        mcp=mcp,
        UA_port=UA_port,
        UA_port_wll=UA_port_wll,
        A_p_fld_mean=A_p_fld_mean,
        port_gsp=port_gsp,
        grid_spacing=sar['grid_spacing'][0],
        lam_T=lam_T,
        cp_port=cp_port,
        lam_port_fld=lam_port_fld,
        T_port=T_port,
    )

    step_stable, vN_max_step, max_factor = _vonNeumann_stability_invar(
        part_id=part_id,
        stability_breaches=stability_breaches,
        UA_tb=UA_tb,
        UA_port=UA_port,
        UA_amb_shell=UA_amb_shell,
        dm_io=dm_io,
        rho_T=rho_T,
        rhocp=rhocp,
        grid_spacing=sar['grid_spacing'][0],
        port_subs_gsp=port_subs_gsp,
        A_cell=sar['A_cell'],
        A_port=A_p_fld_mean,
        A_shell=sar['A_shell_ins'],
        r_total=sar['r_total'][0],
        V_cell=sar['V_cell'][0],
        step_stable=step_stable,
        vN_max_step=vN_max_step,
        max_factor=max_factor,
        stepnum=stepnum,
        timestep=timestep,
    )

    # CALCULATE DIFFERENTIALS
    # calculate heat transfer by conduction
    dT_cond[:] = (
        +UA_tb[:-1] * (T_ext[:-2] - T_ext[1:-1])
        + UA_tb[1:] * (T_ext[2:] - T_ext[1:-1])
        #  + UA_port * (T_port - T_ext[1:-1])
        + UA_amb_shell * (T_amb[0] - T_ext[1:-1])
    ) / mcp
    #    dT_cond[0] += dT_cond_port[0]
    #    dT_cond[-1] += dT_cond_port[-1]
    # calculate heat transfer by advection
    dT_adv[:] = (
        (
            +dm_top * (cp_T[:-2] * T_ext[:-2] - ui)
            + dm_bot * (cp_T[2:] * T_ext[2:] - ui)
        )
        #             + dm_port * (cp_port * T_port - ui))
        / mcp
    )
    #    dT_adv[0] += dm_port[0] * (cp_port[0] * T_port[0] - ui[0]) / mcp[0]
    #    dT_adv[-1] += dm_port[-1] * (cp_port[-1] * T_port[-1] - ui[-1]) / mcp[-1]

    # T_port and cp_port NOT collapsed
    #    for i in range(port_own_idx.size):
    #        idx = port_own_idx[i]
    #        dT_cond[idx] += dT_cond_port[i]
    #        dT_adv[idx] += (
    #                dm_port[idx] * (cp_port[idx] * T_port[idx] - ui[idx])
    #                / mcp[idx])
    # all (except dm_port) collapsed:
    for i in range(port_own_idx.size):
        idx = port_own_idx[i]
        dT_cond[idx] += dT_cond_port[i]
        #        dT_adv[idx] += (  # dm_port like T
        #                dm_port[idx] * (cp_port[i] * T_port[i] - ui[idx])
        #                / mcp[idx])
        dT_adv[idx] += (  # dm port only 2 cells
            dm_port[i] * (cp_port[i] * T_port[i] - ui[idx]) / mcp[idx]
        )

    dT_total[:] = dT_cond + dT_adv

    return dT_total


@njit(nogil=GLOB_NOGIL, cache=True)
def pipe1D_branched_diff(
    T_ext,
    T_port,
    T_s,
    T_amb,
    ports_all,  # temperatures
    dm_io,
    dm,
    dm_top,
    dm_bot,
    dm_port,
    res_dm,  # flows
    cp_T,
    lam_T,
    rho_T,
    ny_T,
    lam_mean,
    cp_port,
    lam_port_fld,
    mcp,
    rhocp,
    lam_wll,
    lam_ins,
    mcp_wll,
    ui,  # material properties.
    alpha_i,
    alpha_inf,  # alpha values
    UA_tb,
    UA_tb_wll,
    UA_amb_shell,
    UA_port,
    UA_port_wll,  # UA values
    port_own_idx,
    port_link_idx,  # indices
    grid_spacing,
    port_gsp,
    port_subs_gsp,
    d_i,
    cell_dist,  # lengths
    flow_length,
    r_total,
    r_ln_wll,
    r_ln_ins,
    r_rins,  # lengths
    A_cell,
    V_cell,
    A_shell_i,
    A_shell_ins,
    A_p_fld_mean,  # areas and vols
    process_flows,
    vertical,
    step_stable,  # bools
    part_id,
    stability_breaches,
    vN_max_step,
    max_factor,  # misc.
    stepnum,  # step information
    dT_cond,
    dT_adv,
    dT_total,  # differentials
    timestep,
):

    process_flows[0] = _process_flow_var(
        process_flows=process_flows,
        dm_io=dm_io,
        dm=dm,
        dm_top=dm_top,
        dm_bot=dm_bot,
        dm_port=dm_port,
        port_own_idx=port_own_idx,
        stepnum=stepnum,
        res_dm=res_dm,
    )

    water_mat_props_ext_view(
        T_ext=T_ext, cp_T=cp_T, lam_T=lam_T, rho_T=rho_T, ny_T=ny_T
    )

    # get mean lambda value between cells:
    _lambda_mean_view(lam_T=lam_T, out=lam_mean)

    UA_plate_tb(
        A_cell=A_cell,
        grid_spacing=grid_spacing,
        lam_mean=lam_mean,
        UA_tb_wll=UA_tb_wll,
        out=UA_tb,
    )

    # for conduction between current cell and ambient:
    # get outer pipe (insulation) surface temperature using a linearized
    # approach assuming steady state (assuming surface temperature = const.
    # for t -> infinity) and for cylinder shell (lids are omitted)
    surface_temp_steady_state_inplace(
        T=T_ext[1:-1],
        T_inf=T_amb[0],
        A_s=A_shell_ins,
        alpha_inf=alpha_inf,
        UA=UA_amb_shell,
        T_s=T_s,
    )
    # get inner alpha value between fluid and wall from nusselt equations:
    pipe_alpha_i(
        dm, T_ext[1:-1], rho_T, ny_T, lam_T, A_cell, d_i, cell_dist, alpha_i
    )
    # get outer alpha value between insulation and surrounding air:
    cylinder_alpha_inf(  # for a cylinder
        T_s=T_s,
        T_inf=T_amb[0],
        flow_length=flow_length,
        vertical=vertical,
        r_total=r_total,
        alpha_inf=alpha_inf,
    )
    # get resulting UA to ambient:
    UA_fld_wll_ins_amb_cyl(
        A_i=A_shell_i,
        r_ln_wll=r_ln_wll,
        r_ln_ins=r_ln_ins,
        r_rins=r_rins,
        alpha_i=alpha_i,
        alpha_inf=alpha_inf,
        lam_wll=lam_wll,
        lam_ins=lam_ins,
        out=UA_amb_shell,
    )

    # precalculate values which are needed multiple times:
    cell_temp_props_ext(
        T_ext=T_ext,
        V_cell=V_cell,
        cp_T=cp_T,
        rho_T=rho_T,
        mcp_wll=mcp_wll,
        rhocp=rhocp,
        mcp=mcp,
        ui=ui,
    )

    dT_cond_port = _process_ports_collapsed(
        ports_all=ports_all,
        port_link_idx=port_link_idx,
        port_own_idx=port_own_idx,
        T=T_ext[1:-1],
        mcp=mcp,
        UA_port=UA_port,
        UA_port_wll=UA_port_wll,
        A_p_fld_mean=A_p_fld_mean,
        port_gsp=port_gsp,
        grid_spacing=grid_spacing,
        lam_T=lam_T,
        cp_port=cp_port,
        lam_port_fld=lam_port_fld,
        T_port=T_port,
    )

    step_stable, vN_max_step, max_factor = _vonNeumann_stability_invar(
        part_id=part_id,
        stability_breaches=stability_breaches,
        UA_tb=UA_tb,
        UA_port=UA_port,
        UA_amb_shell=UA_amb_shell,
        dm_io=dm_io,
        rho_T=rho_T,
        rhocp=rhocp,
        grid_spacing=grid_spacing,
        port_subs_gsp=port_subs_gsp,
        A_cell=A_cell,
        A_port=A_p_fld_mean,
        A_shell=A_shell_ins,
        r_total=r_total,
        V_cell=V_cell,
        step_stable=step_stable,
        vN_max_step=vN_max_step,
        max_factor=max_factor,
        stepnum=stepnum,
        timestep=timestep,
    )

    # CALCULATE DIFFERENTIALS
    # calculate heat transfer by conduction
    dT_cond[:] = (
        +UA_tb[:-1] * (T_ext[:-2] - T_ext[1:-1])
        + UA_tb[1:] * (T_ext[2:] - T_ext[1:-1])
        #  + UA_port * (T_port - T_ext[1:-1])
        + UA_amb_shell * (T_amb[0] - T_ext[1:-1])
    ) / mcp
    # calculate heat transfer by advection
    dT_adv[:] = (
        +dm_top * (cp_T[:-2] * T_ext[:-2] - ui)
        + dm_bot * (cp_T[2:] * T_ext[2:] - ui)
    ) / mcp

    # sum up heat conduction and advection for port values:
    for i in range(port_own_idx.size):
        idx = port_own_idx[i]  # idx of port values at temperature/diff array
        # conduction
        dT_cond[idx] += dT_cond_port[i]
        # advection
        dT_adv[idx] += (
            dm_port[i] * (cp_port[i] * T_port[i] - ui[idx]) / mcp[idx]
        )

    dT_total[:] = dT_cond + dT_adv

    return dT_total


@njit(nogil=GLOB_NOGIL, cache=True)
def heatedpipe1D_diff(
    T_ext,
    T_port,
    T_s,
    T_amb,
    ports_all,  # temperatures
    dm_io,
    dm_top,
    dm_bot,
    dm_port,
    dQ_heating,
    res_dm,
    res_dQ,  # flows
    cp_T,
    lam_T,
    rho_T,
    ny_T,
    lam_mean,
    cp_port,
    lam_port_fld,  # mat. props.
    mcp,
    mcp_heated,
    rhocp,
    lam_wll,
    lam_ins,
    mcp_wll,
    ui,  # material properties.
    alpha_i,
    alpha_inf,  # alpha values
    UA_tb,
    UA_tb_wll,
    UA_amb_shell,
    UA_port,
    UA_port_wll,  # UA values
    port_own_idx,
    port_link_idx,
    heat_mult,  # indices
    grid_spacing,
    port_gsp,
    port_subs_gsp,
    d_i,
    cell_dist,  # lengths
    flow_length,
    r_total,
    r_ln_wll,
    r_ln_ins,
    r_rins,  # lengths
    A_cell,
    V_cell,
    A_shell_i,
    A_shell_ins,
    A_p_fld_mean,  # areas and vols
    process_flows,
    vertical,
    step_stable,  # bools
    part_id,
    stability_breaches,
    vN_max_step,
    max_factor,  # misc.
    stepnum,
    timestep,  # step information
    dT_cond,
    dT_adv,
    dT_heat,
    dT_heated,  # differentials
    emergency_shutdown=110.0,
):
    # shutdown gasboiler immediately if any temperatures are exceeding
    # emergency_shutdown-value
    if np.any(T_ext >= emergency_shutdown):
        dQ_heating[:] = 0.0

    # save rate of heat flow to result array
    if process_flows[0]:  # only if flows not already processed
        res_dQ[stepnum] = dQ_heating

    process_flows[0] = _process_flow_invar(
        process_flows=process_flows,
        dm_io=dm_io,
        dm_top=dm_top,
        dm_bot=dm_bot,
        dm_port=dm_port,
        stepnum=stepnum,
        res_dm=res_dm,
    )

    water_mat_props_ext_view(
        T_ext=T_ext, cp_T=cp_T, lam_T=lam_T, rho_T=rho_T, ny_T=ny_T
    )

    # get mean lambda value between cells:
    _lambda_mean_view(lam_T=lam_T, out=lam_mean)

    UA_plate_tb(
        A_cell=A_cell,
        grid_spacing=grid_spacing,
        lam_mean=lam_mean,
        UA_tb_wll=UA_tb_wll,
        out=UA_tb,
    )

    # for conduction between current cell and ambient:
    # get outer pipe (insulation) surface temperature using a linearized
    # approach assuming steady state (assuming surface temperature = const.
    # for t -> infinity) and for cylinder shell (lids are omitted)
    surface_temp_steady_state_inplace(
        T=T_ext[1:-1],
        T_inf=T_amb[0],
        A_s=A_shell_ins,
        alpha_inf=alpha_inf,
        UA=UA_amb_shell,
        T_s=T_s,
    )
    # get inner alpha value between fluid and wall from nusselt equations:
    pipe_alpha_i(
        dm_io, T_ext[1:-1], rho_T, ny_T, lam_T, A_cell, d_i, cell_dist, alpha_i
    )
    # get outer alpha value between insulation and surrounding air:
    cylinder_alpha_inf(  # for a cylinder
        T_s=T_s,
        T_inf=T_amb[0],
        flow_length=flow_length,
        vertical=vertical,
        r_total=r_total,
        alpha_inf=alpha_inf,
    )
    # get resulting UA to ambient:
    UA_fld_wll_ins_amb_cyl(
        A_i=A_shell_i,
        r_ln_wll=r_ln_wll,
        r_ln_ins=r_ln_ins,
        r_rins=r_rins,
        alpha_i=alpha_i,
        alpha_inf=alpha_inf,
        lam_wll=lam_wll,
        lam_ins=lam_ins,
        out=UA_amb_shell,
    )

    # precalculate values which are needed multiple times:
    cell_temp_props_ext(
        T_ext=T_ext,
        V_cell=V_cell,
        cp_T=cp_T,
        rho_T=rho_T,
        mcp_wll=mcp_wll,
        rhocp=rhocp,
        mcp=mcp,
        ui=ui,
    )

    dT_cond_port = _process_ports_collapsed(
        ports_all=ports_all,
        port_link_idx=port_link_idx,
        port_own_idx=port_own_idx,
        T=T_ext[1:-1],
        mcp=mcp,
        UA_port=UA_port,
        UA_port_wll=UA_port_wll,
        A_p_fld_mean=A_p_fld_mean,
        port_gsp=port_gsp,
        grid_spacing=grid_spacing,
        lam_T=lam_T,
        cp_port=cp_port,
        lam_port_fld=lam_port_fld,
        T_port=T_port,
    )

    step_stable, vN_max_step, max_factor = _vonNeumann_stability_invar(
        part_id=part_id,
        stability_breaches=stability_breaches,
        UA_tb=UA_tb,
        UA_port=UA_port,
        UA_amb_shell=UA_amb_shell,
        dm_io=dm_io,
        rho_T=rho_T,
        rhocp=rhocp,
        grid_spacing=grid_spacing,
        port_subs_gsp=port_subs_gsp,
        A_cell=A_cell,
        A_port=A_p_fld_mean,
        A_shell=A_shell_ins,
        r_total=r_total,
        V_cell=V_cell,
        step_stable=step_stable,
        vN_max_step=vN_max_step,
        max_factor=max_factor,
        stepnum=stepnum,
        timestep=timestep,
    )

    # CALCULATE DIFFERENTIALS
    # calculate heat transfer by internal heat sources
    dT_heated[:] = dQ_heating * heat_mult / mcp_heated

    # calculate heat transfer by conduction
    dT_cond[:] = (
        +UA_tb[:-1] * (T_ext[:-2] - T_ext[1:-1])
        + UA_tb[1:] * (T_ext[2:] - T_ext[1:-1])
        + UA_amb_shell * (T_amb[0] - T_ext[1:-1])
    ) / mcp
    # calculate heat transfer by advection
    dT_adv[:] = (
        +dm_top * (cp_T[:-2] * T_ext[:-2] - ui)
        + dm_bot * (cp_T[2:] * T_ext[2:] - ui)
    ) / mcp

    # sum up heat conduction and advection for port values:
    for i in range(port_own_idx.size):
        idx = port_own_idx[i]  # idx of port values at temperature/diff array
        # conduction
        dT_cond[idx] += dT_cond_port[i]
        # advection
        dT_adv[idx] += (
            dm_port[i] * (cp_port[i] * T_port[i] - ui[idx]) / mcp[idx]
        )

    dT_total = dT_cond + dT_adv + dT_heat

    return dT_total, process_flows, step_stable, vN_max_step, max_factor


@njit(nogil=GLOB_NOGIL, cache=True)
def tes_diff(
    T_ext,
    T_port,
    T_s,
    T_s_lid,
    T_amb,
    ports_all,  # temperatures
    dm_io,
    dm,
    dm_top,
    dm_bot,
    dm_port,
    res_dm,  # flows
    cp_T,
    lam_T,
    rho_T,
    ny_T,
    lam_mean,
    cp_port,
    lam_port_fld,
    mcp,
    rhocp,
    lam_wll,
    lam_ins,
    mcp_wll,
    ui,  # mat. props.
    alpha_i,
    alpha_inf,  # alpha_inf_lid,  # alpha values
    UA_tb,
    UA_tb_wll,
    UA_amb_shell,
    UA_amb_lid,
    UA_port,
    UA_port_wll,  # UA values
    port_own_idx,
    port_link_idx,  # indices
    grid_spacing,
    port_gsp,
    port_subs_gsp,
    d_i,
    cell_dist,
    flow_length,
    flow_length_lid,
    r_total,
    r_ln_wll,
    r_ln_ins,
    r_rins,
    s_wll,
    s_ins,  # lengths
    A_cell,
    V_cell,
    A_shell_i,
    A_shell_ins,
    A_p_fld_mean,  # areas and vols
    process_flows,
    vertical,
    vertical_lid,
    lid_top,
    step_stable,  # bools
    part_id,
    stability_breaches,
    vN_max_step,
    max_factor,  # misc.
    stepnum,  # step information
    dT_cond,
    dT_adv,
    dT_total,  # differentials
    # T, T_top, T_bot, # T+bot/top NOT NEEDED ANYMORE
    # cp_top, cp_bot,  # cp_top/botT NOT NEEDED ANYMORE
    timestep,
):

    process_flows[0] = _process_flow_var(
        process_flows=process_flows,
        dm_io=dm_io,
        dm=dm,
        dm_top=dm_top,
        dm_bot=dm_bot,
        dm_port=dm_port,
        port_own_idx=port_own_idx,
        stepnum=stepnum,
        res_dm=res_dm,
    )

    water_mat_props_ext_view(
        T_ext=T_ext, cp_T=cp_T, lam_T=lam_T, rho_T=rho_T, ny_T=ny_T
    )

    # get mean lambda value between cells:
    _lambda_mean_view(lam_T=lam_T, out=lam_mean)

    # calculate buoyancy with Nusselt correction:
    buoyancy_byNusselt(T=T_ext[1:-1], ny=ny_T, d_i=d_i, lam_mean=lam_mean)

    UA_plate_tb(
        A_cell=A_cell,
        grid_spacing=grid_spacing,
        lam_mean=lam_mean,
        UA_tb_wll=UA_tb_wll,
        out=UA_tb,
    )

    # for conduction between current cell and ambient:
    # get outer pipe (insulation) surface temperature using a linearized
    # approach assuming steady state (assuming surface temperature = const.
    # for t -> infinity) and for cylinder shell (lids are omitted)
    surface_temp_steady_state_inplace(
        T=T_ext[1:-1],
        T_inf=T_amb[0],
        A_s=A_shell_ins,
        alpha_inf=alpha_inf,
        UA=UA_amb_shell,
        T_s=T_s,
    )
    # get inner alpha value between fluid and wall from nusselt equations:
    pipe_alpha_i(
        dm=dm,
        T=T_ext[1:-1],
        rho=rho_T,
        ny=ny_T,
        lam_fld=lam_T,
        A=A_cell,
        d_i=d_i,
        x=cell_dist,
        alpha=alpha_i,
    )
    # get outer alpha value between insulation and surrounding air:
    cylinder_alpha_inf(  # for a cylinder
        T_s=T_s,
        T_inf=T_amb[0],
        flow_length=flow_length,
        vertical=vertical,
        r_total=r_total,
        alpha_inf=alpha_inf,
    )
    alpha_inf_lid = plane_alpha_inf(
        T_s=T_s_lid,
        T_inf=T_amb[0],
        flow_length=flow_length_lid,
        vertical=vertical_lid,
        top=lid_top,
    )
    # get resulting UA to ambient:
    UA_fld_wll_ins_amb_cyl(
        A_i=A_shell_i,
        r_ln_wll=r_ln_wll,
        r_ln_ins=r_ln_ins,
        r_rins=r_rins,
        alpha_i=alpha_i,
        alpha_inf=alpha_inf,
        lam_wll=lam_wll,
        lam_ins=lam_ins,
        out=UA_amb_shell,
    )
    UA_amb_lid[:] = UA_fld_wll_ins_amb_plate(
        A=A_cell,
        s_wll=s_wll,
        s_ins=s_ins,  # alpha_i FIRST AND LAST element! alpha_fld=alpha_i[0],
        alpha_fld=alpha_i[:: alpha_i.size - 1],
        alpha_inf=alpha_inf_lid,
        lam_wll=lam_wll,
        lam_ins=lam_ins,
    )

    # precalculate values which are needed multiple times:
    cell_temp_props_ext(
        T_ext=T_ext,
        V_cell=V_cell,
        cp_T=cp_T,
        rho_T=rho_T,
        mcp_wll=mcp_wll,
        rhocp=rhocp,
        mcp=mcp,
        ui=ui,
    )

    # dT_cond_port = _process_ports_collapsed(
    _ = _process_ports_collapsed(
        ports_all=ports_all,
        port_link_idx=port_link_idx,
        port_own_idx=port_own_idx,
        T=T_ext[1:-1],
        mcp=mcp,
        UA_port=UA_port,
        UA_port_wll=UA_port_wll,
        A_p_fld_mean=A_p_fld_mean,
        port_gsp=port_gsp,
        grid_spacing=grid_spacing,
        lam_T=lam_T,
        cp_port=cp_port,
        lam_port_fld=lam_port_fld,
        T_port=T_port,
    )

    step_stable, vN_max_step, max_factor = _vonNeumann_stability_var(
        part_id=part_id,
        stability_breaches=stability_breaches,
        UA_tb=UA_tb,
        UA_port=UA_port,
        UA_amb_shell=UA_amb_shell,
        dm_top=dm_top,
        dm_bot=dm_bot,
        dm_port=dm_port,
        rho_T=rho_T,
        rhocp=rhocp,
        grid_spacing=grid_spacing,
        port_subs_gsp=port_subs_gsp,
        A_cell=A_cell,
        A_port=A_p_fld_mean,
        A_shell=A_shell_ins,
        r_total=r_total,
        V_cell=V_cell,
        step_stable=step_stable,
        vN_max_step=vN_max_step,
        max_factor=max_factor,
        stepnum=stepnum,
        timestep=timestep,
    )

    if T_port.ndim == 1:
        # calculate heat transfer by conduction
        dT_cond[:] = (  # EXTENDED ARRAY VERSION
            +UA_tb[:-1] * (T_ext[:-2] - T_ext[1:-1])
            + UA_tb[1:] * (T_ext[2:] - T_ext[1:-1])
            + UA_amb_shell * (T_amb[0] - T_ext[1:-1])
        ) / mcp
        # add losses through top and bottom lid:
        dT_cond[0] += (  # EXTENDED ARRAY VERSION
            UA_amb_lid[0] * (T_amb[0] - T_ext[1]) / mcp[0]
        )
        dT_cond[-1] += UA_amb_lid[-1] * (T_amb[0] - T_ext[-2]) / mcp[-1]
        # calculate heat transfer by advection
        dT_adv[:] = (  # EXTENDED ARRAY VERSION
            +dm_top * (cp_T[:-2] * T_ext[:-2] - ui)
            + dm_bot * (cp_T[2:] * T_ext[2:] - ui)
        ) / mcp
    else:
        # the same if multiple ports per cell exist. to correctly calculate
        # this, the sum of the arrays has to be taken:
        dT_cond[:] = (  # EXTENDED ARRAY VERSION
            +UA_tb[:-1] * (T_ext[:-2] - T_ext[1:-1])
            + UA_tb[1:] * (T_ext[2:] - T_ext[1:-1])
            + UA_amb_shell * (T_amb[0] - T_ext[1:-1])
        ) / mcp
        # add losses through top and bottom lid:
        dT_cond[0] += (  # EXTENDED ARRAY VERSION
            UA_amb_lid[0] * (T_amb[0] - T_ext[1]) / mcp[0]
        )
        dT_cond[-1] += UA_amb_lid[-1] * (T_amb[0] - T_ext[-2]) / mcp[-1]
        # calculate heat transfer by advection
        dT_adv[:] = (  # EXTENDED ARRAY VERSION
            +dm_top * (cp_T[:-2] * T_ext[:-2] - ui)
            + dm_bot * (cp_T[2:] * T_ext[2:] - ui)
        ) / mcp

    for i in range(T_port.size):
        idx = port_own_idx[i]
        #        dT_cond[idx] += dT_cond_port[i]
        # heat conduction over ports (T_ext[idx+1] since index is not extended)
        dT_cond[idx] += UA_port[i] * (T_port[i] - T_ext[idx + 1]) / mcp[idx]
        # heat advection through ports
        dT_adv[idx] += (
            dm_port.flat[i]
            * (cp_port[i] * T_port[i] - ui[idx])  # collapsed dmport
            / mcp[idx]
        )
    # sum up all differentials
    dT_total[:] = dT_cond + dT_adv

    return (
        dT_total,
        process_flows,
        step_stable,
        vN_max_step,
        max_factor,
        alpha_inf_lid,
    )


@njit(nogil=GLOB_NOGIL, cache=True)
def chp_core_diff(
    T_ext,
    T_port,
    T_s,
    T_amb,
    ports_all,  # temperatures
    dm_io,
    dm_top,
    dm_bot,
    dm_port,
    dQ_heating,
    res_dm,
    res_dQ,  # flows
    cp_T,
    lam_T,
    rho_T,
    ny_T,
    lam_mean,
    cp_port,
    lam_port_fld,  # mat. props.
    mcp,
    mcp_heated,
    rhocp,
    lam_wll,
    lam_ins,
    mcp_wll,
    ui,  # material properties.
    alpha_i,
    alpha_inf,  # alpha values
    UA_tb,
    UA_tb_wll,
    UA_amb_shell,
    UA_port,
    UA_port_wll,  # UA values
    port_own_idx,
    port_link_idx,
    heat_mult,  # indices
    grid_spacing,
    port_gsp,
    port_subs_gsp,
    d_i,
    cell_dist,  # lengths
    flow_length,
    r_total,
    r_ln_wll,
    r_ln_ins,
    r_rins,  # lengths
    A_cell,
    V_cell,
    A_shell_i,
    A_shell_ins,
    A_p_fld_mean,  # areas and vols
    process_flows,
    vertical,
    step_stable,  # bools
    part_id,
    stability_breaches,
    vN_max_step,
    max_factor,  # misc.
    stepnum,
    timestep,  # step information
    dT_cond,
    dT_adv,
    dT_heat,
    dT_heated,  # differentials
):

    # save rate of heat flow to result array
    if process_flows[0]:  # only if flows not already processed
        res_dQ[stepnum] = dQ_heating

    process_flows[0] = _process_flow_invar(
        process_flows=process_flows,
        dm_io=dm_io,
        dm_top=dm_top,
        dm_bot=dm_bot,
        dm_port=dm_port,
        stepnum=stepnum,
        res_dm=res_dm,
    )

    water_mat_props_ext_view(
        T_ext=T_ext, cp_T=cp_T, lam_T=lam_T, rho_T=rho_T, ny_T=ny_T
    )

    # get mean lambda value between cells:
    _lambda_mean_view(lam_T=lam_T, out=lam_mean)

    UA_plate_tb(
        A_cell=A_cell,
        grid_spacing=grid_spacing,
        lam_mean=lam_mean,
        UA_tb_wll=UA_tb_wll,
        out=UA_tb,
    )

    # for conduction between current cell and ambient:
    # get outer pipe (insulation) surface temperature using a linearized
    # approach assuming steady state (assuming surface temperature = const.
    # for t -> infinity) and for cylinder shell (lids are omitted)
    surface_temp_steady_state_inplace(
        T=T_ext[1:-1],
        T_inf=T_amb[0],
        A_s=A_shell_ins,
        alpha_inf=alpha_inf,
        UA=UA_amb_shell,
        T_s=T_s,
    )
    # get inner alpha value between fluid and wall from nusselt equations:
    pipe_alpha_i(
        dm_io, T_ext[1:-1], rho_T, ny_T, lam_T, A_cell, d_i, cell_dist, alpha_i
    )
    # get outer alpha value between insulation and surrounding air:
    cylinder_alpha_inf(  # for a cylinder
        T_s=T_s,
        T_inf=T_amb[0],
        flow_length=flow_length,
        vertical=vertical,
        r_total=r_total,
        alpha_inf=alpha_inf,
    )
    # get resulting UA to ambient:
    UA_fld_wll_ins_amb_cyl(
        A_i=A_shell_i,
        r_ln_wll=r_ln_wll,
        r_ln_ins=r_ln_ins,
        r_rins=r_rins,
        alpha_i=alpha_i,
        alpha_inf=alpha_inf,
        lam_wll=lam_wll,
        lam_ins=lam_ins,
        out=UA_amb_shell,
    )

    # precalculate values which are needed multiple times:
    cell_temp_props_ext(
        T_ext=T_ext,
        V_cell=V_cell,
        cp_T=cp_T,
        rho_T=rho_T,
        mcp_wll=mcp_wll,
        rhocp=rhocp,
        mcp=mcp,
        ui=ui,
    )

    dT_cond_port = _process_ports_collapsed(
        ports_all=ports_all,
        port_link_idx=port_link_idx,
        port_own_idx=port_own_idx,
        T=T_ext[1:-1],
        mcp=mcp,
        UA_port=UA_port,
        UA_port_wll=UA_port_wll,
        A_p_fld_mean=A_p_fld_mean,
        port_gsp=port_gsp,
        grid_spacing=grid_spacing,
        lam_T=lam_T,
        cp_port=cp_port,
        lam_port_fld=lam_port_fld,
        T_port=T_port,
    )

    step_stable, vN_max_step, max_factor = _vonNeumann_stability_invar(
        part_id=part_id,
        stability_breaches=stability_breaches,
        UA_tb=UA_tb,
        UA_port=UA_port,
        UA_amb_shell=UA_amb_shell,
        dm_io=dm_io,
        rho_T=rho_T,
        rhocp=rhocp,
        grid_spacing=grid_spacing,
        port_subs_gsp=port_subs_gsp,
        A_cell=A_cell,
        A_port=A_p_fld_mean,
        A_shell=A_shell_ins,
        r_total=r_total,
        V_cell=V_cell,
        step_stable=step_stable,
        vN_max_step=vN_max_step,
        max_factor=max_factor,
        stepnum=stepnum,
        timestep=timestep,
    )

    # CALCULATE DIFFERENTIALS
    # calculate heat transfer by internal heat sources
    dT_heated[:] = dQ_heating * heat_mult / mcp_heated

    # calculate heat transfer by conduction
    dT_cond[:] = (
        +UA_tb[:-1] * (T_ext[:-2] - T_ext[1:-1])
        + UA_tb[1:] * (T_ext[2:] - T_ext[1:-1])
        + UA_amb_shell * (T_amb[0] - T_ext[1:-1])
    ) / mcp
    # calculate heat transfer by advection
    dT_adv[:] = (
        +dm_top * (cp_T[:-2] * T_ext[:-2] - ui)
        + dm_bot * (cp_T[2:] * T_ext[2:] - ui)
    ) / mcp

    # sum up heat conduction and advection for port values:
    for i in range(port_own_idx.size):
        idx = port_own_idx[i]  # idx of port values at temperature/diff array
        # conduction
        dT_cond[idx] += dT_cond_port[i]
        # advection
        dT_adv[idx] += (
            dm_port[i] * (cp_port[i] * T_port[i] - ui[idx]) / mcp[idx]
        )

    dT_total = dT_cond + dT_adv + dT_heat

    return dT_total, process_flows, step_stable, vN_max_step, max_factor


@njit(nogil=GLOB_NOGIL, cache=True)
def hexnum_diff(
    T_ext,
    T_port,
    T_amb,
    ports_all,  # temperatures
    dm_io,
    dm_top,
    dm_bot,
    dm_port,
    res_dm,  # flows
    cp_T,
    lam_fld,
    rho_T,
    ny_T,
    lam_mean,
    cp_port,
    lam_port_fld,
    mcp,
    rhocp,
    cp_wll,
    lam_wll,
    ui,  # material properties.
    alpha_i,  # alpha values
    UA_dim1,
    UA_dim2,
    UA_dim1_wll,
    UA_port,
    UA_port_wll,  # UA values
    port_own_idx,
    port_link_idx,  # indices
    grid_spacing,
    port_gsp,
    port_subs_gsp,
    d_h,
    s_plate,
    cell_dist,
    dist_min,  # lengths
    A_channel,
    V_cell_fld,
    A_plate_eff,
    A_p_fld_mean,  # areas and vols
    channel_divisor,
    corr_Re,
    process_flows,
    step_stable,  # bools
    part_id,
    stability_breaches,
    vN_max_step,
    max_factor,  # misc.
    stepnum,  # step information
    dT_cond,
    dT_adv,
    dT_total,  # differentials
    timestep,
):

    # generate views needed to make calculations easier:
    T_sup = T_ext[1:-1, 1]  # view to supply side
    T_dmd = T_ext[1:-1, 3]  # view to demand side
    # T_wll = T_ext[1:-1, 2]  # view to wall temperature
    dm_sup = dm_io[:1]  # view to supply side massflow
    dm_dmd = dm_io[1:]  # view to demand side massflow

    process_flows[0] = _process_flow_multi_flow(
        process_flows=process_flows,
        dm_io=dm_io,
        dm_top=dm_top,
        dm_bot=dm_bot,
        dm_port=dm_port,
        stepnum=stepnum,
        res_dm=res_dm,
    )

    flow_per_channel = np.abs(dm_io / channel_divisor)

    water_mat_props_ext_view(  # only pass fluid columns to T_ext
        T_ext=T_ext[:, 1::2], cp_T=cp_T, lam_T=lam_fld, rho_T=rho_T, ny_T=ny_T
    )

    _lambda_mean_view(lam_T=lam_fld, out=lam_mean)

    UA_plate_tb_fld(  # only pass the fluid columns to out
        A_cell=A_channel,
        grid_spacing=grid_spacing,
        lam_mean=lam_mean,
        out=UA_dim1[:, ::2],
    )

    UA_plate_tb_wll(  # only pass the wall column to out
        UA_tb_wll=UA_dim1_wll, out=UA_dim1[:, 1]
    )

    phex_alpha_i_wll_sep_discretized(
        dm=dm_sup / channel_divisor[0],
        T_fld=T_sup,
        T_wll=T_sup,
        rho=rho_T[:, 0],
        ny=ny_T[:, 0],
        lam_fld=lam_fld[:, 0],
        A=A_channel,
        d_h=d_h,
        x=cell_dist,
        corr_Re=corr_Re,
        alpha=alpha_i[:, 0],
    )
    phex_alpha_i_wll_sep_discretized(
        dm=dm_dmd / channel_divisor[1],
        T_fld=T_dmd,
        T_wll=T_dmd,
        rho=rho_T[:, 1],
        ny=ny_T[:, 1],
        lam_fld=lam_fld[:, 1],
        A=A_channel,
        d_h=d_h,
        x=cell_dist,
        corr_Re=corr_Re,
        alpha=alpha_i[:, 1],
    )

    UA_dim2[:, 1:3] = UA_fld_wll_plate(
        A=A_plate_eff, s_wll=s_plate / 2, alpha_fld=alpha_i, lam_wll=lam_wll
    )

    cell_temp_props_fld(
        T_ext_fld=T_ext[:, 1::2],
        V_cell=V_cell_fld,
        cp_T=cp_T,
        rho_T=rho_T,
        rhocp_fld=rhocp[:, ::2],
        mcp_fld=mcp[:, ::2],
        ui_fld=ui[:, ::2],
    )

    specific_inner_energy_wll(T_wll=T_ext[1:-1, 2], cp_wll=cp_wll, ui=ui[:, 1])

    dT_cond_port = _process_ports_collapsed(
        ports_all=ports_all,
        port_link_idx=port_link_idx,
        port_own_idx=port_own_idx,
        T=T_ext[1:-1, 1:-1],
        mcp=mcp,
        UA_port=UA_port,
        UA_port_wll=UA_port_wll,
        A_p_fld_mean=A_p_fld_mean,
        port_gsp=port_gsp,
        grid_spacing=grid_spacing,
        lam_T=lam_fld,
        cp_port=cp_port,
        lam_port_fld=lam_port_fld,
        T_port=T_port,
    )

    (
        _step_stable,
        _vN_max_step,
        _max_factor,
    ) = _vonNeumann_stability_invar_hexnum(
        part_id=part_id,
        stability_breaches=stability_breaches,
        UA_dim1=UA_dim1,
        UA_dim2=UA_dim2,
        UA_port=UA_port,
        dm_io=flow_per_channel,
        rho_T=rho_T,
        rhocp=rhocp,
        grid_spacing=grid_spacing,
        port_subs_gsp=port_subs_gsp,
        A_channel=A_channel,
        A_plate_eff=A_plate_eff,
        A_port=A_p_fld_mean,
        V_cell=V_cell_fld,
        step_stable=step_stable,
        vN_max_step=vN_max_step,
        max_factor=max_factor,
        stepnum=stepnum,
        timestep=timestep,
    )

    UA_amb_shell = 0.0

    dT_cond[:] = (
        # heat conduction in first dimension (axis 0), top -> bottom:
        (
            +UA_dim1[:-1] * (T_ext[:-2, 1:-1] - T_ext[1:-1, 1:-1])
            # heat conduction in first dimension (axis 0), bottom -> top:
            + UA_dim1[1:] * (T_ext[2:, 1:-1] - T_ext[1:-1, 1:-1])
            # heat conduction in second dimension (axis 1), left -> right:
            + UA_dim2[:, :-1] * (T_ext[1:-1, :-2] - T_ext[1:-1, 1:-1])
            # heat conduction in second dimension (axis 1), right -> left:
            + UA_dim2[:, 1:] * (T_ext[1:-1, 2:] - T_ext[1:-1, 1:-1])
            # heat conduction to ambient (currently set to 0):
            + UA_amb_shell * (T_amb - T_ext[1:-1, 1:-1])
        )
        / mcp
    )
    # calculate heat transfer by advection in the fluid channels
    dT_adv[:, ::2] = (
        # advective heat transport (only axis 0), top -> bottom:
        (
            +dm_top * (cp_T[:-2] * T_ext[:-2, 1::2] - ui[:, ::2])
            # advective heat transport (only axis 0), bottom -> top:
            + dm_bot * (cp_T[2:] * T_ext[2:, 1::2] - ui[:, ::2])
        )
        / mcp[:, ::2]
    )

    # sum up heat conduction and advection for port values:
    for i in range(port_own_idx.size):
        idx = port_own_idx[i]  # idx of port values at temperature/diff array
        # conduction
        dT_cond.flat[idx] += dT_cond_port[i]
        # advection
        dT_adv.flat[idx] += (
            dm_port[i]
            * (cp_port[i] * T_port[i] - ui.flat[idx])
            / mcp.flat[idx]
        )

    # divide advective transfer by the number of channels:
    dT_adv[:, ::2] /= channel_divisor
    # sum up the differentials for conduction and advection
    dT_total[:] = dT_cond + dT_adv

    return dT_total


@nb.njit(cache=True)
def condensing_hex_solve(
    T,
    T_port,
    ports_all,
    res,
    res_dm,
    dm_io,
    dm_port,
    port_own_idx,
    port_link_idx,
    X_pred,
    flow_scaling,
    water_dm_range,
    gas_dv_range,
    int_comb_idx,
    nvars_per_ftr,
    pca_mean,
    pca_components,
    lm_intercept,
    lm_coef,
    stepnum,
):
    """
    Calculate condensing flue gas HEX by using a PCA-transformed polynome LR.

    Parameters
    ----------
    T : TYPE
        DESCRIPTION.
    T_port : TYPE
        DESCRIPTION.
    ports_all : TYPE
        DESCRIPTION.
    res : TYPE
        DESCRIPTION.
    res_dm : TYPE
        DESCRIPTION.
    dm_io : TYPE
        DESCRIPTION.
    dm_port : TYPE
        DESCRIPTION.
    port_own_idx : TYPE
        DESCRIPTION.
    port_link_idx : TYPE
        DESCRIPTION.
    X_pred : TYPE
        DESCRIPTION.
    flow_scaling : TYPE
        DESCRIPTION.
    water_dm_range : TYPE
        DESCRIPTION.
    gas_dv_range : TYPE
        DESCRIPTION.
    int_comb_idx : TYPE
        DESCRIPTION.
    nvars_per_ftr : TYPE
        DESCRIPTION.
    pca_mean : TYPE
        DESCRIPTION.
    pca_components : TYPE
        DESCRIPTION.
    lm_intercept : TYPE
        DESCRIPTION.
    lm_coef : TYPE
        DESCRIPTION.
    stepnum : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    _port_values_to_cont(
        ports_all=ports_all, port_link_idx=port_link_idx, out=T_port
    )

    # extract inflowing temperatures for water (idx 0) and flue gas (idx 2)
    X_pred[:, :2] = T_port[::2]
    # extract water massflow (cell 3) and flue gas volume flow (cell 4) and
    # scale with scaling factors
    X_pred[:, 2:] = dm_io / flow_scaling
    # make some flow checks:
    # check for water/flue gas massflow bounds. only do something if violated
    bypass = False  # bypass is initialized to False
    if (water_dm_range[0] != 0.0) and (X_pred[0, 2] < water_dm_range[0]):
        # if flow smaller than lower bound, decide if using 0 or lower bound
        # by rounding to the closer value:
        X_pred[0, 2] = (
            round(X_pred[0, 2] / water_dm_range[0]) * water_dm_range[0]
        )
    elif X_pred[0, 2] > water_dm_range[1]:
        # bypassing excess mass flow, to avoid huge power output
        # when outside HEX heat meters are calculated with unclipped flows:
        # backup full flow for calculations:
        # water_dm_full = X_pred[0, 2]  # not needed anymore
        # get excess flow over max. range. this amount is bypassed
        water_dm_excess = X_pred[0, 2] - water_dm_range[1]
        bypass = True  # set bypassing to true
        # clip the amount of water over the hex to the range
        X_pred[0, 2] = water_dm_range[1]
    if (gas_dv_range[0] != 0.0) and (X_pred[0, 3] < gas_dv_range[0]):
        # if flow smaller than lower bound, decide if using 0 or lower bound
        # by rounding to the closer value:
        X_pred[0, 3] = round(X_pred[0, 3] / gas_dv_range[0]) * gas_dv_range[0]
    elif X_pred[0, 3] > gas_dv_range[1]:
        print(
            '\nFluegas volume flow in condensing HEX exceeded. The '
            'following value was encountered:'
        )
        print(X_pred[0, 3])
        raise ValueError

    # calculate results. but only if NO massflow is 0
    if np.all(X_pred[0, 2:] != 0):
        dm_water_thresh = 0.1  # threshhold below which no regr. preds. exist
        n_samples = 1  # this is always one for this function
        # only if water massflow greater 10%, else quad polynome
        if X_pred[0, 2] > dm_water_thresh:
            # transform input data to polynome, then to principal components
            X_pf = transform_to_poly_nb(
                X_pred, int_comb_idx, nvars_per_ftr, n_samples
            )
            X_PC = transform_pca_nb(X_pf, pca_mean, pca_components)
            # predict
            T_pred = poly_tranfs_pred(X_PC, lm_intercept, lm_coef)
            # save results to temperature array
            T[0, 0] = X_pred[0, 0]  # t w in
            T[1, 0] = T_pred[0, 0]  # t w out
            T[0, 1] = X_pred[0, 1]  # t fg in
            T[1, 1] = T_pred[0, 1]  # t fg out
        else:  # for massflow below thresh, use quad polynome
            T_pred_below_thresh = condensing_hex_quad_poly(
                X_pred,  # X vector
                int_comb_idx,
                nvars_per_ftr,  # polynomial transf.
                pca_mean,
                pca_components,  # PCA transformation
                lm_intercept,
                lm_coef,  # linear model transformation
                dm_water_thresh=0.1,
                dx=0.01,
            )
            # save results to temperature array
            T[0, 0] = X_pred[0, 0]  # t w in
            T[1, 0] = T_pred_below_thresh[0, 0]  # t w out
            T[0, 1] = X_pred[0, 1]  # t fg in
            T[1, 1] = T_pred_below_thresh[0, 1]  # t fg out

    else:  # if ANY massflow is 0, all output temps are equal to input temps
        T[:, 0] = X_pred[0, 0]  # t w in & t w out = t w in
        T[:, 1] = X_pred[0, 1]  # t fg in & t fg out = t fg in

    # if bypassing the hex with a part of the water flow:
    if bypass:
        # get heat capacity rates for bypassing, hex traversing and
        # outflowing (mixed) water. hcr is dimensionless, since flows have been
        # scaled before, thus value is same as leveraged Cp in unit J/kg/K
        hcr_bypass = cp_water(X_pred[0, 0]) * water_dm_excess
        hcr_hex_out = cp_water(T[1, 0]) * water_dm_range[1]
        hcr_out = hcr_bypass + hcr_hex_out
        # calculate outflowing (mix of bypass and hex traversing water) temp:
        T_out = (hcr_bypass * X_pred[0, 0] + hcr_hex_out * T[1, 0]) / hcr_out
        # set to the temperature result array:
        T[1, 0] = T_out  # t w out

    res[stepnum[0]] = T
    res_dm[stepnum[0]] = dm_io


# %% Simulation Env. implicit/explicit specific functions and tests:
@njit(nogil=GLOB_NOGIL, cache=True)
def hexnum_diff_impl(
    T_ext,
    T_port,
    T_amb,
    ports_all,  # temperatures
    dm_io,
    dm_top,
    dm_bot,
    dm_port,
    res_dm,  # flows
    cp_T,
    lam_fld,
    rho_T,
    ny_T,
    lam_mean,
    cp_port,
    lam_port_fld,
    mcp,
    rhocp,
    cp_wll,
    lam_wll,
    ui,  # material properties.
    alpha_i,  # alpha values
    UA_dim1,
    UA_dim2,
    UA_dim1_wll,
    UA_port,
    UA_port_wll,  # UA values
    port_own_idx,
    port_link_idx,  # indices
    grid_spacing,
    port_gsp,
    d_h,
    s_plate,
    cell_dist,
    dist_min,  # lengths
    A_channel,
    V_cell_fld,
    A_plate_eff,
    A_p_fld_mean,  # areas and vols
    channel_divisor,
    corr_Re,
    process_flows,  # bools
    stepnum,  # step information
    dT_cond,
    dT_adv,
    dT_total,  # differentials
):

    # generate views needed to make calculations easier:
    T_sup = T_ext[1:-1, 1]  # view to supply side
    T_dmd = T_ext[1:-1, 3]  # view to demand side
    # T_wll = T_ext[1:-1, 2]  # view to wall temperature
    dm_sup = dm_io[:1]  # view to supply side massflow
    dm_dmd = dm_io[1:]  # view to demand side massflow

    process_flows[0] = _process_flow_multi_flow(
        process_flows=process_flows,
        dm_io=dm_io,
        dm_top=dm_top,
        dm_bot=dm_bot,
        dm_port=dm_port,
        stepnum=stepnum,
        res_dm=res_dm,
    )

    water_mat_props_ext_view(  # only pass fluid columns to T_ext
        T_ext=T_ext[:, 1::2], cp_T=cp_T, lam_T=lam_fld, rho_T=rho_T, ny_T=ny_T
    )

    _lambda_mean_view(lam_T=lam_fld, out=lam_mean)

    UA_plate_tb_fld(  # only pass the fluid columns to out
        A_cell=A_channel,
        grid_spacing=grid_spacing,
        lam_mean=lam_mean,
        out=UA_dim1[:, ::2],
    )

    UA_plate_tb_wll(  # only pass the wall column to out
        UA_tb_wll=UA_dim1_wll, out=UA_dim1[:, 1]
    )

    phex_alpha_i_wll_sep_discretized(
        dm=dm_sup / channel_divisor[0],
        T_fld=T_sup,
        T_wll=T_sup,
        rho=rho_T[:, 0],
        ny=ny_T[:, 0],
        lam_fld=lam_fld[:, 0],
        A=A_channel,
        d_h=d_h,
        x=cell_dist,
        corr_Re=corr_Re,
        alpha=alpha_i[:, 0],
    )
    phex_alpha_i_wll_sep_discretized(
        dm=dm_dmd / channel_divisor[1],
        T_fld=T_dmd,
        T_wll=T_dmd,
        rho=rho_T[:, 1],
        ny=ny_T[:, 1],
        lam_fld=lam_fld[:, 1],
        A=A_channel,
        d_h=d_h,
        x=cell_dist,
        corr_Re=corr_Re,
        alpha=alpha_i[:, 1],
    )

    UA_dim2[:, 1:3] = UA_fld_wll_plate(
        A=A_plate_eff, s_wll=s_plate / 2, alpha_fld=alpha_i, lam_wll=lam_wll
    )

    cell_temp_props_fld(
        T_ext_fld=T_ext[:, 1::2],
        V_cell=V_cell_fld,
        cp_T=cp_T,
        rho_T=rho_T,
        rhocp_fld=rhocp[:, ::2],
        mcp_fld=mcp[:, ::2],
        ui_fld=ui[:, ::2],
    )

    specific_inner_energy_wll(T_wll=T_ext[1:-1, 2], cp_wll=cp_wll, ui=ui[:, 1])

    dT_cond_port = _process_ports_collapsed(
        ports_all=ports_all,
        port_link_idx=port_link_idx,
        port_own_idx=port_own_idx,
        T=T_ext[1:-1, 1:-1],
        mcp=mcp,
        UA_port=UA_port,
        UA_port_wll=UA_port_wll,
        A_p_fld_mean=A_p_fld_mean,
        port_gsp=port_gsp,
        grid_spacing=grid_spacing,
        lam_T=lam_fld,
        cp_port=cp_port,
        lam_port_fld=lam_port_fld,
        T_port=T_port,
    )

    UA_amb_shell = 0.0

    dT_cond[:] = (
        # heat conduction in first dimension (axis 0), top -> bottom:
        (
            +UA_dim1[:-1] * (T_ext[:-2, 1:-1] - T_ext[1:-1, 1:-1])
            # heat conduction in first dimension (axis 0), bottom -> top:
            + UA_dim1[1:] * (T_ext[2:, 1:-1] - T_ext[1:-1, 1:-1])
            # heat conduction in second dimension (axis 1), left -> right:
            + UA_dim2[:, :-1] * (T_ext[1:-1, :-2] - T_ext[1:-1, 1:-1])
            # heat conduction in second dimension (axis 1), right -> left:
            + UA_dim2[:, 1:] * (T_ext[1:-1, 2:] - T_ext[1:-1, 1:-1])
            # heat conduction to ambient (currently set to 0):
            + UA_amb_shell * (T_amb - T_ext[1:-1, 1:-1])
        )
        / mcp
    )
    # calculate heat transfer by advection in the fluid channels
    dT_adv[:, ::2] = (
        # advective heat transport (only axis 0), top -> bottom:
        (
            +dm_top * (cp_T[:-2] * T_ext[:-2, 1::2] - ui[:, ::2])
            # advective heat transport (only axis 0), bottom -> top:
            + dm_bot * (cp_T[2:] * T_ext[2:, 1::2] - ui[:, ::2])
        )
        / mcp[:, ::2]
    )

    # sum up heat conduction and advection for port values:
    for i in range(port_own_idx.size):
        idx = port_own_idx[i]  # idx of port values at temperature/diff array
        # conduction
        dT_cond.flat[idx] += dT_cond_port[i]
        # advection
        dT_adv.flat[idx] += (
            dm_port[i]
            * (cp_port[i] * T_port[i] - ui.flat[idx])
            / mcp.flat[idx]
        )

    # divide advective transfer by the number of channels:
    dT_adv[:, ::2] /= channel_divisor
    # sum up the differentials for conduction and advection
    dT_total[:] = dT_cond + dT_adv

    return dT_total


@nb.njit(cache=True, nogil=GLOB_NOGIL)
def euler_forward(diff, diff_input_args, yprev, _h):
    return yprev + _h * diff(*diff_input_args)


@nb.njit(cache=True, nogil=GLOB_NOGIL)
def hexnum_imp_root_diff(y, yprev, h, input_args):
    input_args[0][1:-1, 1:-1] = y.reshape(input_args[0][1:-1, 1:-1].shape)
    return y - yprev - h * hexnum_diff_impl(*input_args).ravel()


@nb.njit
def hexnum_imp_fixedpoint(y, y_prev, h, input_args):  # fixed point function
    """
    Find fixed point of the hexnum implicit function.

    Warning: Fixed point iteration may be several"""
    input_args[0][1:-1, 1:-1] = y.reshape(input_args[0][1:-1, 1:-1].shape)
    return (
        y_prev.reshape(input_args[0][1:-1, 1:-1].shape)
        + h * hexnum_diff_impl(*input_args)
    ).ravel()


@nb.njit
def fixed_point_to_root(y, fp_fun, y_prev, h, input_args):
    return y - fp_fun(y, y_prev, h, input_args)


@nb.njit(cache=True, nogil=GLOB_NOGIL)
def hexnum_imp_fixedp_diff(y, yprev, h, input_args):
    input_args[0][1:-1, 1:-1] = y
    return yprev + h * hexnum_diff_impl(*input_args)


# @nb.njit(cache=True, nogil=GLOB_NOGIL)
def hexnum_imp_newt_diff(yprev, _h, input_args, rtol=1e-6):
    """
    https://math.stackexchange.com/questions/152159/how-to-correctly-apply-newton-raphson-method-to-backward-euler-method
    https://scicomp.stackexchange.com/questions/5042/how-to-implement-newtons-method-for-solving-the-algebraic-equations-in-the-back
    """
    input_args[0][1:-1, 1:-1] = yprev
    y_lastiter = yprev.copy()

    err = 1.0

    # initial guess:
    diff = hexnum_diff_impl(*input_args)
    y = yprev + _h * diff
    f = np.zeros_like(y)

    while np.any(err > rtol):
        #        y_lastiter = y.copy()
        input_args[0][1:-1, 1:-1] = y
        #        y = (
        #            y_lastiter
        #            + (euler_forward(hexnum_diff_impl, input_args, yprev, _h)
        #               / hexnum_diff_impl(*input_args))
        #        )
        diff = hexnum_diff_impl(*input_args)
        f_lastiter = f
        f = y - yprev - _h * diff
        nz = f != 0.0  # make a mask with non zero values
        slope = (f[nz] - f_lastiter[nz]) / (y[nz] - y_lastiter[nz])
        #        diff[diff == 0.] = yprev[diff == 0.]
        #        diff2 = diff * _h
        #        y = y_lastiter - ((yprev + diff) / (diff))
        #        y[np.abs(y) == np.inf] = y_lastiter[np.abs(y) == np.inf]
        #        y = y_lastiter - yprev / diff - 1.
        #        err = np.sqrt(np.sum((np.abs(y - y_lastiter))**2))
        #        err = (y - y_lastiter) / y_lastiter

        y[nz] = y_lastiter[nz] - f[nz] / slope
        err = (y - y_lastiter) / y_lastiter
        y_lastiter = y.copy()

    return y


# %% Simulation Env. old (mostly deprecated) solve methods:
@njit(nogil=GLOB_NOGIL, cache=True)
def solve_connector_3w_overload(arglist):
    solve_connector_3w(*arglist)


@nb.njit(nogil=GLOB_NOGIL, cache=True)
def solve_connector_3w(T, ports_all, cp_T, dm, port_link_idx, res, stepnum):
    # depending on the flow conditions this 3w connector acts as a flow
    # mixing or splitting device. This state has to be determined by
    # checking the direction of the massflows through the ports.
    # A negative sign means that the massflow is exiting through the
    # respective port, a positive sign is an ingoing massflow.

    # get connected port temperatures:
    # get port array:
    _port_values_to_cont(
        ports_all=ports_all, port_link_idx=port_link_idx, out=T
    )
    # get cp-values of all temperatures:
    get_cp_water(T, cp_T)

    # save bool indices of massflows greater (in) and less (out) than 0:
    # (using dm as massflow array only works since it is a view of _dm_io!)
    dm_in = np.greater(dm, 0)
    dm_out = np.less(dm, 0)

    # if 2 ports > 0 are True, 3w connector is mixer:
    if np.sum(dm_in) == 2:
        # get cp of outflowing massflow (error of mean temp is <<0.5% compared
        # to a heat cap. ratio calculation, thus negligible and ok):
        cp_out = cp_water(np.sum(T[dm_in]) / 2)
        # calc T_out by mixing the inflowing massflows (*-1 since outgoing
        # massflows have a negative sign):
        T_out = np.sum(dm[dm_in] * cp_T[dm_in] * T[dm_in]) / (
            cp_out * -1 * dm[dm_out]
        )
        # pass on port values by switching temperatures:
        # set old T_out to both in-ports
        T[dm_in] = T[dm_out]
        # set calculated T_out to out-port
        T[dm_out] = T_out
    # if 2 ports < 0 are True, 3w connector is splitter:
    elif np.sum(dm_out) == 2:
        # no real calculation has to be done here, just switching
        # temperatures and passing them on to opposite ports
        # calc the temp which will be shown at the inflowing port as a mean
        # of the temps of outflowing ports (at in port connected part will
        # see a mean value of both temps for heat conduction):
        T_in = T[dm_out].sum() / 2
        # pass inflowing temp to outflowing ports:
        T[dm_out] = T[dm_in]
        # pass mean out temp to in port:
        T[dm_in] = T_in
    # if one port has 0 massflow, sum of dm_in == 1:
    elif np.sum(dm_in) == 1:
        # get port with 0 massflow:
        dm0 = np.equal(dm, 0)
        # this port 'sees' a mean of the other two temperatures:
        T[dm0] = T[~dm0].sum() / 2
        # the out ports heat flow is dominated by convection, thus it
        # only 'sees' the in flow temperature but not the 0 flow temp:
        T[dm_out] = T[dm_in]
        # the in ports heat flow is also dominated by convection, but here
        # it is easy to implement the 0-flow port influence, since heat
        # flow by convection of part connected to in port is not affected
        # by connected temperature, thus also get a mean value:
        T[dm_in] = T[~dm_in].sum() / 2
    # if all ports have 0 massflow:
    else:
        # here all ports see a mean of the other ports:
        # bkp 2 ports
        T0 = (T[1] + T[2]) / 2
        T1 = (T[0] + T[2]) / 2
        # save means to port values:
        T[2] = (T[0] + T[1]) / 2
        T[0] = T0
        T[1] = T1

    # save results:
    res[stepnum[0]] = T


# #@nb.njit((float64[:,:], float64[:,:], float64[:], float64[:], float64[:,:,:],
#         float64[:,:], float64[:,:],
#         float64[:], float64[:], float64[:], float64[:], float64[:,:],
#         float64[:], float64[:], float64[:],
#         int32[:], int32[:],
#         float64, float64, float64, float64, float64[:],
#         int32, int32, int32, float64, int32))
@nb.njit(nogil=GLOB_NOGIL, cache=True)
def solve_platehex(
    T,
    T_port,
    T_mean,
    ports_all,
    res,
    dm_io,
    dm_port,
    cp_mean,
    lam_mean,
    rho_mean,
    ny_mean,
    lam_wll,
    alpha_i,
    UA_fld_wll,
    UA_fld_wll_fld,
    port_own_idx,
    port_link_idx,
    A_plate_eff,
    A_channel,
    d_h,
    s_plate,
    cell_dist,
    num_A,
    num_channels_sup,
    num_channels_dmd,
    corr_Re,
    stepnum,
):

    # get temperatures of connected ports:
    _port_values_to_cont(
        ports_all=ports_all, port_link_idx=port_link_idx, out=T_port
    )

    # get massflows
    # only positive flows for side supply and set entry temperature
    # depending on flow direction:
    if dm_io[0] >= 0:
        dm_sup = dm_io[0]  # positive sup side flow
        T_sup_in = T_port[0]  # entry temp. sup is sup_in
        T_sup_out = T_port[1]  # out temp. sup is sup_out
        dm_port[0] = dm_io[0]
        dm_port[1] = 0.0
    else:
        dm_sup = -dm_io[0]  # positive sup side flow
        T_sup_in = T_port[1]  # entry temp. sup is sup_out
        T_sup_out = T_port[0]  # out temp. sup is sup_in
        dm_port[0] = 0.0
        dm_port[1] = -dm_io[0]
    # only positive flows for side demand and set entry temperature
    # depending on flow direction:
    if dm_io[1] >= 0:
        dm_dmd = dm_io[1]  # positive dmd side flow
        T_dmd_in = T_port[2]  # entry temp. dmd is dmd_in
        T_dmd_out = T_port[3]  # out temp. dmd is dmd_out
        dm_port[2] = dm_io[1]
        dm_port[3] = 0.0
    else:
        dm_dmd = -dm_io[1]  # positive dmd side flow
        T_dmd_in = T_port[3]  # entry temp. dmd is dmd_out
        T_dmd_out = T_port[2]  # out temp. dmd is dmd_in
        dm_port[2] = 0.0
        dm_port[3] = -dm_io[1]

    # do all the calculations only if both massflows are not 0
    if dm_sup != 0 and dm_dmd != 0:
        # get mean temperature of both fluid sides as a mean of the neighboring
        # port temperatures which is a good approximation when there is a flow
        # through the HEX (without flow no calc. will be done anyways):
        T_mean[0] = (T_sup_in + T_sup_out) / 2  # sup side
        T_mean[1] = (T_dmd_in + T_dmd_out) / 2  # dmd side

        # get thermodynamic properties of water
        # for mean cell temp:
        water_mat_props(T_mean, cp_mean, lam_mean, rho_mean, ny_mean)

        # for conduction between fluid cells and wall:
        # get inner alpha value between fluid and wall from nusselt equations:
        # supply side:
        phex_alpha_i_wll_sep(
            dm_sup / num_channels_sup,
            T_mean[0],
            T_mean[0],
            rho_mean[0],
            ny_mean[0],
            lam_mean[0],
            A_channel,
            d_h,
            cell_dist,
            corr_Re,
            alpha_i[0:1],
        )
        # demand side:
        phex_alpha_i_wll_sep(
            dm_dmd / num_channels_dmd,
            T_mean[1],
            T_mean[1],
            rho_mean[1],
            ny_mean[1],
            lam_mean[1],
            A_channel,
            d_h,
            cell_dist,
            corr_Re,
            alpha_i[1:2],
        )
        # get resulting UA from both fluid sides, assuming same values in all
        # channels of one pass, to the midpoint (-> /2) of the separating wall.
        # index [1, 1] for lam_wll selects own lam_wll to avoid overwriting by
        # _get_port_connections method of simenv.
        UA_fld_wll[:] = UA_fld_wll_plate(
            A_plate_eff, s_plate / 2, alpha_i, lam_wll[0]
        )

        # get total UA value from fluid to fluid (in VDI Wärmeatlas this is kA)
        # by calculating the series circuit of the UA fluid wall values with
        # the number of effective heat transfer areas (num plates - 2)
        UA_fld_wll_fld[0] = (
            series_circuit_UA(UA_fld_wll[0], UA_fld_wll[1]) * num_A
        )

        # Heat exchanger dimensionless coefficients:
        # heat capacity flows (ok, this is not dimensionless...)
        dC_sup = dm_sup * cp_mean[0]
        dC_dmd = dm_dmd * cp_mean[1]
        # calculate NTU value of the supply side:
        if dC_sup != 0:
            NTU_sup = UA_fld_wll_fld[0] / dC_sup
        else:
            NTU_sup = np.inf
        # calculate heat capacity flow ratio for the supply to demand side:
        if dC_dmd != 0:
            R_sup = dC_sup / dC_dmd
        else:
            R_sup = np.inf

        # get dimensionless change in temperature
        rs_ntus = (R_sup - 1) * NTU_sup  # precalc. for speed
        # for the supply side
        if (
            R_sup != 1 and rs_ntus < 100  # heat cap flow ratio not 1 and valid
        ):  # range for exp
            P_sup = (1 - np.exp(rs_ntus)) / (1 - R_sup * np.exp(rs_ntus))
        elif rs_ntus > 100:  # if exp in not-defined range
            P_sup = 1 / R_sup  # largely only depending on 1/R
            # above a specific value. for float64 everything above around
            # 50 to 100 is cut of due to float precision and quite exactly
            # equal 1/R.
        else:  # heat cap flow ratio equal 1
            P_sup = NTU_sup / (1 + NTU_sup)
        # for the demand side:
        P_dmd = P_sup * R_sup
        # if P_sup has a NaN value, for example when a flow is zero or very
        # close to zero (NaN check is: Number is not equal to itself!):
        if P_sup != P_sup:
            P_sup = 0
            P_dmd = 0

        # calculate supply and demand outlet temperatures from this and
        # overwrite the estimate value taken from ports:
        T_sup_out = T_sup_in - P_sup * (  # supply side outlet temp.
            T_sup_in - T_dmd_in
        )
        T_dmd_out = T_dmd_in + P_dmd * (  # demand side outlet temp.
            T_sup_in - T_dmd_in
        )
        # calculate heat flow from supply fluid to wall and demand fluid:
        # dQ = dC_sup * (T_sup_in - T_sup_out)

    else:
        # else if at least one side is zero.
        # fill with the values of connected ports where the flow is 0 (this
        # is already done automatically in the beginning where temperature
        # values are set depending on the flow direction, so do nothing
        # for zero flow).
        # pass on the value where the flow is not 0.
        if dm_sup != 0:  # check supply side for flow not zero
            T_sup_out = T_sup_in  # pass on if sup flow not 0
        elif dm_dmd != 0:  # if sup flow not zero
            T_dmd_out = T_dmd_in  # pass on if dmd flow not 0

    # set new values to array for port interaction with other parts,
    # depending on flow direction:
    if dm_io[0] >= 0:  # sup side normal flow
        T[0] = T_sup_in  # - 273.15
        T[1] = T_sup_out  # - 273.15
    else:  # sup side inversed flow
        T[1] = T_sup_in  # - 273.15
        T[0] = T_sup_out  # - 273.15
    # only positive flows for side demand and set entry temperature
    # depending on flow direction:
    if dm_io[1] >= 0:  # dmd side normal flow
        T[2] = T_dmd_in  # - 273.15
        T[3] = T_dmd_out  # - 273.15
    else:  # dmd side inversed flow
        T[3] = T_dmd_in  # - 273.15
        T[2] = T_dmd_out  # - 273.15

    # save results:
    res[stepnum[0]] = T

    # dT_cond[1, 1] = 0


@jit(
    (float64[:], int32[:], float64[:]),
    nopython=True,
    nogil=GLOB_NOGIL,
    cache=True,
)  # parallel=GLOB_PARALLEL useful
def _get_p_arr_pump(ports_all, port_link_idx, T):
    """
    Values of requested ports are saved to temperature array.
    """

    T[:] = ports_all[port_link_idx][::-1]


@njit(nogil=GLOB_NOGIL, cache=True)
def solve_mix_overload(arglist):
    solve_mix(*arglist)


@nb.jit(
    nopython=True, nogil=GLOB_NOGIL, cache=True
)  # parallel=GLOB_PARALLEL useful
def solve_mix(port_array, _port_link_idx, dm_io, T):
    # get port array:
    T[:] = port_array[_port_link_idx]

    # calc T_out by mixing A and B if there is a flow through the valve
    if dm_io[2] != 0:
        # get outlet temperature as mean of both inlet temperatures for cp
        # calculation:
        T[2] = (T[0] + T[1]) / 2
        # get heat capacities:
        cp = cp_water(T)
        # get outlet temperature by mixing the massflows:
        T_AB = (dm_io[0] * cp[0] * T[0] + dm_io[1] * cp[1] * T[1]) / (
            dm_io[2] * cp[2]
        )
        # set mean outlet temp. to both in-ports for heat conduction
        T[0:2] = T[2]
        # set calculated T_out to out-port
        T[2] = T_AB
    else:
        # else if dm of AB port is zero, the temperatures all are a mean of
        # the other ports temperatures to enable heat calculation:
        T_AB = (T[0] + T[1]) / 2
        T_A = (T[1] + T[2]) / 2
        T_B = (T[0] + T[2]) / 2
        # set to temperature array:
        T[0] = T_A
        T[1] = T_B
        T[2] = T_AB


@njit(nogil=GLOB_NOGIL, cache=True)
def solve_split_overload(arglist):
    solve_split(*arglist)


@nb.jit(
    (float64[:], int32[:], float64[:]),
    nopython=True,
    nogil=GLOB_NOGIL,
    cache=True,
)
def solve_split(port_array, _port_link_idx, T):
    T[:] = port_array[_port_link_idx]
    T_in = T[0:2].sum() / 2
    T[0:2] = T[2]
    T[2] = T_in


@njit(nogil=GLOB_NOGIL, cache=True)
def solve_pump_overload(arglist):
    solve_pump(*arglist)


@nb.njit(nogil=GLOB_NOGIL, cache=True)  # parallel=GLOB_PARALLEL useful
def solve_pump(ports_all, port_link_idx, T, res, res_dm, dm, stepnum):
    """
    Solve method of part pump.
    """

    # get and invert temperatures
    _get_p_arr_pump(ports_all=ports_all, port_link_idx=port_link_idx, T=T)
    # save massflow to massflow result grid
    res_dm[stepnum[0], 0] = dm[0]
    # save temperatures to temperature result grid
    res[stepnum[0]] = T


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def ctrl_deadtime(
    deadtime,
    timestep,
    dt_arr,
    pv_arr,
    len_dt_arr,
    dt_idx,
    last_dt_idx,
    delayed_pv,
    sp,
    pv,
):
    dt_arr += timestep
    # last deadtime index is saved for interpolation if in last
    # step a new pv was found, otherwise last deadtime index will
    # be increased by one to include roll by one element:
    if dt_idx != -1:
        last_dt_idx = dt_idx
    else:
        last_dt_idx += 1
    # reset deadtime index with a value which will be kept if no
    # new pv value reached (so the old one will be kept):
    dt_idx = -1
    # loop through deadtime array
    for i in range(len_dt_arr):
        # if time in deadtime array is equal or greater deadtime
        # return index of the position (only the first occurrence
        # will be found!)
        if dt_arr[i] >= deadtime:
            dt_idx = i
            break
    # calculate delayed pv (will not be overwritten after calc.
    # until next step, thus can be reused if no new value is found)
    # if a new value has reached deadtime delay in only one step:
    if dt_idx == 0:
        # interpolate delayed pv from previous pv, new pv and
        # expired time and time between prev. and new pv:
        delayed_pv = delayed_pv + (pv_arr[0] - delayed_pv) / (deadtime) * (
            dt_arr[0]
        )
    # if a new value has reached deadtime delay after more than
    # one step:
    elif dt_idx > 0:
        # if deadtime is hit exactly (for example with constant
        # timesteps):
        if dt_arr[dt_idx] == deadtime:
            delayed_pv = pv_arr[dt_idx]
        else:
            # interpolate value if deadtime is overshot andnot hit:
            delayed_pv = pv_arr[dt_idx - 1] + (
                pv_arr[dt_idx] - pv_arr[dt_idx - 1]
            ) / (dt_arr[dt_idx] - dt_arr[dt_idx - 1]) * (
                deadtime - dt_arr[dt_idx - 1]
            )
    # if deadtime delay was not reached:
    else:
        # interpolate delayed pv from previous pv, next pv and
        # expired time and time till next pv:
        delayed_pv = delayed_pv + (pv_arr[last_dt_idx] - delayed_pv) / (
            deadtime - (dt_arr[last_dt_idx] - timestep)
        ) * (timestep)

    # calculate error from delayed pv_value (delayed pv will not
    # be overwritten until next step):
    error = sp[0] - delayed_pv
    # set all time values in deadtime array after found value to 0:
    dt_arr[dt_idx:] = 0
    # roll deadtime and pv array one step backwards:
    dt_arr[1:] = dt_arr[0:-1]
    pv_arr[1:] = pv_arr[0:-1]
    # insert current pv into first slot of pv_arr:
    pv_arr[0] = pv[0]
    # set expired time of current pv to zero:
    dt_arr[0] = 0

    return error, delayed_pv, dt_idx, last_dt_idx


@nb.njit(nogil=GLOB_NOGIL, cache=True)
def _heun_corrector_adapt(
    res,
    T,
    df0,
    df1,
    trnc_err_cell_weight,
    _h,
    stepnum,
    rtol,
    atol,
    err,
    new_trnc_err,
):
    # solve heun method and save to result:
    res[stepnum] = res[stepnum - 1] + (_h / 2) * (df0 + df1)
    # GET TRUCATION ERROR FOR HEUN COMPARED WITH LOWER ORDER EULER
    # TO CALC. NEW STEPSIZE:
    # truncation error approximation is the difference of the total
    # heun result and the euler (predictor) result saved in T. The
    # trunc. error is calculated by taking the root mean square
    # norm of the differences for each part. This applies a root
    # mean square error weighting over the cells.
    # To get the systems truncation error, the norms have to be
    # added up by taking the root of the sum of the squares.

    # get each part's local relative error as euclidean matrix norm
    # (sqrt not yet taken to enable summing up the part's errors)
    # weighted by the relative and absolute tolerance. tolerance weighting as
    # in:
    # https://github.com/scipy/scipy/blob/ ...
    # 19acfed431060aafaa963f7e530c95e70cd4b85c/scipy/integrate/_ivp/rk.py#L147
    trnc_err = (
        (
            (res[stepnum] - T)
            * trnc_err_cell_weight
            / (np.maximum(res[stepnum - 1], res[stepnum]) * rtol + atol)
        )
        ** 2
    ).sum()
    # sum these local relative errors up for all parts:
    err += trnc_err
    # now get root mean square error for part by dividing part's
    # trnc_err by its amount of cells and taking the root:
    new_trnc_err = (trnc_err / T.size) ** 0.5

    # now also save to T arrays to be able to easily use
    # memoryviews in diff functions:
    T[:] = res[stepnum]

    return err, new_trnc_err


# @nb.njit(nogil=GLOB_NOGIL, cache=True)
def _embedded_adapt_stepsize(
    err,
    sys_trnc_err,
    num_cells_tot_nmrc,
    step_accepted,
    failed_steps,
    safety,
    order,
    solver_state,
    min_factor,
    max_factor,
    min_stepsize,
    max_stepsize,
    ports_all,
    parr_bkp,
    vN_max_step,
    step_stable,
    cnt_instable,
    timeframe,
    _h,
    timestep,
    stepnum,
):
    # ADAPTIVE TIMESTEP CALCULATION:
    # get all part's RMS error by dividing err by the amount of all
    # cells in the system and taking the root:
    err_rms = (err / num_cells_tot_nmrc) ** 0.5
    # save to array to enable stepwise system error lookup:
    sys_trnc_err[stepnum] = err_rms

    # check for good timesteps:
    # err_rms already has the relative and absolute tolerance included,
    # thus only checking against its value:
    if err_rms < 1:
        # error is lower than tolerance, thus step is accepted.
        step_accepted = True
        # save successful timestep to simulation environment:
        timestep = _h
        # get new timestep (err_rms is inverted thus negative power):
        _h *= min(
            max_factor[0], max(1, (safety * err_rms ** (-1 / (order + 1))))
        )
        # check if step is not above max step:
        if _h > max_stepsize:
            _h = max_stepsize  # reduce to max stepsize
            # save to state that max stepsize was reached:
            solver_state[stepnum] = 5
        else:
            # else save to state that error was ok in i steps:
            solver_state[stepnum] = 4
    elif err_rms == 0.0:
        # if no RMS (most probably the step was too small so rounding
        # error below machine precision led to cut off of digits) step
        # will also be accepted:
        step_accepted = True
        # save successful timestep to simulation environment:
        timestep = _h
        # get maximum step increase for next step:
        _h *= max_factor[0]
        # save to state that machine epsilon was reached:
        solver_state[stepnum] = 7
        # check if step is not above max step:
        if _h > max_stepsize:
            _h = max_stepsize  # reduce to max stepsize
    else:
        # else error was too big.
        # check if stepsize already is at minimum stepsize. this can
        # only be true, if stepsize has already been reduced to min.
        # stepsize, thus to avoid infinite loop set step_accepted=True
        # and skip the rest of the loop:
        if _h == min_stepsize:
            step_accepted = True
            # save not successful but still accepted timestep to
            # simulation environment:
            timestep = _h
            # save this special event to solver state:
            solver_state[stepnum] = 6
        else:
            # else if stepsize not yet at min stepsize, reduce stepsize
            # further by error estimate if this is not less than the
            # minimum factor and redo the step.
            _h *= max(min_factor, (safety * err_rms ** (-1 / (order + 1))))
            # check if step is not below min step:
            if _h < min_stepsize:
                _h = min_stepsize  # increase to min stepsize
            # reset ports array for retrying step:
            ports_all[:] = parr_bkp
            # count failed steps at this step number:
            failed_steps[stepnum] += 1
    # catch von Neumann stability condition:
    if not step_stable[0]:
        # if von Neumann stability violated, do not accept step.
        # This can happen even though the RMS-error is ok, since single
        # non-stable parts can have a too small impact on the RMS. In
        # this case _step_accepted will be overwritten.
        step_accepted = False  # redo the step
        # inrease counter for failed loops
        cnt_instable += 1
        # set new step to maximum von Neumann step (calc. in parts):
        _h = vN_max_step[0]
        # count failed steps at this step number:
        failed_steps[stepnum] += 1
        # reset ports array for retrying step:
        ports_all[:] = parr_bkp
        # break loop if no solution was found after 50 tries:
        if cnt_instable == 50:
            # set timeframe to 0 to break the outer simulation loop
            timeframe = 1e-9
            # save error to solver state:
            solver_state[stepnum] = 99
            """
            TODO: Wie integriere ich das break hier?
            """
    #            break

    return step_accepted, timestep, timeframe, cnt_instable


# %% CALCULATE DIMENSIONLESS NUMBERS:
@njit(nogil=GLOB_NOGIL, cache=True)
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


# %% CALCULATE MATERIAL PROPERTIES:

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
@njit(nogil=GLOB_NOGIL, cache=True)
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


# calc Reynolds number
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def get_Re_water(v, L, ny, Re):
    Re[:] = np.abs(v) * L / ny


# calc Reynolds number and RETURN the result
@nb.njit(nogil=GLOB_NOGIL, cache=True)
def Re_water_return(v, L, ny):
    return np.abs(v) * L / ny


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
@njit(nogil=GLOB_NOGIL, cache=True)
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
@njit(nogil=GLOB_NOGIL, cache=True)
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


# %% part shape specific calculations:
def calc_flow_length(*, part_shape, vertical, **kwargs):
    """
    Calculate the shape specific flow length of a part for the calculation
    of heat-transfer specific numbers, like the Rayleigh number.
    """

    err_str = (
        '`part_shape=' + str(part_shape) + '` was passed to '
        '`calc_flow_length()`. The following shapes are supported:\n'
        '\'plane\', \'cylinder\', \'sphere\'.'
    )
    assert part_shape in ['plane', 'cylinder', 'sphere'], err_str
    err_str = (
        '`vertical=' + str(vertical) + '` was passed to '
        '`calc_flow_length()`. `vertical` must be a bool value, '
        'depicting the orientation of the surface of which the flow '
        'length shall be calculated. For a sphere this argument will '
        'be ignored.'
    )
    assert type(vertical) == bool, err_str
    err_str_len = (
        'The part shape specific length parameters to be passed to '
        '`calc_flow_length()` depend on the part\'s shape and '
        'orientation. The following parameters are needed to calculate '
        'the flow length for each shape:\n'
        '    plane, vertical=True: `height=X`\n'
        '    plane, vertical=False (horizontal): `width=X`, `depth=Y`. '
        'Pass the diameter as value for width and depth for a circular '
        'disk.\n'
        '    cylinder, vertical=True: `height=X`\n'
        '    cylinder, vertical=False (horizontal): `diameter=X`\n'
        '    sphere:  `diameter=X`'
    )

    if part_shape in ('plane', 'cylinder') and vertical:
        assert 'height' in kwargs and isinstance(
            kwargs['height'], (int, float)
        ), err_str_len
        return kwargs['height']  # VDI Wärmeatlas 2013, F2.1
    elif part_shape == 'plane' and not vertical:
        # VDI Wärmeatlas 2013, F2.3
        assert 'width' in kwargs and isinstance(
            kwargs['width'], (int, float)
        ), err_str_len
        assert 'depth' in kwargs and isinstance(
            kwargs['depth'], (int, float)
        ), err_str_len
        return (kwargs['width'] * kwargs['depth']) / (
            2 * (kwargs['width'] + kwargs['depth'])
        )
    elif part_shape == 'cylinder' and not vertical:
        assert 'diameter' in kwargs and isinstance(
            kwargs['diameter'], (int, float)
        ), err_str_len
        return kwargs['diameter'] * np.pi / 2  # VDI Wärmeatlas 2013, F2.4.1
    else:
        assert 'diameter' in kwargs and isinstance(
            kwargs['diameter'], (int, float)
        ), err_str_len
        return kwargs['diameter']  # VDI Wärmeatlas 2013, F2.4.2


# caller to calculate Reynolds number for a round pipe/TES:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def pipe_get_Re(dm, rho, ny, A, d_i, Re):
    get_Re_water(dm / (rho * A), d_i, ny, Re)


# manual inlining function to calculate Reynolds number for a round pipe/TES:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def pipe_get_Re2(dm, rho, ny, A, d_i, Re):
    Re[:] = dm * d_i / (rho * A * ny)


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def pipe_alpha_i(dm, T, rho, ny, lam_fld, A, d_i, x, alpha):
    """
    Calculates the inner alpha value in [W/(m**2K)] between the fluid inside a
    pipe and the pipe wall for each cell of a round pipe or thermal energy
    storage of diameter `d_i` and length `Len`.
    In this case, the wall is considererd in the same row of the temperature
    array as the fluid and thus can't have temperatures different from the
    fluid temperature.

    Parameters:
    -----------
    dm : np.ndarray, float, integer
        Massflow in the pipe/TES in [kg/s].
    rho : np.ndarray
        Fluid density in [kg/m**3].
    ny : np.ndarray
        Fluid kinematic viscosity in [m**2/s].
    lam_fld : np.ndarray
        Fluid heat conductivity in [W/(mK)].
    A : float, integer
        Inner pipe cross section in [m**2] for round pipes or hydraulic cross
        section for pipes of other shapes.
    d_i : float, integer
        Inner pipe diameter in [m] for round pipes or hydraulic diameter for
        pipes of other shapes.
    x : float, integer
        Distance of cell from start of the pipe [m]. If the massflow `dm` is
        negative, the inverse (the distance from the other end of the pipe) is
        taken.
    alpha : np.ndarray
        Array to save the resulting alpha value in [W/(m**2K)] for all cells
        of the pipe/TES.
    """

    # save shape:
    shape = rho.shape
    #    shape = (100,)
    # preallocate arrays:
    Re = np.zeros(shape)
    #    Pr = np.zeros((100,))
    Pr_f = np.zeros(T.shape)
    Nu = np.zeros(shape)

    # get Reynolds and Prandtl number:
    get_Re_water(dm / (rho * A), d_i, ny, Re)
    get_Pr_water(T, Pr_f)

    # get Peclet number to replace calculations of Re*Pr
    Pe = Re * Pr_f

    # use reversed x if first cell of dm is negative (this applies only to
    # parts where the massflow is the same in all cells, since these are the
    # only cells with a cell-specific x-array and a single-cell-massflow. For
    # all other parts, this reversing does not change anything):
    if dm[0] < 0:
        xi = x[::-1]  # create reversed view
    else:
        xi = x[:]  # create view

    # get a mask for the turbulent flows:
    turb = Re > 2300
    # equations for laminar Nusselt number following VDI Wärmeatlas 2013,
    # Chapter G1 - 3.1.1 equation (3), (1) and (2)
    Nu[~turb] = (
        49.371  # 49.371 is 3.66**3 of eq (1) + 0.7**3 of eq (3)
        + (1.077 * (Pe[~turb] * d_i / xi[~turb]) ** (1 / 3) - 0.7)
        ** 3  # eq (2)
    ) ** (1 / 3)

    # equations for turbulent Nusselt number following VDI Wärmeatlas 2013,
    # Chapter G1 - 4.1 equations (27) and (28):
    f = (1.8 * np.log10(Re[turb]) - 1.5) ** (-2)
    Nu[turb] = (
        (f / 8 * Pe[turb])
        / (1 + 12.7 * (Pr_f[turb] ** (2 / 3) - 1) * (f / 8) ** (0.5))
        * (1 + (d_i / xi[turb]) ** (2 / 3) / 3)
    )

    # alpha value is Nusselt number * fluid lambda / d_i,
    # VDI Wärmeatlas 2013, Chapter G1 - 4.1:
    alpha[:] = Nu * lam_fld / d_i


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def pipe_alpha_i_wll_sep(dm, T, rho, ny, lam_fld, A, d_i, x, alpha):
    """
    Calculates the inner alpha value in [W/(m**2K)] between the fluid inside a
    pipe and the pipe wall for each cell of a round pipe or thermal energy
    storage of diameter `d_i` and length `Len`.
    In this case, the wall is considererd in a separate row of the temperature
    array and can thus have temperatures different from the fluid temperature.

    Parameters:
    -----------
    dm : np.ndarray, float, integer
        Massflow in the pipe/TES in [kg/s].
    rho : np.ndarray
        Fluid density in [kg/m**3].
    ny : np.ndarray
        Fluid kinematic viscosity in [m**2/s].
    lam_fld : np.ndarray
        Fluid heat conductivity in [W/(mK)].
    A : float, integer
        Inner pipe cross section in [m**2] for round pipes or hydraulic cross
        section for pipes of other shapes.
    d_i : float, integer
        Inner pipe diameter in [m] for round pipes or hydraulic diameter for
        pipes of other shapes.
    Len : float, integer
        Total pipe length in [m].
    alpha : np.ndarray
        Array to save the resulting alpha value in [W/(m**2K)] for all cells
        of the pipe/TES.
    """

    # save shape:
    shape = rho.shape
    #    shape = (100,)
    # preallocate arrays:
    Re = np.zeros(shape)
    #    Pr = np.zeros((100,))
    Pr = np.zeros(T.shape)
    Nu = np.zeros(shape)

    # get Reynolds and Prandtl number:
    get_Re_water(dm / (rho * A), d_i, ny, Re)
    get_Pr_water(T, Pr)

    # get correction factor for the difference in wall and fluid temperature
    # following VDI Wärmeatlas 2013, Chapter G1 - 3.1.3 equation (13):
    K = (Pr[:, 0] / Pr[:, 1]) ** 0.11
    # save Prandtl number of first row (fluid row) to array for fluid Pr number
    Pr_f = Pr[:, 0]

    # get Peclet number to replace calculations of Re*Pr
    Pe = Re * Pr_f

    # use reversed x if first cell of dm is negative (this applies only to
    # parts where the massflow is the same in all cells, since these are the
    # only cells with a cell-specific x-array and a single-cell-massflow. For
    # all other parts, this reversing does not change anything):
    if dm[0] < 0:
        xi = x[::-1]  # create reversed view
    else:
        xi = x[:]  # create view

    # get a mask for the turbulent flows:
    turb = Re > 2300
    # equations for laminar Nusselt number following VDI Wärmeatlas 2013,
    # Chapter G1 - 3.1.1 equation (3), (1) and (2)
    Nu[~turb] = (
        49.371  # 49.371 is 3.66**3 of eq (1) + 0.7**3 of eq (3)
        + (1.077 * (Pe[~turb] * d_i / xi[~turb]) ** (1 / 3) - 0.7)
        ** 3  # eq (2)
    ) ** (1 / 3)

    # equations for turbulent Nusselt number following VDI Wärmeatlas 2013,
    # Chapter G1 - 4.1 equations (27) and (28):
    f = (1.8 * np.log10(Re[turb]) - 1.5) ** (-2)
    Nu[turb] = (
        (f / 8 * Pe[turb])
        / (1 + 12.7 * (Pr_f[turb] ** (2 / 3) - 1) * (f / 8) ** (0.5))
        * (1 + (d_i / xi[turb]) ** (2 / 3) / 3)
    )

    # alpha value is Nusselt number * correction factor * fluid lambda / d_i,
    # VDI Wärmeatlas 2013, Chapter G1 - 4.1:
    alpha[:] = Nu * K * lam_fld / d_i


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def phex_alpha_i_wll_sep(
    dm, T_fld, T_wll, rho, ny, lam_fld, A, d_h, x, corr_Re, alpha
):
    """
    Calculates the inner alpha value in [W/(m**2K)] between the fluid inside a
    plate heat exchanger and the (rectangular) heat exchanger channel wall for
    each cell of a plate heat exchanger.
    In this case, the wall is considererd in a separate row of the temperature
    array and can thus have temperatures different from the fluid temperature.

    Parameters:
    -----------
    dm : np.ndarray, float, integer
        Massflow in the pipe/TES in [kg/s].
    rho : np.ndarray
        Fluid density in [kg/m**3].
    ny : np.ndarray
        Fluid kinematic viscosity in [m**2/s].
    lam_fld : np.ndarray
        Fluid heat conductivity in [W/(mK)].
    A : float, integer
        Inner pipe cross section in [m**2] for round pipes or hydraulic cross
        section (fluid area perpendicular to the flow direction) for pipes of
        other shapes.
    d_i : float, integer
        Inner pipe diameter in [m] for round pipes or hydraulic diameter for
        pipes of other shapes.
    x : float, integer
        Total plate heat exchanger length in [m].
    alpha : np.ndarray
        Array to save the resulting alpha value in [W/(m**2K)] for all cells
        of the pipe/TES.
    """

    # save shape:
    #    shape = rho.shape
    shape = alpha.shape
    # preallocate arrays:
    Re = np.zeros(shape)  # not needed, since for a pipe/hex this is a scalar
    #    Pr = np.zeros((100,))

    Nu = np.zeros(shape)

    # get Reynolds and Prandtl number:
    get_Re_water(dm / (rho * A), d_h, ny, Re)  # hydraulic diameter as length!
    #    Re = Re_water_return(dm / (rho * A), d_h, ny)  # hydraulic diameter as len!
    Pr_f = Pr_water_return(T_fld)
    Pr_wll = Pr_water_return(T_wll)

    # apply correction difference for turbulators on Reynolds number:
    Re += corr_Re  # [0]

    # get correction factor for the difference in wall and fluid temperature
    # following VDI Wärmeatlas 2013, Chapter G1 - 3.1.3 equation (13):
    K = (Pr_f / Pr_wll) ** 0.11

    # get Peclet number to replace calculations of Re*Pr
    Pe = Re * Pr_f

    # get a mask for the turbulent flows:
    turb = Re > 2300
    # equations for mean laminar Nusselt number following VDI Wärmeatlas 2013,
    # Chapter G1 - 3.1.2 equation (12) with (4), (5) and (11)
    Pe_dx = Pe[~turb] * d_h / x  # precalculate this
    Nu[~turb] = (
        49.371  # 49.371 is 3.66**3 of eq (4) + 0.7**3 of eq (12)
        + (1.615 * (Pe_dx) ** (1 / 3) - 0.7) ** 3  # equation (5)
        + ((2 / (1 + 22 * Pr_f)) ** (1 / 6) * (Pe_dx) ** 0.5) ** 3  # eq(11)
    ) ** (1 / 3)

    # equations for mean turbulent Nusselt number following VDI Wärmeatlas
    # 2013 Chapter G1 - 4.1 equations (27) and (26):
    f = (1.8 * np.log10(Re[turb]) - 1.5) ** (-2)
    Nu[turb] = (
        (f / 8 * Pe[turb])
        / (1 + 12.7 * (Pr_f ** (2 / 3) - 1) * (f / 8) ** (0.5))
        * (1 + (d_h / x) ** (2 / 3))
    )

    # alpha value is Nusselt number * correction factor * fluid lambda / d_i,
    # VDI Wärmeatlas 2013, Chapter G1 - 4.1:
    alpha[:] = Nu * K * lam_fld / d_h


@nb.njit(nogil=GLOB_NOGIL, cache=True)
def phex_alpha_i_wll_sep_discretized(
    dm, T_fld, T_wll, rho, ny, lam_fld, A, d_h, x, corr_Re, alpha
):
    """
    Calculates the inner alpha value in [W/(m**2K)] between the fluid inside a
    plate heat exchanger and the (rectangular) heat exchanger channel wall for
    each cell of a plate heat exchanger.
    In this case, the wall is considererd in a separate row of the temperature
    array and can thus have temperatures different from the fluid temperature.

    Parameters:
    -----------
    dm : np.ndarray, float, integer
        Massflow in the pipe/TES in [kg/s].
    rho : np.ndarray
        Fluid density in [kg/m**3].
    ny : np.ndarray
        Fluid kinematic viscosity in [m**2/s].
    lam_fld : np.ndarray
        Fluid heat conductivity in [W/(mK)].
    A : float, integer
        Inner pipe cross section in [m**2] for round pipes or hydraulic cross
        section (fluid area perpendicular to the flow direction) for pipes of
        other shapes.
    d_i : float, integer
        Inner pipe diameter in [m] for round pipes or hydraulic diameter for
        pipes of other shapes.
    x : float, integer
        Total plate heat exchanger length in [m].
    alpha : np.ndarray
        Array to save the resulting alpha value in [W/(m**2K)] for all cells
        of the pipe/TES.
    """

    # save shape:
    #    shape = rho.shape
    shape = alpha.shape
    # preallocate arrays:
    Re = np.zeros(shape)  # not needed, since for a pipe/hex this is a scalar
    #    Pr = np.zeros((100,))

    Nu = np.zeros(shape)

    # get Reynolds and Prandtl number:
    get_Re_water(dm / (rho * A), d_h, ny, Re)  # hydraulic diameter as length!
    #    Re = Re_water_return(dm / (rho * A), d_h, ny)  # hydraulic diameter as len!
    Pr_f = Pr_water_return(T_fld)
    Pr_wll = Pr_water_return(T_wll)

    # apply correction difference for turbulators on Reynolds number:
    Re += corr_Re  # [0]

    # get correction factor for the difference in wall and fluid temperature
    # following VDI Wärmeatlas 2013, Chapter G1 - 3.1.3 equation (13):
    K = (Pr_f / Pr_wll) ** 0.11

    # get Peclet number to replace calculations of Re*Pr
    Pe = Re * Pr_f

    # get a mask for the turbulent flows:
    turb = Re > 2300
    # equations for mean laminar Nusselt number following VDI Wärmeatlas 2013,
    # Chapter G1 - 3.1.2 equation (12) with (4), (5) and (11)
    Pe_dx = Pe[~turb] * d_h / x[~turb]  # precalculate this
    Nu[~turb] = (
        49.371  # 49.371 is 3.66**3 of eq (4) + 0.7**3 of eq (12)
        + (1.615 * (Pe_dx) ** (1 / 3) - 0.7) ** 3  # equation (5)
        + ((2 / (1 + 22 * Pr_f[~turb])) ** (1 / 6) * (Pe_dx) ** 0.5)
        ** 3  # eq(11)
    ) ** (1 / 3)

    # equations for mean turbulent Nusselt number following VDI Wärmeatlas
    # 2013 Chapter G1 - 4.1 equations (27) and (26):
    f = (1.8 * np.log10(Re[turb]) - 1.5) ** (-2)
    Nu[turb] = (
        (f / 8 * Pe[turb])
        / (1 + 12.7 * (Pr_f[turb] ** (2 / 3) - 1) * (f / 8) ** (0.5))
        * (1 + (d_h / x[turb]) ** (2 / 3))
    )

    # alpha value is Nusselt number * correction factor * fluid lambda / d_i,
    # VDI Wärmeatlas 2013, Chapter G1 - 4.1:
    alpha[:] = Nu * K * lam_fld / d_h


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def cylinder_alpha_inf(T_s, T_inf, flow_length, vertical, r_total, alpha_inf):
    """
    Calculates the outer alpha value in [W/(m**2K)], between the outer cylinder
    wall and the fluid of the environment, of a cylinder in a standard
    environment on the outer surface.

    Parameters:
    -----------
    r_total : float, int
        Total radius of the cylinder including wall and additional material
        layer like insulation.
    alpha_inf : np.ndarray
        Heat transfer coefficient in [W / (m**2K)] between the outer layer and
        the ambient. The shape must equal the fluid temperature array shape or
        be a single array cell. This array is used to calculate the new outer
        surface temperature and to get the new alpha_inf value for the
        calculation of the current U*A-value. Thus this array is
    T_inf : float, int, np.ndarray
        Ambient temperature in [°C] or [K]. If given as array, it must be a
        single cell!
    flow_length : float, int
        Equivalent low length of the horizontal pipe or vertical pipe/TES in
        [m].
    vertical : bool
        Giving information if this pipe/TES is vertical or horizontal. If
        vertical,
    """

    # Kelvin temperature:
    Kelvin = 273.15
    # Prandtl number of DRY air is nearly constant and thus set to:
    Pr = 0.708
    #    f_Pr = 0.347
    # get mean temperature of wall and ambient air:
    T_mean = (T_inf + T_s) / 2
    # get kin. viscosity and lambda for mean temperature:
    ny = np.zeros(T_mean.shape)
    lam = np.zeros(T_mean.shape)
    get_ny_dryair(T_mean, ny)
    get_lam_dryair(T_mean, lam)
    # get Rayleigh number according to VDI Wärmeatlas 2013 chapter F1
    # eq (7), replacing kappa with kappa = ny/Pr (F1 eq (8)) and beta
    # with 1/T_inf (F1 eq (2)):
    Ra = (
        np.abs(T_s - T_inf)
        * 9.81
        * flow_length ** 3
        * Pr
        / ((T_inf + Kelvin) * ny ** 2)
    )
    # check if the cylinder is vertical or horizontal:
    if vertical:
        # get Prandtl number influence function for vertical surfaces according
        # to VDI Wärmeatlas 2013 chapter F2 equation (2):
        #        f_Pr = (1 + (0.492 / Pr)**(9/16))**(-16/9)  this is const for const Pr
        f_Pr = 0.3466023585520853
        # get the Nusselt number for a vertical cylinder by use of VDI
        # Wärmeatlas 2013 chapter F2.1 eq(1) and eq(3):
        Nu = (
            0.825 + 0.387 * (Ra * f_Pr) ** (1 / 6)
        ) ** 2 + 0.435 * flow_length / (2 * r_total)
    else:
        # get Prandtl number influence function for horizontal cylinders
        # according to VDI Wärmeatlas 2013 chapter F2.4 equation (13):
        #        f_Pr = (1 + (0.559 / Pr)**(9/16))**(-16/9)  this is const for const Pr
        f_Pr = 0.3269207911296459
        # get the Nusselt number for a horizontal cylinder by use of VDI
        # Wärmeatlas 2013 chapter F2.4 eq(11):
        Nu = (0.752 + 0.387 * (Ra * f_Pr) ** (1 / 6)) ** 2
    # get alpha:
    alpha_inf[:] = Nu * lam / flow_length


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def plane_alpha_inf(T_s, T_inf, flow_length, vertical, top):
    """
    Calculates the outer alpha value in [W/(m**2K)], between the outer cylinder
    wall and the fluid of the environment, of a cylinder in a standard
    environment on the outer surface.

    Parameters:
    -----------
    r_total : float, int
        Total radius of the cylinder including wall and additional material
        layer like insulation.
    out : np.ndarray
        Heat transfer coefficient in [W / (m**2K)] between the outer layer and
        the ambient. The shape must equal the fluid temperature array shape or
        be a single array cell. This array is used to calculate the new outer
        surface temperature and to get the new alpha_inf value for the
        calculation of the current U*A-value. Thus this array is
    T_inf : float, int, np.ndarray
        Ambient temperature in [°C] or [K]. If given as array, it must be a
        single cell!
    flow_length : float, int
        Equivalent low length of the horizontal pipe or vertical pipe/TES in
        [m].
    vertical : bool
        Giving information if this pipe/TES is vertical or horizontal. If
        vertical,
    """

    # check if the plane is vertical or horizontal:
    if vertical:
        return vert_plane_alpha_inf(T_s, T_inf, flow_length)
    else:
        return hor_plane_alpha_inf(T_s, T_inf, flow_length, top)


@njit(nogil=GLOB_NOGIL, cache=True)
def vert_plane_alpha_inf(T_s, T_inf, flow_length):
    """
    Calculates the outer alpha value in [W/(m**2K)] between the vertical plane
    surface wall and the fluid of the standard environment.

    """

    # Kelvin temperature:
    Kelvin = 273.15
    # Prandtl number of DRY air is nearly constant and thus set to:
    Pr = 0.708
    #    f_Pr = 0.347
    # get mean temperature of wall and ambient air:
    T_mean = (T_inf + T_s) / 2
    # get kin. viscosity and lambda for mean temperature:
    ny = np.zeros(T_mean.shape)
    lam = np.zeros(T_mean.shape)
    get_ny_dryair(T_mean, ny)
    get_lam_dryair(T_mean, lam)

    #    # get Rayleigh number according to VDI Wärmeatlas 2013 chapter F1
    #    # eq (7), replacing kappa with kappa = ny/Pr (F1 eq (8)) and beta
    #    # with 1/T_inf (F1 eq (2)):
    #    Ra = ((T_s - T_inf) * 9.81 * flow_length**3 * Pr
    #          / ((T_inf + Kelvin) * ny**2))
    # get Prandtl number influence function for vertical surfaces according
    # to VDI Wärmeatlas 2013 chapter F2.1 equation (2):
    #    f_Pr = (1 + (0.492 / Pr)**(9/16))**(-16/9)  this is const for const Pr
    f_Pr = 0.3466023585520853
    # get the Nusselt number for a vertical cylinder by use of VDI
    # Wärmeatlas 2013 chapter F2.1 eq(1):
    Nu = (
        0.825
        + 0.387
        * (rayleigh_number(T_s, T_inf, Pr, ny, Kelvin, flow_length) * f_Pr)
        ** (1 / 6)
    ) ** 2

    # get alpha:
    return Nu * lam / flow_length


# @njit(float64(float64, float64, float64, nb.boolean),
@njit(nogil=GLOB_NOGIL, cache=True)
def hor_plane_alpha_inf(T_s, T_inf, flow_length, top):
    """
    Calculates the outer alpha value in [W/(m**2K)] between the plane surface
    wall and the fluid of the standard environment of a horizontal plane.
    This is only implemented for single scalar values! Not to be used with
    arrays!

    This is a reST style.

    :param param1: this is a first param
    :param param2: this is a second param
    :returns: this is a description of what is returned
    :raises keyError: raises an exception :math:`a=b`

    """

    # Kelvin temperature:
    Kelvin = 273.15
    # Prandtl number of DRY air is nearly constant and thus set to:
    Pr = 0.708
    #    f_Pr = 0.347
    # get mean temperature of wall and ambient air:
    T_mean = (T_inf + T_s) / 2
    # get kin. viscosity and lambda for mean temperature:
    ny = ny_dryair_return(T_mean)
    lam = lam_dryair_return(T_mean)
    Nu = np.empty(T_mean.shape)

    Ra = rayleigh_number(  # get Rayleigh-number:
        T_s, T_inf, Pr, ny, Kelvin, flow_length
    )

    # calculation following VDI Wärmeatlas 2013
    for i in range(T_s.shape[0]):
        if (top[i] and T_s[i] >= T_inf) or (not top[i] and T_s[i] < T_inf):
            # VDI F2.3.1
            # heat conduction from the top of the plate to fluid OR from the fluid
            # to the bottom of the plate
            # get Prandtl number influence function for hor. surfaces according
            # to VDI Wärmeatlas 2013 chapter F2.3.1 equation (9):
            #        f_Pr = (1 + (0.322 / Pr)**(11/20))**(-20/11)this is const for const Pr
            f_Pr = 0.40306002707296223
            #            Ra_f_Pr = rayleigh_number(  # get Ra*f_Pr for turbulence check
            #                    T_s[i], T_inf, Pr, ny[i], Kelvin, flow_length) * f_Pr
            Ra_f_Pr = Ra[i] * f_Pr
            # get the Nusselt number for a hor. plane, VDI Wärmeatlas 2013:
            if Ra_f_Pr <= 7e4:  # laminar flow
                Nu[i] = 0.766 * (Ra_f_Pr) ** (1 / 5)  # VDI F2.3.1 eq (7)
            else:  # turbulent flow
                Nu[i] = 0.15 * (Ra_f_Pr) ** (1 / 3)  # VDI F2.3.1 eq (8)
        else:  # VDI F2.3.2
            # heat conduction from the fluid to the top of the plate OR from the
            # bottom of the plate to the fluid
            # get Prandtl number influence function for vertical surfaces according
            # to VDI Wärmeatlas 2013 chapter F2.1 equation (2):
            #        f_Pr = (1 + (0.492 / Pr)**(9/16))**(-16/9)  this is const for const Pr
            f_Pr = 0.3466023585520853
            #            Ra_f_Pr = rayleigh_number(  # get Ra*f_Pr
            #                    T_s[i], T_inf, Pr, ny[i], Kelvin, flow_length) * f_Pr
            #            Ra_f_Pr = Ra[i] * f_Pr
            # get Nusselt number, only valid for 1e3 <= Ra*f_Pr <= 1e10, but there
            # is no known correlation for turbulent convection!
            Nu[i] = 0.6 * (Ra[i] * f_Pr) ** (1 / 5)  # VDI F2.3.2 eq (10)

    # return alpha:
    return Nu * lam / flow_length


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def UA_fld_wll_ins_amb_cyl(
    A_i, r_ln_wll, r_ln_ins, r_rins, alpha_i, alpha_inf, lam_wll, lam_ins, out
):
    """
    Calculates the U*A-value for the heat flow to or from the fluid inside a
    cylinder like a pipe or a round TES to or from the ambient in radial
    direction.
    Layers which are considered: fluid, wall material, insulation or any
    other additional material layer, ambient.
    The reference area must always be the fluid-wall-contact-area for
    consistency with other calculations.

    Parameters:
    -----------
    A_i : float, int
        The fluid-wall-contact area PER CELL. Calculated with:
        A_i = np.pi * r_i * 2 * grid_spacing
    r_ln_wll : float, int
        Radial thickness factor of the wall heat conductivity referred to the
        reference area. Must be pre-calculated with:
        r_ln_wll = r_i * np.log(r_o / r_i)
    r_ln_ins : float, int
        Radial thickness factor of the insulation heat conductivity referred to
        the reference area. Must be pre-calculated with:
        r_ln_ins = r_i * np.log((r_o + s_ins) / r_o)
    r_rins : float, int
        Radial thickness factor of the insulation-to-ambient heat transfer
        coefficient referred to the reference area. Must be pre-calculated
        with:
        r_rins = r_i / (r_o + s_ins)
    alpha_i : int, float, np.ndarray
        Heat transfer coefficient in [W / (m**2K)] between the fluid inside the
        pipe and the wall. The shape must equal the fluid temperature array
        shape, if given as array.
    alpha_inf : np.ndarray
        Heat transfer coefficient in [W / (m**2K)] between the outer layer and
        the ambient. The shape must equal the fluid temperature array shape or
        be a single array cell. This array is used to calculate the new outer
        surface temperature and to get the new alpha_inf value for the
        calculation of the current U*A-value. Thus this array is
    lam_wll : int, float, np.ndarray
        Wall heat conductivity in [W / (mK)]. The shape must equal the fluid or
        wall temperature array shape, if given as array.
    lam_ins : int, float, np.ndarray
        Outer material layer heat conductivity in [W / (mK)]. The shape must
        equal the fluid temperature array shape, if given as array.
    out : float, int, np.ndarray
        Total heat transfer coefficient in [W/K] result output array.
    """

    out[:] = A_i / (
        1 / alpha_i
        + r_ln_wll / lam_wll
        + r_ln_ins / lam_ins
        + r_rins / alpha_inf
    )


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def UA_fld_wll_amb_cyl(A_i, r_ln_wll, r_ro, alpha_i, alpha_inf, lam_wll, UA):
    """
    Calculates the U*A-value for the heat flow to or from the fluid inside a
    cylinder like a pipe or a round TES to or from the ambient in radial
    direction.
    Layers which are considered: fluid, wall material, ambient.
    The reference area must always be the fluid-wall-contact-area for
    consistency with other calculations.

    Parameters:
    -----------
    A_i : float, int
        The fluid-wall-contact area PER CELL. Calculated with:
        A_i = np.pi * r_i * 2 * grid_spacing
    r_ln_wll : float, int
        Radial thickness factor of the wall heat conductivity referred to the
        reference area. Must be pre-calculated with:
        r_ln_wll = r_i * np.log(r_o / r_i)
    r_ro : float, int
        Radial thickness factor of the wall-to-ambient heat transfer
        coefficient referred to the reference area. Must be pre-calculated
        with:
        r_ro = r_i / r_o
    alpha_i : int, float, np.ndarray
        Heat transfer coefficient in [W / (m**2K)] between the fluid inside the
        pipe and the wall. The shape must equal the fluid temperature array
        shape, if given as array.
    alpha_inf : int, float, np.ndarray
        Heat transfer coefficient in [W / (m**2K)] between the outer layer and
        the ambient. The shape must equal the fluid temperature array shape,
        if given as array.
    lam_wll : int, float, np.ndarray
        Wall heat conductivity in [W / (mK)]. The shape must equal the fluid or
        wall temperature array shape, if given as array.
    """

    UA[:] = A_i / (1 / alpha_i + r_ln_wll / lam_wll + r_ro / alpha_inf)


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def UA_fld_wll_cyl(A_i, r_i, r_o, alpha_i, lam_wll, UA):
    """
    Calculates the U*A-value for the heat flow to or from the fluid inside a
    cylinder like a pipe or a round TES to or from the wall in radial
    direction. The wall is considered as a single finite volume element per
    cell, thus the heat flow is calculated to the mid-point (radius wise, not
    mass wise, thus r_mid = (r_o + r_i) / 2) of the wall.
    Layers which are considered: fluid, wall material.
    The reference area must always be the fluid-wall-contact-area for
    consistency with other calculations.

    Parameters:
    -----------
    A_i : float, int
        The fluid-wall-contact area PER CELL. Calculated with:
        A_i = np.pi * r_i * 2 * grid_spacing
    r_i : float, int
        Radius in [m] of the fluid-wall-contact-area.
    r_o : float, int
        Radius in [m] of the outer wall surface.
    alpha_i : int, float, np.ndarray
        Heat transfer coefficient in [W / (m**2K)] between the fluid inside the
        pipe and the wall. The shape must equal the fluid temperature array
        shape, if given as array.
    lam_wll : int, float, np.ndarray
        Wall heat conductivity in [W / (mK)]. The shape must equal the fluid or
        wall temperature array shape, if given as array.
    """

    # the log mean outer diameter is taken for length of lam_wll:
    # np.log((r_o / r_i + 1) / 2) = np.log((r_o + r_i)/ 2 / r_i)
    # with r_wll = (r_o + r_i) / 2
    print('UA_fld_wll -> replace np.log with const!')
    UA[:] = A_i / (1 / alpha_i + r_i * np.log((r_o / r_i + 1) / 2) / lam_wll)


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def UA_wll_ins_amb_cyl(
    A_i, r_i, r_o, r_ln_ins, r_rins, alpha_inf, lam_wll, lam_ins, UA
):
    """
    Calculates the U*A-value for the heat flow to or from the wall of a
    cylinder like a pipe or a round TES to or from the ambient in radial
    direction. The wall is considered as a single finite volume element per
    cell, thus the heat flow is calculated from/to the mid-point of the wall.
    Layers which are considered: wall material, insulation or any other
    additional material layer, ambient.
    The reference area must always be the fluid-wall-contact-area for
    consistency with other calculations.

    Parameters:
    -----------
    A_i : float, int
        The fluid-wall-contact area PER CELL. Calculated with:
        A_i = np.pi * r_i * 2 * grid_spacing
    r_i : float, int
        Radius in [m] of the fluid-wall-contact-area.
    r_o : float, int
        Radius in [m] of the outer wall surface.
    r_ln_ins : float, int
        Radial thickness factor of the insulation heat conductivity referred to
        the reference area. Must be pre-calculated with:
        r_ln_ins = r_i * np.log((r_o + s_ins) / r_o)
    r_rins : float, int
        Radial thickness factor of the insulation-to-ambient heat transfer
        coefficient referred to the reference area. Must be pre-calculated
        with:
        r_rins = r_i / (r_o + s_ins)
    alpha_inf : int, float, np.ndarray
        Heat transfer coefficient in [W / (m**2K)] between the outer layer and
        the ambient. The shape must equal the fluid temperature array shape,
        if given as array.
    lam_wll : int, float, np.ndarray
        Wall heat conductivity in [W / (mK)]. The shape must equal the fluid or
        wall temperature array shape, if given as array.
    lam_ins : int, float, np.ndarray
        Outer material layer heat conductivity in [W / (mK)]. The shape must
        equal the fluid temperature array shape, if given as array.
    """

    # the log mean outer diameter is taken for length of lam_wll:
    # np.log(2 / (r_i / r_o + 1)) = np.log(r_o * 2 / (r_o + r_i))
    # with r_wll = (r_o + r_i) / 2
    UA[:] = A_i / (
        r_i * np.log(2 / (r_i / r_o + 1)) / lam_wll
        + r_ln_ins / lam_ins
        + r_rins / alpha_inf
    )


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def UA_wll_amb_cyl(A_i, r_i, r_o, alpha_inf, lam_wll, UA):
    """
    Calculates the U*A-value for the heat flow to or from the wall of a
    cylinder like a pipe or a round TES to or from the ambient in radial
    direction. The wall is considered as a single finite volume element per
    cell, thus the heat flow is calculated to the mid-point of the wall.
    Layers which are considered: wall material, ambient.
    The reference area must always be the fluid-wall-contact-area for
    consistency with other calculations.

    Parameters:
    -----------
    A_i : float, int
        The fluid-wall-contact area PER CELL. Calculated with:
        A_i = np.pi * r_i * 2 * grid_spacing
    r_i : float, int
        Radius in [m] of the fluid-wall-contact-area.
    r_o : float, int
        Radius in [m] of the outer wall surface.
    alpha_inf : int, float, np.ndarray
        Heat transfer coefficient in [W / (m**2K)] between the outer layer and
        the ambient. The shape must equal the fluid temperature array shape,
        if given as array.
    lam_wll : int, float, np.ndarray
        Wall heat conductivity in [W / (mK)]. The shape must equal the fluid or
        wall temperature array shape, if given as array.
    """

    # the log mean outer diameter is taken for length of lam_wll:
    # np.log(2 / (r_i / r_o + 1)) = np.log(r_o * 2 / (r_o + r_i))
    # with r_wll = (r_o + r_i) / 2
    UA[:] = A_i / (
        r_i * np.log(2 / (r_i / r_o + 1)) / lam_wll + r_i / (r_o * alpha_inf)
    )


@nb.njit(nogil=GLOB_NOGIL, cache=True)
def UA_fld_wll_plate(A, s_wll, alpha_fld, lam_wll):
    """
    Calculates the U*A-value for the heat flow to or from a fluid at a plate
    to or from the ambient.
    Layers which are considered: fluid, wall material.
    The reference area must always be the cross section area.

    Parameters:
    -----------
    A : float, int
        The fluid-wall-contact area in [m^2].
    s_wll : float, int
        Wall thickness in [m].
    alpha_fld : int, float
        Heat transfer coefficient in [W/(m^2K)] between the fluid and the wall.
    lam_wll : int, float
        Wall heat conductivity in [W/(mK)].
    UA : np.ndarray
        Result array where U*A in [W/K] will be saved to.
    """

    return A / (1 / alpha_fld + s_wll / lam_wll)


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def UA_fld_wll_ins_amb_plate(
    A, s_wll, s_ins, alpha_fld, alpha_inf, lam_wll, lam_ins
):
    """
    Calculates the U*A-value for the heat flow to or from a fluid at a plate
    with or without insulation to or from the ambient.
    Layers which are considered: fluid, wall material, insulation, ambient.
    The reference area must always be the cross section area.

    Parameters:
    -----------
    A : float, int
        The fluid-wall-contact area in [m^2].
    s_wll : float, int
        Wall thickness in [m].
    s_ins : float, int
        Insulation thickness in [m]. Can be zero.
    alpha_fld : int, float
        Heat transfer coefficient in [W/(m^2K)] between the fluid and the wall.
    alpha_inf : int, float
        Heat transfer coefficient in [W/(m^2K)] between the outer layer and
        the ambient.
    lam_wll : int, float
        Wall heat conductivity in [W/(mK)].
    lam_ins : int, float
        Insulation heat conductivity in [W/(mK)].
    lam_fld : int, float
        Fluid heat conductivity in [W/(mK)].
    UA : np.ndarray
        Result array where U*A in [W/K] will be saved to.
    """

    return A / (
        1 / alpha_fld + s_wll / lam_wll + s_ins / lam_ins + 1 / alpha_inf
    )


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def UA_wll_ins_amb_plate(A, s_wll, s_ins, lam_wll, lam_ins, alpha_inf):
    """
    Calculates the U*A-value for the heat flow to or from a plate with or
    without insulation to or from the ambient.
    Layers which are considered: wall material, insulation, ambient.
    The reference area must always be the cross section area.

    Parameters:
    -----------
    A : float, int
        The fluid-wall-contact area in [m^2].
    s_wll : float, int
        Wall thickness in [m].
    s_ins : float, int
        Insulation thickness in [m].
    lam_wll : int, float
        Wall heat conductivity in [W/(mK)].
    lam_ins : int, float
        Insulation heat conductivity in [W/(mK)].
    alpha_inf : int, float
        Heat transfer coefficient in [W/(m^2K)] between the insulation and the
        ambient.
    UA : np.ndarray
        Result array where U*A in [W/K] will be saved to.
    """

    return A / (s_wll / lam_wll + s_ins / lam_ins + 1 / alpha_inf)


@nb.njit(nogil=GLOB_NOGIL, cache=True)
def surface_temp_steady_state_inplace(T, T_inf, A_s, alpha_inf, UA, T_s):
    """

    Parameters:
    -----------
    A_s : float, int
        The outer surface area (air-contact-area) PER CELL. Calculated with:
        A_s = np.pi * r_s * 2 * grid_spacing
    alpha_inf : np.ndarray
        Heat transfer coefficient in [W / (m**2K)] between the outer layer and
        the ambient. The shape must equal the fluid temperature array shape or
        be a single array cell. This array is used to calculate the new outer
        surface temperature and to get the new alpha_inf value for the
        calculation of the current U*A-value. Thus this array is
    UA : float, int, np.ndarray
        Total heat transfer coefficient in [W/K].
    T_inf : float, int, np.ndarray
        Ambient temperature in [°C] or [K]. If given as array, it must be a
        single cell!
    """
    # get outer surface temperature, following WTP Formelsammlung Chapter 3.3
    # with sigma = (T-T_inf) / (T_i - T_inf) instead of the constant heat
    # production formula. This formula is only for steady state, thus an error
    # will be incorporated. To get the outer layer temperature, the heatflow
    # from the fluid through the pipe-wall (and insulation) to ambient is set
    # equal with the heatflow from the outer surface (index o) to ambient:
    # (T_s - T_inf) * alpha_inf * A_s = U * A_s * (T_i - T_inf)
    # Since UA already incorporates the inner fluid-wall-contact-surface as
    # reference area, alpha_inf needs to be adjusted by its area.
    T_s[:] = T_inf + (T - T_inf) * UA / (alpha_inf * A_s)


@nb.njit(nogil=GLOB_NOGIL, cache=True)
def surface_temp_steady_state(T, T_inf, A_s, alpha_inf, UA):
    """

    Parameters:
    -----------
    A_s : float, int
        The outer surface area (air-contact-area) PER CELL. Calculated with:
        A_s = np.pi * r_s * 2 * grid_spacing
    alpha_inf : np.ndarray
        Heat transfer coefficient in [W / (m**2K)] between the outer layer and
        the ambient. The shape must equal the fluid temperature array shape or
        be a single array cell. This array is used to calculate the new outer
        surface temperature and to get the new alpha_inf value for the
        calculation of the current U*A-value. Thus this array is
    UA : float, int, np.ndarray
        Total heat transfer coefficient in [W/K].
    T_inf : float, int, np.ndarray
        Ambient temperature in [°C] or [K]. If given as array, it must be a
        single cell!
    """
    # get outer surface temperature, following WTP Formelsammlung Chapter 3.3
    # with sigma = (T-T_inf) / (T_i - T_inf) instead of the constant heat
    # production formula. This formula is only for steady state, thus an error
    # will be incorporated. To get the outer layer temperature, the heatflow
    # from the fluid through the pipe-wall (and insulation) to ambient is set
    # equal with the heatflow from the outer surface (index o) to ambient:
    # (T_s - T_inf) * alpha_inf * A_s = U * A_s * (T_i - T_inf)
    # Since UA already incorporates the inner fluid-wall-contact-surface as
    # reference area, alpha_inf needs to be adjusted by its area.
    return T_inf + (T - T_inf) * UA / (alpha_inf * A_s)


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def series_circuit_UA(*args):
    """
    Calculates the total U*A-value for a series circuit of two or more U*A
    values.

    Parameters:
    -----------
    UA : float, int, np.ndarray
        U*A value (heat conductivity) in [W/K] for each part of the series
        circuit. If given as np.ndarray, all arrays have to be of the same
        shape.

    Returns:
    --------
    UA_series : float, np.ndarray
        Total U*A value (heat conductivity) in [W/K] of the series.

    """

    UA_series = 1 / args[0]  # get inverse of first value
    arg_iter = iter(args)  # make iterator out of args
    next(arg_iter)  # skip first entry since it is already taken
    for arg in arg_iter:  # iterate over the rest of args
        UA_series += 1 / arg  # sum up inverse values
    return 1 / UA_series  # return inverse of sum


@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def parallel_circuit_UA(*args):
    """
    Calculates the total U*A-value for a parallel circuit of two or more U*A
    values.

    Parameters:
    -----------
    UA : float, int, np.ndarray
        U*A value (heat conductivity) in [W/K] for each part of the parallel
        circuit. If given as np.ndarray, all arrays have to be of the same
        shape.

    Returns:
    --------
    UA_series : float, np.ndarray
        Total U*A value (heat conductivity) in [W/K] of the parallel circuit.

    """

    UA_parallel = args[0]  # get first value
    arg_iter = iter(args)  # make iterator out of args
    next(arg_iter)  # skip first entry since it is already taken
    for arg in arg_iter:  # iterate over the rest of args
        UA_parallel += arg  # sum up values
    return UA_parallel  # return sum


# ---> GENERAL FUNCTIONS:
# logarithmic mean temperature difference:
@nb.njit(nogil=GLOB_NOGIL, cache=True)
def log_mean_temp_diff(T_A_one, T_A_two, T_B_one, T_B_two):
    """
    Calculate the logarithmic mean temperature difference (LMTD) of two fluid
    streams `one` and `two` of a heat exchanger with two ends `A` and `B`.

    Parameters:
    -----------
    T_A_one : float, int, np.array
        Fluid temperature of stream one at end A.
    T_A_two : float, int, np.array
        Fluid temperature of stream two at end A.
    T_B_one : float, int, np.array
        Fluid temperature of stream one at end B.
    T_B_two : float, int, np.array
        Fluid temperature of stream two at end B.
    """

    Delta_T_A = T_A_one - T_A_two
    Delta_T_B = T_B_one - T_B_two
    lmtd = (Delta_T_A - Delta_T_B) / (np.log(Delta_T_A / Delta_T_B))
    return lmtd


# get simple moving average of the array x and N cells:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def moving_avg(x, N):
    arr = np.zeros(x.shape[0] + 1)
    arr[1:] = x
    cumsum = np.cumsum(arr)
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# fill the edges of x to new_length, so that input x is placed in the middle
# of the output array. if the number of new cells is not even, the array is
# shifted one cell to the end:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def fill_edges(x, new_length):
    old_length = x.shape[0]  # get old length
    residual = new_length - old_length  # get difference in lengths
    x_new = np.zeros(new_length)  # create new array
    start = residual // 2 + residual % 2  # get start point where to insert
    x_new[start : start + old_length] = x  # fill new array in the middle
    x_new[:start] = x[0]  # fill before start with first value
    x_new[old_length + start :] = x[-1]  # fill at end with last value
    return x_new


# this function calls simple moving average on array x and N cells AND fills
# the edges with the last and first value:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def moving_avg_fill_edges(x, N):
    return fill_edges(moving_avg(x, N), x.shape[0])


# get window weighted moving average over array x with window weight wght and
# the possibility to fill the edges with the last correct value to get an array
# of the same shape as x:
@nb.jit(nopython=True, nogil=GLOB_NOGIL, cache=True)
def weighted_moving_avg(x, wght, fill_edges=False):
    # get number of cells to calculate average in each step and get total
    # average array length:
    N = wght.size
    length = x.shape[0] - N + 1
    # if edges shall be filled, create an array like the input array and calc.
    # new starting point where the "real" moving average is starting:
    if fill_edges:
        wa_len = x.shape[0]  # get length
        residual = wa_len - length  # calc. remaining edge points to be filled
        start = residual // 2 + residual % 2  # calc. starting point
        wa = np.zeros(wa_len)  # create result array
    else:
        start = 0  # start at 0
        wa = np.zeros(length)  # create result array
    # loop over array:
    for i in range(length):
        wa[i + start] = (x[i : i + N] * wght).sum()  # calc weighted mean
    # fill edges before start with first value and after end with last value
    if fill_edges:
        wa[:start] = wa[start]
        wa[length + start :] = wa[i + start]
    return wa


@nb.njit(parallel=GLOB_PARALLEL)
def root_finder(poly_coeff, roots):
    """
    Finds the roots of a polynome for an array of root values `roots`.
    This means that a polynome, given by its polynome coefficient array
    `poly_coeff`, is reversed at each value of `roots`. A polynome defining
    the saturated water mass in air for a given temperature, this returns the
    Taupunkt temperature for a given water mass.
    Since the results have a shape of n-1 for a polynome of degree n, the
    results have to be filtered. This may be done in the following way:
        >>> # set all imaginary dominated values to zero:
        >>> rts_arr[np.abs(rts_arr.imag) > 1e-12] = 0.
        >>> # set values above an upper and lower boundary to zero:
        >>> rts_arr[rts_arr > 85] = 0.
        >>> rts_arr[rts_arr < 10] = 0.
        >>> # extract all non-zero values:
        >>> rts_arr.real[rts_arr.real != 0]
        >>> # check if the shape is correct, else use other imaginary and real
        >>> # bounds for setting to zero:
        >>> assert rts_arr.shape == roots.shape

    Parameters:
        poly_coeff : np.ndarray
            Polynomial coefficients to be reversed. Should be given as
            **dtype=np.complex128** to avoid typing errors.
        roots : np.ndarray
            Roots to solve the polynomial for.
    """
    polcoeffs = poly_coeff.copy()
    lin_coeff = polcoeffs[-1]
    rts_arr = np.zeros(
        (roots.shape[0], poly_coeff.shape[0] - 1), dtype=np.complex128
    )
    for i in nb.prange(roots.shape[0]):
        polcoeffs[-1] = lin_coeff - roots[i]
        rts_arr[i, :] = np.roots(polcoeffs)
    return rts_arr


# %% Empiric relations, polynomes etc. for startup times, regression...
@nb.njit
def lim_growth(x, s, b0, k):
    """
    Function for limited growth. Used in several fits, thus it is implemented
    here as a raw function, which can be used in closures, inlining etc.

    Parameters
    ----------
    x : float, int, np.ndarray
        x values of the growth function.
    s : float, optional
        Limit of the growth function.
    b0 : float, optional
        Starting value. Values of 0 are **NOT RECOMMENDED**.
    k : float, optional
        Curvature parameter.

    Returns
    -------
    float, np.ndarray
        Value at point `x`.

    """
    return s - (s - b0) * k ** x


@nb.njit(cache=True)
def chp_startup_th(
    time, s=1.0, b0=4.3715647889609857e-4, k=8.61423130773867e-3
):
    """
    Thermal power output and/or efficiency factor during EC Power XRGi20
    CHP Startup.
    See auswertung_bhkw.chp_fits for generation of the fit.

    Parameters
    ----------
    time : float, int, np.ndarray
        Time or timepoints in **seconds** [s] at which the startup progress
        shall be evaluated. 0 ist the CHP start time.
    s : float, optional
        Maximum power to reach, maximum modulation or efficiency.
        If set to 1, will return the result as a fraction of 1, else the
        absolute value will be returned. The default is 1..
    b0 : float, optional
        Starting value. Cannot be set to zero.
        The default is 4.3715647889609857e-4.
    k : float, optional
        Curvature parameter. The default is 8.61423130773867e-3.

    Returns
    -------
    float, np.ndarray
        Value fraction of `s` at the time `time`. If `time` is an np.ndarray,
        the same type will be returned.

    """
    return s / (1 + (s / b0 - 1) * np.e ** (-k * s * time))


@nb.njit(cache=True)
def chp_startup_el(
    time,
    s=4.6611096613889975,
    b0=-3.6832212021813398e-09,
    k=0.9997484824090741,
):
    """
    Electrical power output and/or efficiency factor during EC Power XRGi20
    CHP Startup. Limited growth, close to linear growth, startup to full power
    (99% of modulation) within 950 seconds was found to be matching
    measurement data.
    See auswertung_bhkw.chp_fits for generation of the fit.

    Parameters
    ----------
    time : float, int, np.ndarray
        Time or timepoints in **seconds** [s] at which the startup progress
        shall be evaluated. 0 ist the CHP start time.
    s : float, optional
        Maximum power to reach, maximum modulation or efficiency.
        If set to 1, will return the result as a fraction of 1, else the
        absolute value will be returned. The default is 4.6611.
    b0 : float, optional
        Starting value. Cannot be set to zero.
        The default is -3.68322e-9.
    k : float, optional
        Curvature parameter. The default is 0.9997484824090741.

    Returns
    -------
    float, np.ndarray
        Value fraction of `s` at the time `time`. If `time` is an np.ndarray,
        the same type will be returned.

    """
    return lim_growth(time, s, b0, k)
    # return s - (s - b0) * k**time


@nb.njit(cache=True)
def chp_startup_gas(time):
    """
    Gas power input and/or efficiency factor during EC Power XRGi20
    CHP Startup. Compound of thermal and electrical startup factors, scaled by
    the efficiencies given in the datasheet. With this, full gas power input
    is reached 287s after startup. The resultung efficiency of a startup from
    0 to 100% is 60%, including the extraction of remaining heat during
    shutdown, the 0-1-0 efficiency is 69.1%.

    Parameters
    ----------
    time : float, int, np.ndarray
        Time or timepoints in **seconds** [s] at which the startup progress
        shall be evaluated. 0 ist the CHP start time.

    Returns
    -------
    float, np.ndarray
        Value fraction of `s` at the time `time`. If `time` is an np.ndarray,
        the same type will be returned.

    """
    return chp_startup_el(time) / 0.32733 + chp_startup_th(time) / 0.6334


@nb.njit(cache=True)
def chp_thermal_power(
    modulation, s=0.60275, b0=0.972917, k=3.5990506789130166
):
    """
    Thermal power output and/or efficiency factor in dependence of the
    electrical power modulation of an EC Power XRGi20 CHP plant.
    Limited growth fit was found to be matching measurement data.
    See auswertung_bhkw.chp_fits for generation of the fit.

    Parameters
    ----------
    time : float, int, np.ndarray
        Time or timepoints in **seconds** [s] at which the startup progress
        shall be evaluated. 0 ist the CHP start time.
    s : float, optional
        Maximum power to reach, maximum modulation or efficiency.
        If set to 1, will return the result as a fraction of 1, else the
        absolute value will be returned. The default is 1..
    b0 : float, optional
        Starting value. Cannot be set to zero.
        The default is 0.
    k : float, optional
        Duration until full power is reached parameter. The default is 960.

    Returns
    -------
    float, np.ndarray
        Value fraction of `s` at the time `time`. If `time` is an np.ndarray,
        the same type will be returned.

    """
    return lim_growth(modulation, s, b0, k)
    # return s - (s - b0) * k**modulation


@nb.njit(cache=True)
def chp_gas_power(
    modulation, s=-1.17, b0=0.995828402366862, k=1.9507547298681704
):
    """
    Gas power input (**lower heating value**) in dependency of the
    electrical power modulation of an EC Power XRGi20 CHP plant.
    Limited growth fit was found to be matching measurement data.
    See auswertung_bhkw.chp_fits for generation of the fit.

    Parameters
    ----------
    time : float, int, np.ndarray
        Time or timepoints in **seconds** [s] at which the startup progress
        shall be evaluated. 0 ist the CHP start time.
    s : float, optional
        Maximum power to reach, maximum modulation or efficiency.
        If set to 1, will return the result as a fraction of 1, else the
        absolute value will be returned. The default is 1..
    b0 : float, optional
        Starting value. Cannot be set to zero.
        The default is 0.
    k : float, optional
        Duration until full power is reached parameter. The default is 960.

    Returns
    -------
    float, np.ndarray
        Value fraction of `s` at the time `time`. If `time` is an np.ndarray,
        the same type will be returned.

    """
    return lim_growth(modulation, s, b0, k)


@nb.njit
def chp_shutdown_th(time, a=-1.2532286835042036e-09, b=927.5198588530006):
    """
    Thermal power output/efficiency of a CHP plant.

    Thermal power output and/or efficiency factor during EC Power XRGi20
    CHP switchoff. Cubic fit chosen for the measurement data.
    See auswertung_bhkw.chp_fits for generation of the fit.

    Parameters
    ----------
    time : float, int, np.ndarray
        Time or timepoints in **seconds** [s] at which the switchoff progress
        shall be evaluated. 0 ist the CHP start time.
    a : float, optional
        Maximum power to reach, maximum modulation or efficiency.
        If set to 1, will return the result as a fraction of 1, else the
        absolute value will be returned. The default is 1..
    b : float, optional
        Slope parameter. The default is -1.035329352327848e-3.

    Returns
    -------
    float, np.ndarray
        Value fraction of `s` at the time `time`. If `time` is an np.ndarray,
        the same type will be returned.

    """
    return a * (time - b) ** 3


@nb.njit
def quad_get_c(x, dy, b, c_base):
    """Get c parameter for quad polynome for condensing hex."""
    return c_base - dy / (2 * b)


@nb.njit
def quad_get_b(y0, a, c):
    """Get b parameter for quad polynome for condensing hex."""
    return (np.expand_dims(y0, -1) - a) / c ** 2


@nb.njit
def condensing_hex_quad_poly(
    X_pred,
    int_comb_idx,
    nvars_per_ftr,  # polynomial transf.
    pca_mean,
    pca_components,  # PCA transformation
    lm_intercept,
    lm_coef,  # linear model transformation
    dm_water_thresh=0.1,
    dx=0.01,
):
    """
    Calculate condensing HEX temperatures below the valid massflow threshold.

    **ONLY VALID below the valid massflow range**, typically from 0-10% of the
    maximum massflow.

    Parameters
    ----------
    dx : TYPE, optional
        Delta x to determin slope from. The default is .01.

    Returns
    -------
    None.

    """
    # extract n samples:
    n_samples = X_pred.shape[0]
    # prediction arrays at the boundary and +dx for the slope
    # extract and save massflow values:
    dm_bkp = X_pred[:, 2:3].copy()  # :3 extracts it as 2D arr and avoids resh.
    X_pred_bc = np.vstack(  # prediction x array at the boundary
        (
            X_pred[:, 0],
            X_pred[:, 1],
            np.full((X_pred.shape[0],), dm_water_thresh),
            X_pred[:, 3],
        )
    ).T
    X_pred_dx = X_pred_bc.copy()  # prediction x arr with dx for slope
    X_pred_dx[:, 2] += dx
    y0 = X_pred[:, 1]  # extract fg entry temperature
    # make predictions at dm_water_thresh, the boundary of the valid
    # region
    X_pf_bc = transform_to_poly_nb(
        X_pred_bc, int_comb_idx, nvars_per_ftr, n_samples
    )
    X_PC_bc = transform_pca_nb(X_pf_bc, pca_mean, pca_components)
    # predict
    y_hat_bc = poly_tranfs_pred(X_PC_bc, lm_intercept, lm_coef)
    # make predictions at dm_water_thresh+dx for generating the slope
    X_pf_dx = transform_to_poly_nb(
        X_pred_dx, int_comb_idx, nvars_per_ftr, n_samples
    )
    X_PC_dx = transform_pca_nb(X_pf_dx, pca_mean, pca_components)
    # predict
    y_hat_dx = poly_tranfs_pred(X_PC_dx, lm_intercept, lm_coef)
    dy = (y_hat_bc - y_hat_dx) / dx  # get the slopes
    # set c to dm_water_thresh for the first iteration of both temperatures
    c = np.array([[dm_water_thresh, dm_water_thresh]], dtype=np.float64)  #
    for i in range(1):
        b = quad_get_b(y0=y0, a=y_hat_bc, c=c)
        c = quad_get_c(x=dm_bkp, dy=dy, b=b, c_base=dm_water_thresh)
    T_pred_below_thresh = y_hat_bc + b * (dm_bkp - dm_water_thresh) ** 2
    return T_pred_below_thresh


# @njit(nogil=GLOB_NOGIL, cache=True)  # parallel=GLOB_PARALLEL useful
def _process_chp_core_modulation(
    process_flows,
    power_modulation,
    T,
    T_chp_in,
    T_chp_in_max,
    T_chp_in_max_emrgncy,
    mod_lower,
    min_on_time,
    min_off_time,
    max_ramp_el,
    startup_in_progress,
    shutdown_in_progress,
    chp_state,
    startup_at_time,
    shutdown_at_time,
    startup_duration,
    shutdown_duration,
    chp_on_perc,
    remaining_heat,
    bin_pow_fac,
    startup_factor_th,
    startup_factor_el,
    shutdown_factor_th,
    startuptsteps,
    chp_off_perc,
    dt_time_temp_exc,
    max_temp_exc_time,
    stepnum,
    time_vec,
    timestep,
):
    """
    Process masssflows for parts with multiple flow channels.

    Massflows are being processed for parts which have multiple separated flow
    channels. The massflow in each flow channel must be invariant.
    The massflow through ports in `dm_io` is aquired by update_FlowNet.
    """
    # process flows is only executed ONCE per timestep, afterwards the bool
    # process_flows is set to False.
    if process_flows[0]:  # only if flows not already processed
        # get current elapsed time
        curr_time = time_vec[stepnum[0] - 1] + timestep
        # get state of the last step
        state_last_step = chp_state[stepnum[0] - 1]

        # check for modulation range and set on-off-integer:
        if power_modulation[0] < mod_lower:
            # binary power multiplication factor to enable off-state
            # for modulations < mod_lower, f.i. to avoid modulations below
            # 50%.
            bin_pow_fac = 0.0
            chp_on = False  # chp is off
        else:
            bin_pow_fac = 1.0
            chp_on = True  # chp is on

        # detect changes in the state to save start/stop times
        if (state_last_step != 0.0) != chp_on:
            if not chp_on:  # if last step chp was on and now off
                # assert that minimum run time is fullfilled. if not,
                # avoid switching off by keeping min. modulation
                if min_on_time > (curr_time - startup_at_time):
                    # if minimum run time not reached, set chp to on
                    bin_pow_fac = 1.0
                    chp_on = True
                    power_modulation[0] = mod_lower
                else:  # else allow shutdown
                    shutdown_at_time = curr_time  # chp was shutdown
                    shutdown_in_progress = True
            #                        print('shutdown at {0:.3f} s'.format(curr_time))
            else:  # if last step chp was off and now it is on
                # assert that minimum off time is fulfilled AND
                # (both ok -> OR statetment) inlet temp. is not exceeding
                # max temp.. If any is True, avoid CHP startup
                if (
                    (min_off_time > (curr_time - shutdown_at_time))
                    or (T_chp_in[0] > T_chp_in_max)
                    or np.any(T > T_chp_in_max_emrgncy)
                ):
                    # if minimum off time not reached or temperature too
                    # high, set chp to off
                    bin_pow_fac = 0.0
                    chp_on = False
                    power_modulation[0] = 0.0
                else:  # else allow switching on
                    startup_at_time = curr_time  # chp was started
                    startup_in_progress = True
        #                        print('start at {0:.3f} s'.format(curr_time))
        elif chp_on:
            # if CHP was on last step AND is on now, check for ramps
            # get difference of modulation and absolute ramp per second
            mod_diff = state_last_step - power_modulation[0]
            mod_ramp_abs = np.abs(mod_diff) / timestep
            # if absolute ramp is higher than max ramp, limit change to
            # ramp
            if mod_ramp_abs > max_ramp_el:
                if mod_diff <= 0.0:  # ramp up too fast
                    power_modulation[0] = (  # set ramp to max ramp
                        state_last_step + max_ramp_el * timestep
                    )
                else:  # ramp down too fast
                    power_modulation[0] = (  # set ramp to max ramp
                        state_last_step - max_ramp_el * timestep
                    )
        # if chp is on, check if inlet temperature was exceeded or any
        # temperature is above emergency shutdown temp., then shutdown
        if chp_on and (
            (T_chp_in[0] > T_chp_in_max) or np.any(T > T_chp_in_max_emrgncy)
        ):
            # if max inlet temp. is exceeded, check max. allowed time for
            # exceeding and if too large, shutdown CHP due to overtemp.,
            # independend of min. run times and other parameters.
            # also if inlet temp. is above an emergency threshold.
            if (dt_time_temp_exc > max_temp_exc_time) or np.any(
                T > T_chp_in_max_emrgncy
            ):
                power_modulation[0] = 0.0
                bin_pow_fac = 0.0
                chp_on = False
                shutdown_at_time = curr_time
                shutdown_in_progress = True
            #                    emergeny_shutdown = True
            #                    print('emergency shutdown at {0:.3f} s'.format(curr_time))
            else:  # if timer not exceeded
                # delta t how long the temp. has been exceeded. after the
                # if-else check, since +timestep is at the end of the
                # step, thus relevant for the next step.
                dt_time_temp_exc += timestep
        else:  # else if temp. not exceeded, reset timer
            dt_time_temp_exc = 0.0

        # save chp state:
        chp_state[stepnum[0]] = bin_pow_fac * power_modulation[0]

        # process startup and shutdown procedure
        # is the CHP switched on? If yes, startup time is larger than
        # shutdown time.
        if startup_at_time > shutdown_at_time:
            # if chp shutdown was quite recent, thus heat is remaining
            # -> shorten startup procedure
            if shutdown_factor_th > chp_off_perc:
                # if shutdown was recent, take the shutdown factor and
                # look where in startup can be found. then add this
                # timestep where it was found to the startup time
                # (=increase startup duration) to account for remaining
                # heat in the system
                remaining_heat = np.argmin(
                    np.abs(startuptsteps - shutdown_factor_th)
                )
                # and reset shutdown factor to zero and set shutdown in
                # progress False to avoid doing this twice:
                shutdown_factor_th = 0.0
                shutdown_in_progress = False
            # get startup duration:
            startup_duration = (  # on since
                curr_time - startup_at_time + remaining_heat
            )
            # do factor calculations only, if startup not yet finished,
            # else do nothing, since factors are already set to 1
            if startup_in_progress:
                # power multiplication factors:
                startup_factor_th = chp_startup_th(startup_duration)
                startup_factor_el = chp_startup_el(startup_duration)
                # limit values to 0<=x<=1
                startup_factor_th = (
                    0.0
                    if startup_factor_th < 0.0
                    else 1.0
                    if startup_factor_th > 1.0
                    else startup_factor_th
                )
                startup_factor_el = (
                    0.0
                    if startup_factor_el < 0.0
                    else 1.0
                    if startup_factor_el > 1.0
                    else startup_factor_el
                )
                # check if thermal startup is completed, else go on
                if startup_factor_th > chp_on_perc:
                    # if thermal startup is completed, set all startups as
                    # completed
                    startup_in_progress = False
                    startup_factor_th = 1.0
                    startup_factor_el = 1.0
                    remaining_heat = 0.0
        else:  # if shutdown was more recent
            shutdown_duration = curr_time - shutdown_at_time  # off since
            if shutdown_in_progress:
                shutdown_factor_th = chp_shutdown_th(shutdown_duration)
                if shutdown_factor_th < chp_off_perc:
                    # shutdown finished. reset values
                    shutdown_in_progress = False
                    shutdown_factor_th = 0.0

    # return process flows bool to disable processing flows until next step
    return (
        bin_pow_fac,
        startup_at_time,
        shutdown_at_time,
        startup_in_progress,
        shutdown_in_progress,
        startup_factor_th,
        startup_factor_el,
        shutdown_factor_th,
        dt_time_temp_exc,
    )


# %% Simulation environment functions:
@nb.njit
def predictor_step(diff_fun, args, h, y_prev):
    return y_prev + h * diff_fun(*args, h)


@nb.njit
def solve_pred_loop(h, diff_funs, all_args, all_y_prev, interm_res):
    ndiffs = len(diff_funs)
    for n in range(ndiffs):
        interm_res[n][:] = predictor_step(
            diff_funs[n], all_args[n], h, all_y_prev[n]
        ).ravel()
    return interm_res


@nb.jit(parallel=GLOB_PARALLEL)
def heun_predictor(_h, solve_num, parts, stepnum, i):
    for part in solve_num:
        # if first try, add last step's part truncation error to part:
        if i == 0:
            parts[part]._trnc_err += parts[part].__new_trnc_err
        # get results from last timestep and pass them to
        # current-timestep-temperature-array:
        parts[part].T[:] = parts[part].res[stepnum - 1]
        # calculate differential at result:
        parts[part]._df0 = solve_num[part](_h)
        # calculate and save predictor step:
        parts[part].T[:] = parts[part].T + _h * parts[part]._df0


# %% regression based functions
def make_poly_transf_combs_for_nb(
    mdl, n_features=None, pipeline=True, poly_step='polynomialfeatures'
):
    """Construct regressor combinations for polynomial."""
    if pipeline:
        # get polynomial features from pipeline
        poly_feats = mdl.named_steps[poly_step]
    else:
        poly_feats = mdl
    if n_features is None:
        n_features = getattr(poly_feats, 'n_input_features_', None)
        if n_features is None:
            raise ValueError
    # extract combinations as persisten tuple
    cmbntns = tuple(
        poly_feats._combinations(
            n_features,
            poly_feats.degree,
            poly_feats.interaction_only,
            poly_feats.include_bias,
        )
    )
    # make to integer indexing array and fill with dummy false value
    int_cmb_idx = (
        np.zeros((len(cmbntns), len(cmbntns[-1])), dtype=np.int64) - 99
    )
    # create tuple with number of variables per combination
    nvars_per_ftr = tuple([len(combo) for combo in cmbntns])
    # make combinations tuple to integer index array, leaving blank cells
    # filled with dummy value (which is ought to raise an error if called)
    for i, c in enumerate(cmbntns):
        int_cmb_idx[i, : nvars_per_ftr[i]] = c
    return int_cmb_idx, nvars_per_ftr


def extract_pca_results(mdl, pipeline=True, pca_step='pca'):
    """Extract PCA result vectors/matrices for transformation."""
    if pipeline:
        # get polynomial features from pipeline
        pca_mdl = mdl.named_steps[pca_step]
    else:
        pca_mdl = mdl
    return pca_mdl.mean_, pca_mdl.components_


# the following 3 functions are numba compatible:
@nb.njit
def transform_to_poly_nb(X, int_comb_idx, nvars_per_ftr, n_samples):
    """Transform X vector to polynomial for predictions."""
    XP = np.ones((n_samples, int_comb_idx.shape[0]), dtype=np.float64)
    for n in range(XP.shape[0]):
        for i in range(XP.shape[1]):
            XP[n, i] = (
                XP[n, i] * X[n][int_comb_idx[i, : nvars_per_ftr[i]]].prod()
            )
    return XP


@nb.njit
def transform_pca_nb(XP, pca_mean, pca_components):
    """Generate PCA transformation matrix."""
    return np.dot(XP - pca_mean, pca_components.T)


@nb.njit
def poly_tranfs_pred(XP_pca_transf, intercept, coef):
    """Predict PCA transformed polynomial."""
    return intercept + np.dot(XP_pca_transf, coef.T)


# %% root solvers to use in/with numba funtions


@nb.njit
def root_array_secant(func, x0, args, tol, maxiter):
    """
    A vectorized version of Newton, Halley, and secant methods for arrays.
    Do not use this method directly. This method is called from `newton`
    when ``np.size(x0) > 1`` is ``True``. For docstring, see `newton`.

    Taken and adapted from scipy.optimize.newton
    This solver may be slower than the excplicit secant solver, but it is
    stable and has a higher precision. In contrast

    **This is the preferred solver for solving implicit differential
    equations.**

    """
    # Explicitly copy `x0` as `p` will be modified inplace, but the
    # user's array should not be altered.
    p = x0.copy()

    # failures = np.ones_like(p).astype(bool)
    # nz_der = np.ones_like(failures)
    failures = p != -1234.4321  # bool array creation for numba
    nz_der = failures.copy()

    # print('using secant method')
    # Secant method
    dx = np.finfo(np.float64).eps ** 0.33
    p1 = p * (1 + dx) + np.where(p >= 0, dx, -dx)
    q0 = func(p, *args)
    q1 = func(p1, *args)
    # active = np.ones_like(p, dtype=bool)
    active = failures.copy()
    for _ in range(maxiter):
        nz_der = q1 != q0
        # stop iterating if all derivatives are zero
        if not nz_der.any():
            p = (p1 + p) / 2.0
            break
        # Secant Step
        dp = (q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]
        # only update nonzero derivatives
        p[nz_der] = p1[nz_der] - dp
        active_zero_der = ~nz_der & active
        p[active_zero_der] = (p1 + p)[active_zero_der] / 2.0
        active &= nz_der  # don't assign zero derivatives again
        failures[nz_der] = np.abs(dp) >= tol  # not yet converged
        # stop iterating if there aren't any failures, not incl zero der
        if not failures[nz_der].any():
            break
        p1, p = p, p1
        q0 = q1
        q1 = func(p1, *args)

    return p


@nb.njit
def root_array_newton(func, x0, fprime, args, tol, maxiter):
    """
    A vectorized version of Newton, Halley, and secant methods for arrays.
    Do not use this method directly. This method is called from `newton`
    when ``np.size(x0) > 1`` is ``True``. For docstring, see `newton`.

    Taken from scipy.optimize.newton.
    Also accepts a derivative function in `fprime`.
    """
    # Explicitly copy `x0` as `p` will be modified inplace, but the
    # user's array should not be altered.
    p = x0.copy()

    # failures = np.ones_like(p).astype(bool)
    # nz_der = np.ones_like(failures)
    failures = p != -1234.4321  # bool array creation for numba
    nz_der = failures.copy()
    if fprime is not None:
        # print('using newton raphson method')
        # Newton-Raphson method
        for iteration in range(maxiter):
            # first evaluate fval
            fval = func(p, *args)
            # If all fval are 0, all roots have been found, then terminate
            if not fval.any():
                failures = fval.astype(bool)
                break
            fder = fprime(p, *args)
            nz_der = fder != 0
            # stop iterating if all derivatives are zero
            if not nz_der.any():
                break
            # Newton step
            dp = fval[nz_der] / fder[nz_der]

            # only update nonzero derivatives
            p[nz_der] -= dp
            failures[nz_der] = np.abs(dp) >= tol  # items not yet converged
            # stop iterating if there aren't any failures, not incl zero der
            if not failures[nz_der].any():
                break
    else:
        # print('using secant method')
        # Secant method
        dx = np.finfo(np.float64).eps ** 0.33
        p1 = p * (1 + dx) + np.where(p >= 0, dx, -dx)
        q0 = func(p, *args)
        q1 = func(p1, *args)
        # active = np.ones_like(p, dtype=bool)
        active = failures.copy()
        for iteration in range(maxiter):
            nz_der = q1 != q0
            # stop iterating if all derivatives are zero
            if not nz_der.any():
                p = (p1 + p) / 2.0
                break
            # Secant Step
            dp = (q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]
            # only update nonzero derivatives
            p[nz_der] = p1[nz_der] - dp
            active_zero_der = ~nz_der & active
            p[active_zero_der] = (p1 + p)[active_zero_der] / 2.0
            active &= nz_der  # don't assign zero derivatives again
            failures[nz_der] = np.abs(dp) >= tol  # not yet converged
            # stop iterating if there aren't any failures, not incl zero der
            if not failures[nz_der].any():
                break
            p1, p = p, p1
            q0 = q1
            q1 = func(p1, *args)

    return p


@nb.njit
def root_array_newton_fast(func, x0, fprime, args, tol, maxiter):
    """
    A vectorized version of Newton, Halley, and secant methods for arrays.
    Do not use this method directly. This method is called from `newton`
    when ``np.size(x0) > 1`` is ``True``. For docstring, see `newton`.

    **ONLY USE THIS WHEN ACCURACY IS NOT IMPORTANT!!**
    """
    # Explicitly copy `x0` as `p` will be modified inplace, but the
    # user's array should not be altered.
    p = x0.copy()

    # failures = np.ones_like(p).astype(bool)
    # nz_der = np.ones_like(failures)
    nz_der = p != -1234.4321  # bool array creation for numba
    if fprime is not None:
        # print('using newton raphson method')
        # Newton-Raphson method
        for iteration in range(maxiter):
            # first evaluate fval
            fval = func(p, *args)
            # If all fval are 0, all roots have been found, then terminate
            if not fval.any():
                failure = False
                break
            fder = fprime(p, *args)
            nz_der = fder != 0
            # stop iterating if all derivatives are zero
            if not nz_der.any():
                break
            # Newton step
            dp = fval[nz_der] / fder[nz_der]
            # only update nonzero derivatives
            p[nz_der] -= dp
            failure = ((dp - tol) ** 2).mean() > tol  # items not yet converged
            # stop iterating if there aren't any failures, not incl zero der
            if not failure:
                break
    else:
        # print('using secant method')
        # Secant method
        dx = np.finfo(np.float64).eps ** 0.33
        p1 = p * (1 + dx) + np.where(p >= 0, dx, -dx)
        q0 = func(p, *args)
        q1 = func(p1, *args)
        # active = np.ones_like(p, dtype=bool)
        active = nz_der.copy()
        for iteration in range(maxiter):
            nz_der = q1 != q0
            # stop iterating if all derivatives are zero
            if not nz_der.any():
                p = (p1 + p) / 2.0
                break
            # Secant Step
            dp = (q1 * (p1 - p))[nz_der] / (q1 - q0)[nz_der]
            # only update nonzero derivatives
            p[nz_der] = p1[nz_der] - dp
            active_zero_der = ~nz_der & active
            p[active_zero_der] = (p1 + p)[active_zero_der] / 2.0
            active &= nz_der  # don't assign zero derivatives again
            # failures[nz_der] = np.abs(dp) >= tol  # not yet converged
            failure = ((dp - tol) ** 2).mean() > tol
            # stop iterating if there aren't any failures, not incl zero der
            if not failure:
                break
            p1, p = p, p1
            q0 = q1
            q1 = func(p1, *args)

    return p, iteration, q1


@nb.njit(cache=True)
def root_secant(f, x0, h, yprev, input_args, tol=1e-3, max_iter=100):
    """
    Solve for root using the secant method.

    This is a pure basic secant method without approximation of the Jacobian.

    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    x0 : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.
    yprev : TYPE
        DESCRIPTION.
    input_args : TYPE
        DESCRIPTION.
    tol : TYPE, optional
        DESCRIPTION. The default is 1e-3.
    max_iter : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # set bracket to +-10% of starting point
    p0 = x0 * (1 + 1e-1)
    p1 = x0 * (1 - 1e-1)

    # store y values instead of recomputing them
    fp0 = f(p0, yprev, h, input_args)
    fp1 = f(p1, yprev, h, input_args)

    false_mask = fp0 * fp1 >= 0
    p0[false_mask], p1[false_mask] = p1[false_mask], p0[false_mask]

    # gt_eps = np.ones_like(false_mask, dtype=np.bool)
    eps = 1e-8  # np.finfo(np.float64).eps * 1e4
    # succesful vars:
    # tol_ok = np.abs(fp1) <= tol

    # iterate up to maximum number of times
    for _ in range(max_iter):
        # see whether the answer has converged (MSE)
        if ((fp1 - tol) ** 2).mean() < tol:
            return p1
        # check if epsilon is reached or no diff
        gt_eps = (np.abs(fp1) > eps) | (np.abs(fp0) > eps) | (fp0 != fp1)
        # do calculation
        p2 = (p0 * fp1 - p1 * fp0) / (fp1 - fp0)
        # shift variables (prepare for next loop) and except lower eps values
        p0[gt_eps], p1[gt_eps] = p1[gt_eps], p2[gt_eps]
        # shift for next step
        fp0, fp1 = fp1, f(p1, yprev, h, input_args)
    return p1  # return if not converged
