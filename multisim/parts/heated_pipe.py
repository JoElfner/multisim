# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:22:53 2018

@author: Johannes
"""

import numpy as np
import pandas as pd

from .pipe import Pipe
from ..precomp_funs import heatedpipe1D_diff


class HeatedPipe(Pipe):
    """
    type: Single Pipe

    Can be added to the simulation environment by using the following method:
        .add_part(Pipe, name, volume=..., grid_points=..., outer_diameter=...,
        shell_thickness=...)
    Part creation parameters:
    -------------------------
    Pipe : class name
        This parameter can\'t be changed and needs to be passed exactly like
        this.
    name : string
        Pipe identifier as string. Needs to be unique.
    length : integer, float
        Thermal energy storage volume in [m^3].
    grid_points : integer
        Number of grid points to discretize the pipe with.

    For part initialization the following additional parameters need to be
    passed:
        .init_part(insulation_thickness=..., insulation_lambda=...)

    Part initialization parameters:
    -------------------------------
    insulation_thickness : float
        in [m]
    insulation_lambda : float
        in [W/(m*K)]

    """

    def __init__(self, name, master_cls, **kwargs):
        self.constr_type = 'HeatedPipe'  # define construction type
        # since this part is a subclass of Pipe, initialize Pipe:
        super().__init__(
            name, master_cls, **kwargs, constr_type=self.constr_type
        )

        # single cell array for heat flow rate:
        self._dQ_heating = np.zeros(1, dtype=np.float64)
        # result array for heating:
        self.res_dQ = np.zeros((1, 1), dtype=np.float64)

        # differential of temperature due to heating:
        self.dT_heat = np.zeros_like(self.T)

        # if the part CAN BE controlled by the control algorithm:
        self.is_actuator = True
        self._actuator_CV = self._dQ_heating[:]  # set array to be controlled
        self._actuator_CV_name = 'rate_of_heat_flow'
        self._unit = '[W]'  # set unit of control variable
        # if the part HAS TO BE controlled by the control algorithm:
        self.control_req = True
        # if the part needs a special control algorithm (for parts with 2 or
        # more controllable inlets/outlets/...):
        self.actuator_special = False
        # initialize bool if control specified:
        self.ctrl_defined = False

        # get heat spread
        err_str = (
            self._base_err
            + self._arg_err.format('heat_spread')
            + 'The heat spread has to be given with `heat_spread=X`, where '
            'X is one of the following:\n'
            '    - \'all\': The rate of heat flow is spread equally on all '
            'cells.\n'
            '    - \'single\': One single cell is heated with the total rate '
            'of heat flow. This may cause the simulation stepsize to decrease '
            'to very small stepsizes if the thermal intertia is low compared '
            'to the rate of heat flow.\n'
            '    - \'range\': The rate of heat flow is spread equally on a '
            'range of consecutive cells.'
        )
        assert 'heat_spread' in kwargs and kwargs['heat_spread'] in [
            'all',
            'single',
            'range',
        ], err_str
        self._heat_spread = kwargs['heat_spread']
        # get other parameters depending on heated cells:
        if self._heat_spread == 'all':  # all cells heated equally
            # cell wise rate of heat flow multiplicator:
            self._heat_mult = 1 / self.num_gp
            # set slice to full array:
            self._heated_cells = slice(0, self.num_gp)
            self._num_heated_cells = self.num_gp
        elif self._heat_spread == 'single':  # single cell heated
            # get slice index to heated cells:
            err_str = (
                self._base_err
                + self._arg_err.format('heated_cells')
                + 'The heated cells have to be given with `heated_cells=X`. '
                'Since `heat_spread=\'single\'` is set, X has to be an '
                'integer index to the target cell for the heat flow in the '
                'range of `0 <= X <= ' + str(self.num_gp - 1) + '`.'
            )
            assert (
                'heated_cells' in kwargs
                and isinstance(kwargs['heated_cells'], int)
                and 0 <= kwargs['heated_cells'] <= (self.num_gp - 1)
            ), err_str
            self._heated_cells = slice(
                kwargs['heated_cells'], kwargs['heated_cells'] + 1
            )
            self._num_heated_cells = 1
            # cell wise rate of heat flow multiplicator:
            self._heat_mult = 1.0
        else:  # range of cells cells heated equally
            # get slice index to heated cells:
            err_str = (
                self._base_err
                + self._arg_err.format('heated_cells')
                + 'The heated cells have to be given with `heated_cells=X`. '
                'Since `heat_spread=\'range\'` is set, X has to be a range of '
                'target cells for the heat flow, given as a list with '
                '`X=[start, end]` where both values are integer values. '
                'Additionally `start < end` and `0 <= start/end <= '
                + str(self.num_gp - 1)
                + '` must be true.\n'
                'As always with indexing in python/numpy, the end-index is '
                'NOT included in the selection. Thus `X=[2, 4]` will heat '
                'cells 2 and 3, but NOT cell 4.'
            )
            assert (
                'heated_cells' in kwargs
                and isinstance(kwargs['heated_cells'], (list, tuple))
                and len(kwargs['heated_cells']) == 2
                and isinstance(kwargs['heated_cells'][0], int)
                and isinstance(kwargs['heated_cells'][1], int)
            ), err_str
            start = kwargs['heated_cells'][0]
            end = kwargs['heated_cells'][1]
            # assert correct indices:
            assert start < end and start >= 0 and end < self.num_gp, err_str
            self._heated_cells = slice(start, end)
            self._num_heated_cells = end - start
            # cell wise rate of heat flow multiplicator
            self._heat_mult = 1 / self._num_heated_cells

        # create view to dT for easy access to heated cells:
        self._dT_heated = self.dT_heat[self._heated_cells]
        # view to heated cell's m*cp value:
        self._mcp_heated = self._mcp[self._heated_cells]

        err_str = (
            self._base_err
            + self._arg_err.format('lower_limit, upper_limit')
            + 'The part was set to be an actuator and need a control with '
            '`no_control=False`, thus `lower_limit` and `upper_limit` '
            'in {0} have to be passed to clip the controller action on '
            'the actuator to the limits.\n'
            'The limits have to be given as integer or float values with '
            '`lower_limit < upper_limit`.'
        ).format(self._unit)
        assert 'lower_limit' in kwargs and 'upper_limit' in kwargs, err_str
        self._lims = np.array(  # set limits to array
            [kwargs['lower_limit'], kwargs['upper_limit']], dtype=np.float64
        )
        self._llim = self._lims[0]  # also save to single floats
        self._ulim = self._lims[1]  # also save to single floats
        assert 0 <= self._lims[0] < self._lims[1], (
            err_str + ' For HeatedPipe limits are additionally restricted '
            'to `0 <= lower_limit < upper_limit`.'
        )

        # IMPORTANT: THIS VARIABLE **MUST NOT BE INHERITED BY SUB-CLASSES**!!
        # If sub-classes are inherited from this part, this bool checker AND
        # the following variables MUST BE OVERWRITTEN!
        # ist the diff function fully njitted AND are all input-variables
        # stored in a container?
        self._diff_fully_njit = False
        # self._diff_njit = pipe1D_diff  # handle to njitted diff function
        # input args are created in simenv _create_diff_inputs method

    def init_part(self, **kwargs):
        # since this part is a subclass of Pipe, call init_part of Pipe:
        super().init_part(**kwargs)

    def _special_array_init(self, num_steps):
        self.res_dQ = np.zeros((num_steps + 1,) + self._dQ_heating.shape)

    def __depcreated_special_free_memory(
        self, disk_store, part, array_length, hdf_idx, stepnum
    ):
        disk_store['store_tmp'].append(
            part + '/heating',
            pd.DataFrame(data=self.res_dQ[:array_length, ...], index=hdf_idx),
        )
        # set current result to row 0 of array and clear the rest:
        self.res_dQ[0, ...] = self.res_dQ[stepnum[0], ...]
        self.res_dQ[1:, ...] = 0.0

    def _special_free_memory(
        self, disk_store, part, array_length, hdf_idx, stepnum
    ):
        disk_store.append(
            part + '/heating',
            pd.DataFrame(data=self.res_dQ[:array_length, ...], index=hdf_idx),
        )
        # set current result to row 0 of array and clear the rest:
        self.res_dQ[0, ...] = self.res_dQ[stepnum[0], ...]
        self.res_dQ[1:, ...] = 0.0

    def _check_results(self):
        """Check results for validity."""
        assert np.all(self.res > 0.0) and np.all(
            self.res < 150
        ), 'Temperature range exceeding 0. < theta < 150.'

    def _final_datastore_backcalc(self, disk_store, part):
        self.res_dQ = np.append(
            disk_store['store_tmp'][part + '/heating'].values,
            self.res_dQ,
            axis=0,
        )

    def _special_crop_results(self, stepnum):
        self.res_dQ = self.res_dQ[0:stepnum]

    def _special_return_store(self, disk_store, dfs, part):
        dfs[part].update(
            {'heating': disk_store['store'][part + '/heating'].copy()}
        )

    def get_diff(self, timestep):
        """
        Call the jitted differential function."""
        (
            self.dT_total,
            self._process_flows,
            self._models._step_stable,
            self._models._vN_max_step,
            self._models._max_factor,
        ) = heatedpipe1D_diff(
            T_ext=self._T_ext,
            T_port=self._T_port,
            T_s=self._T_s,
            T_amb=self._T_amb,
            ports_all=self._models.ports_all,  # temperatures
            dm_io=self._dm_io,
            dm_top=self._dm_top,
            dm_bot=self._dm_bot,
            dm_port=self._dm_port,
            dQ_heating=self._dQ_heating,
            res_dm=self.res_dm,
            res_dQ=self.res_dQ,  # flows
            cp_T=self._cp_T,
            lam_T=self._lam_T,
            rho_T=self._rho_T,
            ny_T=self._ny_T,
            lam_mean=self._lam_mean,
            cp_port=self._cp_port,
            lam_port_fld=self._lam_port_fld,
            mcp=self._mcp,
            mcp_heated=self._mcp_heated,
            rhocp=self._rhocp,
            lam_wll=self._lam_wll,
            lam_ins=self._lam_ins,
            mcp_wll=self._mcp_wll,
            ui=self._ui,  # material properties
            alpha_i=self._alpha_i,
            alpha_inf=self._alpha_inf,  # alpha values
            UA_tb=self._UA_tb,
            UA_tb_wll=self._UA_tb_wll,
            UA_amb_shell=self._UA_amb_shell,
            UA_port=self._UA_port,
            UA_port_wll=self._UA_port_wll,  # UA values
            port_own_idx=self._port_own_idx,
            port_link_idx=self._port_link_idx,  # indices
            heat_mult=self._heat_mult,
            grid_spacing=self.grid_spacing,
            port_gsp=self._port_gsp,
            port_subs_gsp=self._port_subs_gsp,
            d_i=self._d_i,
            cell_dist=self._cell_dist,
            flow_length=self._flow_length,  # lengths
            r_total=self._r_total,
            r_ln_wll=self._r_ln_wll,
            r_ln_ins=self._r_ln_ins,
            r_rins=self._r_rins,  # lengths
            A_cell=self.A_cell,
            V_cell=self.V_cell,
            A_shell_i=self._A_shell_i,
            A_shell_ins=self._A_shell_ins,
            A_p_fld_mean=self._A_p_fld_mean,  # areas and vols
            process_flows=self._process_flows,
            vertical=self._vertical,
            step_stable=self._models._step_stable,  # bools
            part_id=self.part_id,
            stability_breaches=self._stability_breaches,
            vN_max_step=self._models._vN_max_step,
            max_factor=self._models._max_factor,  # misc
            stepnum=self.stepnum,
            timestep=timestep,  # step information
            dT_cond=self.dT_cond,
            dT_adv=self.dT_adv,
            dT_heat=self.dT_heat,
            dT_heated=self._dT_heated,  # differentials
        )

        return self.dT_total
