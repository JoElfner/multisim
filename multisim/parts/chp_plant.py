# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Oct 2019
"""

import numpy as np
import pandas as pd

from .pipe import Pipe
from .. import precomp_funs as _pf


class CHPPlant(Pipe):
    """
    Construct a CHP plant.

    Can be added to the simulation environment by using the following method:
        .add_part(Pipe, name, volume=..., grid_points=..., outer_diameter=...,
        shell_thickness=...)

    Part creation parameters
    ------------------------
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

    Part initialization parameters
    ------------------------------
    insulation_thickness : float
        in [m]
    insulation_lambda : float
        in [W/(m*K)]

    """

    def __init__(
        self,
        name,
        master_cls,
        eta_el=0.32733,
        p2h_ratio=0.516796,
        max_slope_el=0.05,
        min_off_time=600,
        min_on_time=600,
        chp_on_at_start=False,
        **kwds
    ):
        self.constr_type = 'CHPPlant'  # define construction type

        # since this part is a subclass of Pipe, initialize Pipe:
        super().__init__(
            name, master_cls, **kwds, constr_type=self.constr_type
        )

        # check for default args, print messages and set them to kwds
        defaults = (
            ('p2h_ratio', 0.516796),
            ('eta_el', 0.32733),
            ('max_slope_el', 0.05),
            ('min_off_time', 600),
            ('min_on_time', 600),
        )
        base_dflt_str = (
            '    ---> CHP plant {0}: No `{1}` given. '
            'Using default value of `{2:.3f}`.'
        )
        if not self._models.suppress_printing:
            print(
                'Setting defaults. Set a value to `None` to get more '
                'information on effect, type and range.'
            )
        for dflt_nm, dflt_val in defaults:
            if dflt_val == locals()[dflt_nm]:
                if dflt_val is not None:  # only if not None, else print err
                    if not master_cls.suppress_printing:
                        print(
                            base_dflt_str.format(self.name, dflt_nm, dflt_val)
                        )
                    kwds[dflt_nm] = dflt_val
            else:
                kwds[dflt_nm] = locals()[dflt_nm]

        # define arguments and errors:
        self._aae = {  # arguments and errors
            'power_electrical': (
                'CHP plant electrical power output in [W]. Type: int, float. '
                'Range: X > 0'
            ),
            'p2h_ratio': (
                'CHP plant power to heat ratio. Type: float. '
                'Range: 0 < X < 1'
            ),
            'eta_el': (
                'CHP plant electrical efficiency. Type: float. '
                'Range: 0 < X < 1'
            ),
            'modulation_range': (
                'CHP plant electrical modulation range. Type: list, tuple.'
            ),
            'max_slope_el': (
                'CHP plant maximum electrical ramp in %/s. Type: float. '
                'Range: 0 < X <= 1'
            ),
            'min_off_time': (
                'CHP plant minimum consecutive off time in [s] before '
                'switching on is allowed. Type: int, float. Range: X >= 0.'
            ),
            'min_on_time': (
                'CHP plant minimum consecutive on time in [s] before '
                'switching off is allowed. Plant will be kept at min. '
                'modulation if shutdown is requested until min. on time is '
                'reached. Emergency shutdown due to overtemperature will '
                'override this timer. Type: int, float. Range: X >= 0.'
            ),
            'heat_spread': (
                'CHP plant heat spread, that is if a \'single\' cell, a '
                'cell \'range\' or \'all\' cells of the CHP plant are heated '
                'by the thermal power output. Type: str.'
            ),
            'lower_limit': (
                'Lower limit for controller action on the modulation. '
                'Type: int, float. Range: X >= 0.'
            ),
            'upper_limit': (
                'Upper limit for controller action on the modulation. '
                'Type: int, float. Range: X <= 1.'
            ),
        }
        # check for arguments:
        self._print_arg_errs(self.constr_type, name, self._aae, kwds)

        assert isinstance(kwds['power_electrical'], (int, float)) and (
            kwds['power_electrical'] > 0.0
        ), (
            self._base_err
            + self._arg_err.format('power_electrical')
            + self._aae['power_electrical']
        )
        self.chp_power_el = kwds['power_electrical']
        assert isinstance(kwds['p2h_ratio'], float) and (
            0.0 < kwds['p2h_ratio'] < 1.0
        ), (
            self._base_err
            + self._arg_err.format('p2h_ratio')
            + self._aae['p2h_ratio']
        )
        self.p2h_ratio = kwds['p2h_ratio']
        # base p2h-ratio of the CHP plant is about .5, the exact ratio is
        # given by:
        self._p2h_base = 1 / _pf.chp_thermal_power(1.0)
        # since this is already included in the polynomials, the given p2h
        # and the base p2h must be combined into an additional factor:
        self._p2h_ratio = self._p2h_base / self.p2h_ratio
        # electrical efficiency:
        assert isinstance(kwds['eta_el'], float) and (
            0.0 < kwds['eta_el'] < 1.0
        ), (
            self._base_err
            + self._arg_err.format('eta_el')
            + self._aae['eta_el']
        )
        self.eta_el = kwds['eta_el']
        # to get the correct gas consumption given by polynomials in dependency
        # of Pel, an additional conversion factor must be calculated. The base
        # eta el used in the polynomials is
        self._eta_el_base = 1 / _pf.chp_gas_power(1.0)
        self._eta_el_fac = self._eta_el_base / self.eta_el
        # get thermal efficiency from eta el
        self.eta_th = self.eta_el / self.p2h_ratio
        if 'eta_sum' not in kwds:
            eta_sum = 0.961
            assert (self.eta_el + self.eta_th) <= eta_sum, (
                self._base_err
                + self._arg_err.format('eta_el + eta_th')
                + 'The current total CHP plant efficiency is {0:.4f}, but must '
                'be <={1:.3f}. If a higher total efficiency '
                'shall be set, adjust the `eta_sum` parameter.'.format(
                    self.eta_el + self.eta_th, eta_sum
                )
            )
        else:
            assert (
                isinstance(kwds['eta_sum'], (int, float))
                and kwds['eta_sum'] > 0
            )
            assert (self.eta_el + self.eta_th) < kwds['eta_sum'], (
                self._base_err
                + self._arg_err.format('eta_el + eta_th')
                + 'The total CHP plant efficiency must be lower than `eta_sum`.'
            )
        # save thermal and gas power of chp plant at max power output
        self.chp_power_th = self.chp_power_el / self.p2h_ratio
        self.chp_power_gas = self.chp_power_el / self.eta_el

        # get modulation range:
        err_mod = (
            'The valid CHP plant modulation range has to be given with '
            '`modulation_range=(lower, upper)` where lower and upper '
            'represent the lower and upper modulation bounds, f.i. '
            '`(.5, 1)` if the CHP modulation can be set between 50% and '
            '100%. Values must be in the range of 0 < x <= 1.'
        )
        assert isinstance(kwds['modulation_range'], (tuple, list)), err_mod
        self._mod_range = kwds['modulation_range']
        self._mod_lower = float(self._mod_range[0])
        self._mod_upper = float(self._mod_range[1])
        assert isinstance(self._mod_lower, (int, float)) and (
            0 < self._mod_lower <= 1
        ), err_mod
        assert (
            isinstance(self._mod_upper, (int, float))
            and (0 < self._mod_upper <= 1)
            and (self._mod_lower < self._mod_upper)
        ), err_mod

        assert isinstance(kwds['max_slope_el'], float) and (
            0 < kwds['max_slope_el'] <= 1
        ), (
            self._base_err
            + self._arg_err.format('max_slope_el')
            + self._aae['max_slope_el']
        )
        self._max_ramp_el = kwds['max_slope_el']
        self._max_ramp_th = 0.025  # 2.5%/s, NOT USED
        assert isinstance(kwds['min_off_time'], (int, float)) and (
            0 < kwds['min_off_time']
        ), (
            self._base_err
            + self._arg_err.format('min_off_time')
            + self._aae['min_off_time']
        )
        self._min_off_time = kwds['min_off_time']
        assert isinstance(kwds['min_on_time'], (int, float)) and (
            0 < kwds['min_on_time']
        ), (
            self._base_err
            + self._arg_err.format('min_on_time')
            + self._aae['min_on_time']
        )
        self._min_on_time = kwds['min_on_time']
        self._T_chp_in_max = 75.0
        self._T_chp_in_max_emrgncy = 110.0  # immediate shutdown if >temp.

        # single cell array for heat flow rate:
        self._dQ_heating = np.zeros(1, dtype=np.float64)
        # result array for heating:
        self.res_dQ = np.zeros((1, 1), dtype=np.float64)
        # same for electric power:
        self._Pel = np.zeros(1, dtype=np.float64)
        self.res_Pel = np.zeros(1, dtype=np.float64)
        # and gas consumption
        self._Pgas = np.zeros(1, dtype=np.float64)
        self.res_Pgas = np.zeros(1, dtype=np.float64)

        # checker if chp plant is on, in startup, value for time passed since
        # startup, value for time passed since shutoff,
        # time in [s] to account for remaining heat after a recent shutdown,
        # array for current modulation of CHP plant
        self._chp_on = False
        self._chp_state = np.zeros(1, dtype=np.bool)  # vector to save states
        self._shutdown_in_progress = False
        self._startup_duration = 0.0
        self._off_duration = 0.0
        self._startup_at_time = 0
        # has never been switched on. min off time -1 to enable switching on
        # at the start of the sim.
        self._shutdown_at_time = -self._min_off_time - 1.0
        self._remaining_heat = 0.0
        self._power_modulation = np.zeros(1, dtype=np.float64)
        # max. overtemperature time. dt is the consecutive time since first
        # overtemp., max_temp_exc_time the maximum time allowed before shutdown
        self._dt_time_temp_exc = 0.0
        # TODO: print('das noch durch kwds:')
        self._max_temp_exc_time = 2 * 60
        # startup and shutdown factors for thermal and el (only startup, since
        # only heat can remain...) power
        if not chp_on_at_start:  # if CHP was switched off at the beginning
            self._startup_factor_el = 0.0
            self._startup_factor_th = 0.0
            self._startup_factor_gas = 0.0
            self._chp_on = False  # chp is NOT on
            self._startup_in_progress = True  # chp was off, thus startup req.
            self._shutdown_duration = 99999.0
            self._bin_pow_fac = 0
        else:
            self._startup_factor_el = 1.0
            self._startup_factor_th = 1.0
            self._startup_factor_gas = 1.0
            # has never been switched on. min on time to enable switching off
            # at the start of the sim.
            self._startup_at_time = -self._min_on_time  # has been switched on
            # has never been switched off
            self._shutdown_at_time = self._startup_at_time - 1
            self._chp_on = True  # chp IS on
            # chp was ON, thus NO startup required!
            self._startup_in_progress = False
            self._shutdown_duration = 99999.0
            self._bin_pow_fac = 1
        self._shutdown_factor_th = 0.0
        # save last thermal power value before shutdown for shutdown process
        self._last_dQ = np.zeros_like(self._dQ_heating)
        # percentage values, at which the CHP plant is considered as on/off
        self._chp_on_perc = 0.999
        self._chp_off_perc = 0.01
        # power factors for on/off switching for Pth (on and off) and Pel
        # (only on):
        self._pth_power_factor = 0.0
        self._pel_power_factor = 0.0

        # differential of temperature due to heating:
        self.dT_heat = np.zeros_like(self.T)
        # memoryview to the inlet cell for max. temperature checking
        self._T_chp_in = self.T[0:1]

        # if the part CAN BE controlled by the control algorithm:
        self.is_actuator = True
        self._actuator_CV = self._power_modulation[:]  # set to be controlled
        self._actuator_CV_name = 'el_power_modulation'
        self._unit = '[%]'  # set unit of control variable
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
        assert kwds['heat_spread'] in ['all', 'single', 'range'], err_str
        self._heat_spread = kwds['heat_spread']
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
                'heated_cells' in kwds
                and type(kwds['heated_cells']) == int
                and 0 <= kwds['heated_cells'] <= (self.num_gp - 1)
            ), err_str
            self._heated_cells = slice(
                kwds['heated_cells'], kwds['heated_cells'] + 1
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
                'heated_cells' in kwds
                and isinstance(kwds['heated_cells'], (list, tuple))
                and len(kwds['heated_cells']) == 2
                and isinstance(kwds['heated_cells'][0], int)
                and isinstance(kwds['heated_cells'][1], int)
            ), err_str
            start = kwds['heated_cells'][0]
            end = kwds['heated_cells'][1]
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
        self._lims = np.array(  # set limits to array
            [kwds['lower_limit'], kwds['upper_limit']], dtype=np.float64
        )
        self._llim = self._lims[0]  # also save to single floats
        self._ulim = self._lims[1]  # also save to single floats
        assert 0 <= self._lims[0] < self._lims[1], (
            err_str + ' For HeatedPipe limits are additionally restricted '
            'to `0 <= lower_limit < upper_limit`.'
        )

        # precalc arrays needed for shutdown-startup-procedure:
        self._tsteps_startup = np.arange(100000)  # check for up to 1e5 s
        # thermal power factor during startup
        self._startuptsteps = _pf.chp_startup_th(self._tsteps_startup)
        # find step at which full power is reached. idx is also in seconds!
        idx_on = np.argmin(np.abs(self._startuptsteps - self._chp_on_perc))
        # cut arrays to that size:
        self._tsteps_startup = self._tsteps_startup[1 : idx_on + 1]
        self._startuptsteps = self._startuptsteps[1 : idx_on + 1]

        # IMPORTANT: THIS VARIABLE **MUST NOT BE INHERITED BY SUB-CLASSES**!!
        # If sub-classes are inherited from this part, this bool checker AND
        # the following variables MUST BE OVERWRITTEN!
        # ist the diff function fully njitted AND are all input-variables
        # stored in a container?
        self._diff_fully_njit = False
        # self._diff_njit = pipe1D_diff  # handle to njitted diff function
        # input args are created in simenv _create_diff_inputs method

    def init_part(self, start_modulation=0, fluegas_flow=70 / 61.1, **kwds):
        """Initialize part. Do stuff which requires built part dict."""
        # since this part is a subclass of Pipe, call init_part of Pipe:
        super().init_part(**kwds)

        # check for default args, print messages and set them to kwds
        if fluegas_flow == 70 / 61.1:
            if not self._models.suppress_printing:
                print(
                    '    ---> CHP plant {0}: No `{1}` given. '
                    'Using default value of `{2:.3f}Nm3/kWh`.'.format(
                        self.name, 'fluegas_flow', fluegas_flow
                    )
                )
        kwds['fluegas_flow'] = fluegas_flow

        # define arguments and errors:
        self._aaei = {  # arguments and errors
            'connect_fluegas_hex': (
                'Connect a flue gas heat exchanger directly to the CHP plant? '
                'Exhaust gas flow will be passed to the HEX directly, wihout '
                'need for pipes and pumps. Type: bool.'
            ),
            # 'fluegas_flow_at_pmax': (
            #     'Flue gas flow at maximum power output (100% modulation) in '
            #     '[Nm3/h]. Type: int, float. Range: X > 0')
            'fluegas_flow': (
                'Specific flue gas flow relative to the gas consumption in '
                '[Nm3/kWh], f.i. 1.146 for 70Nm3 per 61.1kW gas consumption '
                '(lower heating vlaue). Type: int, float. Range: X > 0'
            ),
        }
        # check for arguments:
        self._print_arg_errs(self.constr_type, self.name, self._aaei, kwds)

        # check modulation parameters
        assert isinstance(start_modulation, (int, float)) and (
            0 <= start_modulation <= 1
        ), 'Start modulation muss zwischen 0 und 1 liegen.'
        self._actuator_CV[:] = start_modulation
        # initialize the actuator
        self._initialize_actuator(variable_name='_power_modulation', **kwds)

        # if CHP is feeding a fluegas HEX:
        assert isinstance(kwds['connect_fluegas_hex'], bool), (
            self._base_err
            + self._arg_err.format('connect_fluegas_hex')
            + self._aaei['connect_fluegas_hex']
        )
        self._connect_fg_hex = kwds['connect_fluegas_hex']
        if self._connect_fg_hex:
            err_hex2 = (
                self._base_err
                + self._arg_err.format('fg_hex_name')
                + 'Part name of the fluegas heat exchanger to connect.'
            )
            assert 'fg_hex_name' in kwds, err_hex2
            err_hex3 = self._base_err + self._arg_err.format(
                'fg_hex_name'
            ) + 'The fluegas heat exchanger with the name `{0}` was not ' 'found.'.format(
                kwds['fg_hex_name']
            )
            assert kwds['fg_hex_name'] in self._models.parts, err_hex3
            self._fg_hex_name = kwds['fg_hex_name']
            # save ref to part:
            self._fg_hex = self._models.parts[self._fg_hex_name]
            # save view to fg hex gas volume flow cell to save values to
            # (in Nm3/s):
            self._fg_hex_gdv = self._fg_hex._dm_io[slice(1, 2)]
            # set flow channel in this part to solved:
            self._models.parts[self._fg_hex_name]._solved_ports.extend(
                ['fluegas_in', 'fluegas_out']
            )
        else:  # fill values with dummies if no hex connected
            self._fg_hex_gdv = np.zeros(1, dtype=np.float64)

        # get flue gas flow:
        assert isinstance(kwds['fluegas_flow'], (int, float)), (
            self._base_err
            + self._arg_err.format('fluegas_flow')
            + self._aaei['fluegas_flow']
        )
        # specific flue gas flow in Nm3/kWh
        self.fluegas_flow_specific = kwds['fluegas_flow']
        # save in Nm3/s
        # self._fg_dv_at_pmax = kwds['fluegas_flow'] / 3600
        self._fg_dv_at_pmax = (
            self.chp_power_gas * self.fluegas_flow_specific / 3.6e6
        )

        assert 0 <= self._lims[0] < self._lims[1], (
            self._base_err
            + self._arg_err.format('lower_limit, upper_limit')
            + ' For a CHP plant limits are additionally restricted '
            'to `0 <= lower_limit < upper_limit`.'
        )
        # get maximum gas power at modulation == 1:
        self._Pgas_max = (
            _pf.chp_gas_power(1.0) * self.chp_power_el * self._eta_el_fac
        )

    def _special_array_init(self, num_steps):
        self.res_dQ = np.zeros((num_steps + 1,) + self._dQ_heating.shape)
        self.res_Pel = np.zeros((num_steps + 1,) + self._Pel.shape)
        self.res_Pgas = np.zeros((num_steps + 1,) + self._Pgas.shape)
        self._chp_state = np.zeros((num_steps + 1,) + self._chp_state.shape)

    def __deprecated_special_free_memory(
        self, disk_store, part, array_length, hdf_idx, stepnum
    ):
        disk_store['store_tmp'].append(
            part + '/heating',
            pd.DataFrame(data=self.res_dQ[:array_length, ...], index=hdf_idx),
        )
        disk_store['store_tmp'].append(
            part + '/Pel',
            pd.DataFrame(data=self.res_Pel[:array_length, ...], index=hdf_idx),
        )
        disk_store['store_tmp'].append(
            part + '/Pgas',
            pd.DataFrame(
                data=self.res_Pgas[:array_length, ...], index=hdf_idx
            ),
        )
        disk_store['store_tmp'].append(
            part + '/chp_state',
            pd.DataFrame(
                data=self._chp_state[:array_length, ...], index=hdf_idx
            ),
        )
        # set current result to row 0 of array and clear the rest:
        self.res_dQ[0, ...] = self.res_dQ[stepnum[0], ...]
        self.res_dQ[1:, ...] = 0.0
        self.res_Pel[0, ...] = self.res_Pel[stepnum[0], ...]
        self.res_Pel[1:, ...] = 0.0
        self.res_Pgas[0, ...] = self.res_Pgas[stepnum[0], ...]
        self.res_Pgas[1:, ...] = 0.0
        self._chp_state[0, ...] = self._chp_state[stepnum[0], ...]
        self._chp_state[1:, ...] = 0.0

    def _special_free_memory(
        self, disk_store, part, array_length, hdf_idx, stepnum
    ):
        disk_store.append(
            part + '/heating',
            pd.DataFrame(data=self.res_dQ[:array_length, ...], index=hdf_idx),
        )
        disk_store.append(
            part + '/Pel',
            pd.DataFrame(data=self.res_Pel[:array_length, ...], index=hdf_idx),
        )
        disk_store.append(
            part + '/Pgas',
            pd.DataFrame(
                data=self.res_Pgas[:array_length, ...], index=hdf_idx
            ),
        )
        disk_store.append(
            part + '/chp_state',
            pd.DataFrame(
                data=self._chp_state[:array_length, ...], index=hdf_idx
            ),
        )
        # set current result to row 0 of array and clear the rest:
        self.res_dQ[0, ...] = self.res_dQ[stepnum[0], ...]
        self.res_dQ[1:, ...] = 0.0
        self.res_Pel[0, ...] = self.res_Pel[stepnum[0], ...]
        self.res_Pel[1:, ...] = 0.0
        self.res_Pgas[0, ...] = self.res_Pgas[stepnum[0], ...]
        self.res_Pgas[1:, ...] = 0.0
        self._chp_state[0, ...] = self._chp_state[stepnum[0], ...]
        self._chp_state[1:, ...] = 0.0

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
        self.res_Pel = np.append(
            disk_store['store_tmp'][part + '/Pel'].values, self.res_Pel, axis=0
        )
        self.res_Pgas = np.append(
            disk_store['store_tmp'][part + '/Pgas'].values,
            self.res_Pgas,
            axis=0,
        )
        self._chp_state = np.append(
            disk_store['store_tmp'][part + '/chp_state'].values,
            self._chp_state,
            axis=0,
        )

    def _special_crop_results(self, stepnum):
        self.res_dQ = self.res_dQ[0:stepnum]
        self.res_Pel = self.res_Pel[0:stepnum]
        self.res_Pgas = self.res_Pgas[0:stepnum]
        self._chp_state = self._chp_state[0:stepnum]

    def _special_return_store(self, disk_store, dfs, part):
        dfs[part].update(
            {
                'heating': disk_store['store'][part + '/heating'].copy(),
                'Pel': disk_store['store'][part + '/Pel'].copy(),
                'Pgas': disk_store['store'][part + '/Pgas'].copy(),
                'chp_state': disk_store['store'][part + '/chp_state'].copy(),
            }
        )

    def get_diff(self, timestep):
        """Calculate differential of CHP core part."""
        # process flows is only executed ONCE per timestep, afterwards the bool
        # process_flows is set to False.
        if self._process_flows[0]:  # only if flows not already processed
            # get current elapsed time
            curr_time = self._models.time_vec[self.stepnum[0] - 1] + timestep
            # get state of the last step
            state_last_step = self._chp_state[self.stepnum[0] - 1]

            # check for modulation range and set on-off-integer:
            if self._power_modulation[0] < self._mod_lower:
                # binary power multiplication factor to enable off-state
                # for modulations < mod_lower, f.i. to avoid modulations below
                # 50%.
                self._bin_pow_fac = 0.0
                self._chp_on = False  # chp is off
            else:
                self._bin_pow_fac = 1.0
                self._chp_on = True  # chp is on

            # detect changes in the state to save start/stop times
            if (state_last_step != 0.0) != self._chp_on:
                if not self._chp_on:  # if last step chp was on and now off
                    # assert that minimum run time is fullfilled. if not,
                    # avoid switching off by keeping min. modulation
                    if self._min_on_time > (curr_time - self._startup_at_time):
                        # if minimum run time not reached, set chp to on
                        self._bin_pow_fac = 1.0
                        self._chp_on = True
                        self._power_modulation[0] = self._mod_lower
                    else:  # else allow shutdown
                        self._shutdown_at_time = curr_time  # chp was shutdown
                        self._shutdown_in_progress = True
                else:  # if last step chp was off and now it is on
                    # assert that minimum off time is fulfilled AND
                    # (both ok -> OR statetment) inlet temp. is not exceeding
                    # max temp.. If any is True, avoid CHP startup
                    if (
                        (
                            self._min_off_time
                            > (curr_time - self._shutdown_at_time)
                        )
                        or (self._T_chp_in[0] > self._T_chp_in_max)
                        or np.any(self.T > self._T_chp_in_max_emrgncy)
                    ):
                        # if minimum off time not reached or temperature too
                        # high, set chp to off
                        self._bin_pow_fac = 0.0
                        self._chp_on = False
                        self._power_modulation[0] = 0.0
                    else:  # else allow switching on
                        self._startup_at_time = curr_time  # chp was started
                        self._startup_in_progress = True
            elif self._chp_on:
                # if CHP was on last step AND is on now, check for ramps
                # get difference of modulation and absolute ramp per second
                mod_diff = state_last_step - self._power_modulation[0]
                mod_ramp_abs = np.abs(mod_diff) / timestep
                # if absolute ramp is higher than max ramp, limit change to
                # ramp
                if mod_ramp_abs > self._max_ramp_el:
                    if mod_diff <= 0.0:  # ramp up too fast
                        self._power_modulation[0] = (  # set ramp to max ramp
                            state_last_step + self._max_ramp_el * timestep
                        )
                    else:  # ramp down too fast
                        self._power_modulation[0] = (  # set ramp to max ramp
                            state_last_step - self._max_ramp_el * timestep
                        )
            # if chp is on, check if inlet temperature was exceeded or any
            # temperature is above emergency shutdown temp., then shutdown
            if self._chp_on and (
                (self._T_chp_in[0] > self._T_chp_in_max)
                or np.any(self.T > self._T_chp_in_max_emrgncy)
            ):
                # if max inlet temp. is exceeded, check max. allowed time for
                # exceeding and if too large, shutdown CHP due to overtemp.,
                # independend of min. run times and other parameters.
                # also if inlet temp. is above an emergency threshold.
                if (
                    self._dt_time_temp_exc > self._max_temp_exc_time
                ) or np.any(self.T > self._T_chp_in_max_emrgncy):
                    self._power_modulation[0] = 0.0
                    self._bin_pow_fac = 0.0
                    self._chp_on = False
                    self._shutdown_at_time = curr_time
                    self._shutdown_in_progress = True
                else:  # if timer not exceeded
                    # delta t how long the temp. has been exceeded. after the
                    # if-else check, since +timestep is at the end of the
                    # step, thus relevant for the next step.
                    self._dt_time_temp_exc += timestep
            else:  # else if temp. not exceeded, reset timer
                self._dt_time_temp_exc = 0.0

            # save chp state:
            self._chp_state[self.stepnum[0]] = (
                self._bin_pow_fac * self._power_modulation[0]
            )

            # process startup and shutdown procedure
            # is the CHP switched on? If yes, startup time is larger than
            # shutdown time.
            if self._startup_at_time > self._shutdown_at_time:
                # if chp shutdown was quite recent, thus heat is remaining
                # -> shorten startup procedure
                if self._shutdown_factor_th > self._chp_off_perc:
                    # if shutdown was recent, take the shutdown factor and
                    # look where in startup it can be found. then add this
                    # timestep where it was found to the startup time
                    # (=increase startup duration) to account for remaining
                    # heat in the system
                    self._remaining_heat = np.argmin(
                        np.abs(self._startuptsteps - self._shutdown_factor_th)
                    )
                    # and reset shutdown factor to zero and set shutdown in
                    # progress False to avoid doing this twice:
                    self._shutdown_factor_th = 0.0
                    self._shutdown_in_progress = False
                # get startup duration:
                self._startup_duration = (  # on since
                    curr_time - self._startup_at_time + self._remaining_heat
                )
                # do factor calculations only, if startup not yet finished,
                # else do nothing, since factors are already set to 1
                if self._startup_in_progress:
                    # power multiplication factors:
                    self._startup_factor_th = _pf.chp_startup_th(
                        self._startup_duration
                    )
                    self._startup_factor_el = _pf.chp_startup_el(
                        self._startup_duration
                    )
                    self._startup_factor_gas = _pf.chp_startup_gas(
                        self._startup_duration
                    )
                    # limit values to 0<=x<=1
                    self._startup_factor_th = (
                        0.0
                        if self._startup_factor_th < 0.0
                        else 1.0
                        if self._startup_factor_th > 1.0
                        else self._startup_factor_th
                    )
                    self._startup_factor_el = (
                        0.0
                        if self._startup_factor_el < 0.0
                        else 1.0
                        if self._startup_factor_el > 1.0
                        else self._startup_factor_el
                    )
                    self._startup_factor_gas = (
                        0.0
                        if self._startup_factor_gas < 0.0
                        else 1.0
                        if self._startup_factor_gas > 1.0
                        else self._startup_factor_gas
                    )
                    # check if thermal startup is completed, else go on
                    if self._startup_factor_th > self._chp_on_perc:
                        # if thermal startup is completed, set all startups as
                        # completed
                        self._startup_in_progress = False
                        self._startup_factor_th = 1.0
                        self._startup_factor_el = 1.0
                        self._startup_factor_gas = 1.0
                        self._remaining_heat = 0.0
            #            elif not emergeny_shutdown:  # if shutdown was more recent, but not
            else:  # if shutdown was more recent
                self._shutdown_duration = (  # off since
                    curr_time - self._shutdown_at_time
                )
                if self._shutdown_in_progress:
                    self._shutdown_factor_th = _pf.chp_shutdown_th(
                        self._shutdown_duration
                    )
                    if self._shutdown_factor_th < self._chp_off_perc:
                        # shutdown finished. reset values
                        self._shutdown_in_progress = False
                        self._shutdown_factor_th = 0.0

        if not self._shutdown_in_progress:
            self._dQ_heating[:] = (
                self._bin_pow_fac  # switch off if mod below 50%
                * _pf.chp_thermal_power(self._power_modulation[0])
                * self.chp_power_el
                * self._startup_factor_th
                * self._p2h_ratio
            )
        else:  # if shutdown is in progress
            self._dQ_heating[:] = self._last_dQ * self._shutdown_factor_th
        self._Pel[:] = (
            self._power_modulation[0]
            * self.chp_power_el
            * self._startup_factor_el
            * self._bin_pow_fac
        )
        self._Pgas[:] = (
            self._startup_factor_gas
            * self._bin_pow_fac
            * self._eta_el_fac
            * _pf.chp_gas_power(self._power_modulation[0])
            * self.chp_power_el
        )

        # save rate of heat flow etc to result array
        if self._process_flows[0]:
            self.res_Pel[self.stepnum[0]] = self._Pel[0]
            self.res_dQ[self.stepnum[0]] = self._dQ_heating[0]
            self.res_Pgas[self.stepnum[0]] = self._Pgas[0]

        # if flue gas hex is connected directly, save flue gas flow to it
        # directly. Flue gas flow in Nm3/s is calculated from the ratio of the
        # current Pgas power to max Pgas power with gas flow at Pgas max.
        if self._connect_fg_hex:
            self._fg_hex_gdv[:] = (
                self._fg_dv_at_pmax * self._Pgas[0] / self._Pgas_max
            )

        # save current dQ as last dQ for the next step
        if not self._shutdown_in_progress:  # only if shutdown not already
            # in progress
            self._last_dQ[:] = self._dQ_heating

        self._process_flows[0] = _pf._process_flow_invar(
            process_flows=self._process_flows,
            dm_io=self._dm_io,
            dm_top=self._dm_top,
            dm_bot=self._dm_bot,
            dm_port=self._dm_port,
            stepnum=self.stepnum,
            res_dm=self.res_dm,
        )

        _pf.water_mat_props_ext_view(
            T_ext=self._T_ext,
            cp_T=self._cp_T,
            lam_T=self._lam_T,
            rho_T=self._rho_T,
            ny_T=self._ny_T,
        )

        # get mean lambda value between cells:
        _pf._lambda_mean_view(lam_T=self._lam_T, out=self._lam_mean)

        _pf.UA_plate_tb(
            A_cell=self.A_cell,
            grid_spacing=self.grid_spacing,
            lam_mean=self._lam_mean,
            UA_tb_wll=self._UA_tb_wll,
            out=self._UA_tb,
        )

        # for conduction between current cell and ambient:
        # get outer pipe (insulation) surface temperature using a linearized
        # approach assuming steady state (assuming surface temperature = const.
        # for t -> infinity) and for cylinder shell (lids are omitted)
        _pf.surface_temp_steady_state_inplace(
            T=self._T_ext[1:-1],
            T_inf=self._T_amb[0],
            A_s=self._A_shell_ins,
            alpha_inf=self._alpha_inf,
            UA=self._UA_amb_shell,
            T_s=self._T_s,
        )
        # get inner alpha value between fluid and wall from nusselt equations:
        _pf.pipe_alpha_i(
            self._dm_io,
            self._T_ext[1:-1],
            self._rho_T,
            self._ny_T,
            self._lam_T,
            self.A_cell,
            self._d_i,
            self._cell_dist,
            self._alpha_i,
        )
        # get outer alpha value between insulation and surrounding air:
        _pf.cylinder_alpha_inf(  # for a cylinder
            T_s=self._T_s,
            T_inf=self._T_amb[0],
            flow_length=self._flow_length,
            vertical=self._vertical,
            r_total=self._r_total,
            alpha_inf=self._alpha_inf,
        )
        # get resulting UA to ambient:
        _pf.UA_fld_wll_ins_amb_cyl(
            A_i=self._A_shell_i,
            r_ln_wll=self._r_ln_wll,
            r_ln_ins=self._r_ln_ins,
            r_rins=self._r_rins,
            alpha_i=self._alpha_i,
            alpha_inf=self._alpha_inf,
            lam_wll=self._lam_wll,
            lam_ins=self._lam_ins,
            out=self._UA_amb_shell,
        )

        # precalculate values which are needed multiple times:
        _pf.cell_temp_props_ext(
            T_ext=self._T_ext,
            V_cell=self.V_cell,
            cp_T=self._cp_T,
            rho_T=self._rho_T,
            mcp_wll=self._mcp_wll,
            rhocp=self._rhocp,
            mcp=self._mcp,
            ui=self._ui,
        )

        dT_cond_port = _pf._process_ports_collapsed(
            ports_all=self._models.ports_all,
            port_link_idx=self._port_link_idx,
            port_own_idx=self._port_own_idx,
            T=self._T_ext[1:-1],
            mcp=self._mcp,
            UA_port=self._UA_port,
            UA_port_wll=self._UA_port_wll,
            A_p_fld_mean=self._A_p_fld_mean,
            port_gsp=self._port_gsp,
            grid_spacing=self.grid_spacing,
            lam_T=self._lam_T,
            cp_port=self._cp_port,
            lam_port_fld=self._lam_port_fld,
            T_port=self._T_port,
        )

        (
            self._models._step_stable,
            self._models._vN_max_step,
            self._models._max_factor,
        ) = _pf._vonNeumann_stability_invar(
            part_id=self.part_id,
            stability_breaches=self._stability_breaches,
            UA_tb=self._UA_tb,
            UA_port=self._UA_port,
            UA_amb_shell=self._UA_amb_shell,
            dm_io=self._dm_io,
            rho_T=self._rho_T,
            rhocp=self._rhocp,
            grid_spacing=self.grid_spacing,
            port_subs_gsp=self._port_subs_gsp,
            A_cell=self.A_cell,
            A_port=self._A_p_fld_mean,
            A_shell=self._A_shell_ins,
            r_total=self._r_total,
            V_cell=self.V_cell,
            step_stable=self._models._step_stable,
            vN_max_step=self._models._vN_max_step,
            max_factor=self._models._max_factor,
            stepnum=self.stepnum,
            timestep=timestep,
        )

        # CALCULATE DIFFERENTIALS
        # calculate heat transfer by internal heat sources
        self._dT_heated[:] = (
            self._dQ_heating * self._heat_mult / self._mcp_heated
        )

        # calculate heat transfer by conduction
        self.dT_cond[:] = (
            +self._UA_tb[:-1] * (self._T_ext[:-2] - self._T_ext[1:-1])
            + self._UA_tb[1:] * (self._T_ext[2:] - self._T_ext[1:-1])
            + self._UA_amb_shell * (self._T_amb[0] - self._T_ext[1:-1])
        ) / self._mcp
        # calculate heat transfer by advection
        self.dT_adv[:] = (
            +self._dm_top * (self._cp_T[:-2] * self._T_ext[:-2] - self._ui)
            + self._dm_bot * (self._cp_T[2:] * self._T_ext[2:] - self._ui)
        ) / self._mcp

        # sum up heat conduction and advection for port values:
        for i in range(self._port_own_idx.size):
            idx = self._port_own_idx[
                i
            ]  # idx of port values at temperature/diff array
            # conduction
            self.dT_cond[idx] += dT_cond_port[i]
            # advection
            self.dT_adv[idx] += (
                self._dm_port[i]
                * (self._cp_port[i] * self._T_port[i] - self._ui[idx])
                / self._mcp[idx]
            )

        self.dT_total = self.dT_cond + self.dT_adv + self.dT_heat

        # (self.dT_total, self._process_flows, self._models._step_stable,
        #  self._models._vN_max_step, self._models._max_factor) = chp_core_diff(
        #     T_ext=self._T_ext, T_port=self._T_port, T_s=self._T_s,
        #     T_amb=self._T_amb, ports_all=self._models.ports_all,  # temperatures
        #     dm_io=self._dm_io, dm_top=self._dm_top, dm_bot=self._dm_bot,
        #     dm_port=self._dm_port, dQ_heating=self._dQ_heating,
        #     res_dm=self.res_dm, res_dQ=self.res_dQ,  # flows
        #     cp_T=self._cp_T, lam_T=self._lam_T, rho_T=self._rho_T,
        #     ny_T=self._ny_T, lam_mean=self._lam_mean, cp_port=self._cp_port,
        #     lam_port_fld=self._lam_port_fld, mcp=self._mcp,
        #     mcp_heated=self._mcp_heated, rhocp=self._rhocp,
        #     lam_wll=self._lam_wll, lam_ins=self._lam_ins,
        #     mcp_wll=self._mcp_wll, ui=self._ui,  # material properties
        #     alpha_i=self._alpha_i, alpha_inf=self._alpha_inf,  # alpha values
        #     UA_tb=self._UA_tb, UA_tb_wll=self._UA_tb_wll,
        #     UA_amb_shell=self._UA_amb_shell, UA_port=self._UA_port,
        #     UA_port_wll=self._UA_port_wll,  # UA values
        #     port_own_idx=self._port_own_idx, port_link_idx=self._port_link_idx,  # indices
        #     heat_mult=self._heat_mult,
        #     grid_spacing=self.grid_spacing, port_gsp=self._port_gsp,
        #     port_subs_gsp=self._port_subs_gsp, d_i=self._d_i,
        #     cell_dist=self._cell_dist, flow_length=self._flow_length,  # lengths
        #     r_total=self._r_total, r_ln_wll=self._r_ln_wll,
        #     r_ln_ins=self._r_ln_ins, r_rins=self._r_rins,  # lengths
        #     A_cell=self.A_cell, V_cell=self.V_cell, A_shell_i=self._A_shell_i,
        #     A_shell_ins=self._A_shell_ins, A_p_fld_mean=self._A_p_fld_mean,  # areas and vols
        #     process_flows=self._process_flows, vertical=self._vertical,
        #     step_stable=self._models._step_stable,  # bools
        #     part_id=self.part_id, stability_breaches=self._stability_breaches,
        #     vN_max_step=self._models._vN_max_step,
        #     max_factor=self._models._max_factor,  # misc
        #     stepnum=self.stepnum, timestep=timestep,  # step information
        #     dT_cond=self.dT_cond, dT_adv=self.dT_adv,
        #     dT_heat=self.dT_heat, dT_heated=self._dT_heated  # differentials
        #     )

        return self.dT_total
