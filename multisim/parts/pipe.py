# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: May 2018
"""

import numpy as np

from ..simenv import SimEnv
from .. import precomp_funs as _pf


class Pipe(SimEnv):
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
        self._models = master_cls

        if 'constr_type' not in kwargs:  # part is not subclassed
            self.constr_type = 'Pipe'  # define construction type
        else:  # part is subclassed. get subclass construction type
            self.constr_type = kwargs['constr_type']

        # define arguments and errors:
        self._aae = {  # arguments and errors
            'length': 'Pipe length in [m]. Type: int, float. Range: X > 0',
            'grid_points': 'Number of grid points. Type: int. Range: X > 0',
        }
        # check for arguments:
        self._print_arg_errs(self.constr_type, name, self._aae, kwargs)

        base_err = (  # define leading base error message
            'While adding {0} `{1}` to the simulation '
            'environment, the following error occurred:\n'
        ).format(self.constr_type, str(name))
        arg_err = (  # define leading error for missing/incorrect argument
            'Missing argument or incorrect type/value: {0}\n\n'
        )
        self._base_err = base_err  # save to self to access it in methods
        self._arg_err = arg_err  # save to self to access it in methods

        # assert that all required arguments have been passed:
        # length
        # =============================================================================
        #         err_str = (
        #             base_err + arg_err.format('length') +
        #             'The pipe length has to be given in [m] with `length=X`, where X '
        #             'is an integer or float value >0.')
        #         assert ('length' in kwargs and
        #                 isinstance(kwargs['length'], (int, float)) and
        #                 kwargs['length'] > 0), err_str
        # =============================================================================
        assert (
            isinstance(kwargs['length'], (int, float)) and kwargs['length'] > 0
        ), (
            self._base_err
            + self._arg_err.format('length')
            + self._aae['length']
        )
        self.length = float(kwargs['length'])
        # grid points
        # =============================================================================
        #         err_str = (
        #             base_err + arg_err.format('grid_points') +
        #             'The number of grid points has to be given with `grid_points=X`, '
        #             'where X is an integer value >0.')
        #         assert ('grid_points' in kwargs and type(kwargs['grid_points']) == int
        #                 and kwargs['grid_points'] > 0), err_str
        # =============================================================================
        assert (
            isinstance(kwargs['grid_points'], int)
            and kwargs['grid_points'] > 0
        ), (
            self._base_err
            + self._arg_err.format('grid_points')
            + self._aae['grid_points']
        )
        self.num_gp = kwargs['grid_points']

        # %% start part construction:
        self.name = name
        self.part_id = self._models.num_parts - 1
        # save smallest possible float number for avoiding 0-division:
        self._tiny = self._models._tiny

        # get grid spacing. other geometry definition will be done after
        # getting ports and pipe specs.
        self.grid_spacing = self.length / self.num_gp
        # set array for each cell's distance from the start of the pipe:
        self._cell_dist = np.arange(
            self.grid_spacing / 2, self.length, self.grid_spacing
        )

        # %% preallocate temperature and property grids:
        # use an extended temeprature array with +1 cell at each end to be able
        # to use memoryviews for all top/bot assigning. Thus DO NOT ALTER T_ext
        self._T_ext = np.zeros(self.num_gp + 2, dtype=np.float64)
        self.T = self._T_ext[1:-1]  # memview of T_ext
        Tshp = self.T.shape  # save shape since it is needed often
        # preallocate views to temperature grid for upper and lower cells
        self._T_top = self._T_ext[:-2]  # memview of T_ext
        self._T_bot = self._T_ext[2:]  # memview of T_ext
        # preallocate temperature grid for ports:
        self._T_port = np.zeros(2)
        Tpshp = self._T_port.shape  # save shape since it is needed often
        # preallocate temperature grid for outer surface temperature:
        self._T_s = np.zeros(Tshp)
        # preallocate lambda grids and alpha grid for heat conduction with the
        # smallest possible value to avoid zero division:
        self._lam_T = np.zeros(self.num_gp)
        self._lam_mean = np.zeros(self.num_gp - 1)  # mean value between cells
        self._alpha_i = np.full_like(self._lam_T, 3.122)
        self._alpha_inf = np.full_like(self._lam_T, 3.122)
        # preallocate fluid heat capacity grids:
        self._cp_T = np.zeros(self.num_gp + 2)  # extended array
        self._cp_top = self._cp_T[:-2]  # view for top and bot grid
        self._cp_bot = self._cp_T[2:]
        self._cp_port = np.zeros(Tpshp)
        # preallocate fluid density grids:
        self._rho_T = np.zeros(self.num_gp)
        self._rhocp = np.zeros_like(self._rho_T)  # volume specific heat cap.
        # preallocate kinematic viscosity grid:
        self._ny_T = np.zeros(self.num_gp)
        # cell mass specific inner energy grid:
        self._ui = np.zeros(Tshp)
        # heat capacity of fluid AND wall:
        self._mcp = np.zeros(Tshp)
        # preallocate U*A grids:
        self._UA_tb = np.zeros(self.T.shape[0] + 1)
        self._UA_port = np.zeros(Tpshp)
        self._UA_amb_shell = np.zeros(Tshp)
        # preallocate mass flow grids:
        self.dm = np.zeros(Tshp)  # general massflow array
        #        self._dm_cell = np.zeros(Tshp)  # array for cell center flow
        self._dm_top = np.zeros(Tshp)  # array for pos. flow from top
        self._dm_bot = np.zeros(Tshp)  # array for pos. flow from bottom
        self._dm_port = np.zeros(Tpshp)  # array for inflow through prts
        #        self._dm_port = np.zeros(2)  # array for inflow through prts
        self._dm_io = np.zeros(Tpshp)  # array for I/O flow
        # this I/O flow array MUST NOT BE CHANGED by anything else than
        # _update_FlowNet() method!

        # preallocate result grid with one row. An estimate of total rows will
        # be preallocated before simulation start in initialize_sim. massflow
        # grid is preallocated in set_initial_cond:
        self.res = np.zeros((1, self.num_gp))
        self.res_dm = np.zeros((2, 1))
        # separate result grids for conduction and advection to make the code
        # less complicated:
        self.dT_cond = np.zeros(Tshp)
        self.dT_adv = np.zeros(Tshp)
        self.dT_total = np.zeros(Tshp)

        # port definition (first and last array element):
        self.port_num = 2
        # Index to own value array to get values of own ports, meaning if I
        # index a FLATTENED self.T.flat with self._port_own_idx, I need to
        # get values accoring to the order given in self.port_names.
        # That is, this must yield the value of the cell in self.T, which is
        # belonging to the port 'in':
        # self.T.flat[self._port_own_idx[self.port_names.index('in')]]
        self._port_own_idx = np.array((0, self.T.shape[0] - 1), dtype=np.int32)
        self._port_own_idx_2D = self._port_own_idx  # save for compatibility
        self.port_ids = np.array((), dtype=np.int32)
        # define port_names
        self.port_names = tuple(('in', 'out'))
        # set massflow characteristics for ports: in means that an inflowing
        # massflow has a positive sign, out means that an outflowing massflow
        # is pos.
        self.dm_char = tuple(('in', 'out'))
        # construct partname+portname to get fast access to own ports:
        dummy_var = list(self.port_names)
        for i in range(self.port_num):
            dummy_var[i] = self.name + ';' + dummy_var[i]
        self._own_ports = tuple(dummy_var)
        # preallocate port values to avoid allocating in loop:
        self._port_vals = np.zeros(self.port_num)
        # preallocate grids for port connection parameters
        # cross section area of wall of connected pipe, fluid cross section
        # area of, gridspacing and lambda of wall of connected pipe
        self._A_wll_conn_p = np.zeros(Tpshp)
        self._A_fld_conn_p = np.zeros(Tpshp)
        self._A_p_wll_mean = np.zeros(Tpshp)
        self._A_p_fld_mean = np.zeros(Tpshp)
        self._port_gsp = np.zeros(Tpshp)
        self._lam_wll_conn_p = np.full_like(self._T_port, 1e-2)
        self._lam_port_fld = np.zeros(Tpshp)
        self._lam_fld_own_p = np.zeros(Tpshp)
        self._UA_port_fld = np.zeros(Tpshp)
        self._UA_port_wll = np.zeros(Tpshp)

        # preallocate list to mark ports which have already been solved in
        # topology (to enable creating subnets)
        self._solved_ports = list()
        # preallocate grids for von Neumann stability checking:
        self._vN_diff = np.zeros(4)
        #        self._vN_dm = np.zeros(1)  # this is a scalar for dm_invariant=True
        self._trnc_err = 0.0
        # add truncation error cell weight to weight or disable the trunc. err.
        # calculation in adaptive steps for the part (weight has to be a single
        # float or integer) or for specific cells (weight has to be an array
        # of the shape of self.T):
        self._trnc_err_cell_weight = 1.0
        # counter for how many times THIS part has breached von Neumann stab.
        self._stability_breaches = np.zeros(1).astype(int)  # array for view
        # preallocate grids for precalculated stuff: REPLACED WITH LOCAL VARS!
        #        self.__rhocp = np.zeros(Tshp)
        #        self.__cpT = np.zeros(Tshp)

        # set if type has to be solved numeric:
        self.solve_numeric = True
        # if part can be solved explicitly (otherwise a time-costly implicit
        # solving will be used):
        self.solve_explicitly = True
        # if port arrays shall be collapsed to amount of ports to improve speed
        self.collapse_arrays = True
        self._collapsed = True  # bool checker if already collapsed
        # if ports can be added to this part:
        self.can_add_ports = True
        self._use_structuredarray = True
        # determine if part is treated as hydraulic compensator
        self.hydr_comp = False
        # if part can be a parent part of a primary flow net:
        self._flow_net_parent = False
        # add each flow channel of part to hydr_comps (will be removed once its
        # massflow solving method is completely integrated in flow_net.
        # remaining parts except real hydr comps will be used to generate an
        # error):
        self._models._hydr_comps.add(self.name)
        # if the topology construction method has to stop when it reaches the
        # part to solve more ports from other sides before completely solving
        # the massflow of it. This will always be True for hydr comps to
        # postpone solving the last port by making expensive cumulative sums.
        self.break_topology = False
        # count how many ports are still open to be solved by topology
        self._cnt_open_prts = self.port_num  # not required here
        self._port_heatcond = True  # if heatcond. over ports is enabled
        # determine if part has the capability to affect massflow (dm) by
        # diverting flow through ports or adding flow through ports:
        self.affect_dm = False
        # if the massflow (dm) has the same value in all cells of the part
        # (respectively in each flow channel for parts with multiple flows):
        self.dm_invariant = True
        # if the part has multiple separated flow channels which do NOT mix
        # (like a heat exchanger for exampe):
        self.multiple_flows = False
        # bool checker if flows were updated in update_flownet to avoid
        # processing flows in get_diff each time (array for referencing):
        self._process_flows = np.array([True])
        # if the part CAN BE controlled by the control algorithm:
        self.is_actuator = False
        # if the part HAS TO BE controlled by the control algorithm:
        self.control_req = False
        # if the part's get_diff method is solved with memory views entirely
        # and thus has arrays which are extended by +2 (+1 at each end):
        self.enlarged_memview = False
        # if the part has a special plot method which is defined within the
        # part's class:
        self.plot_special = False

        # save initialization status:
        self.initialized = False

        # adjust arrays depending on bools set:
        if self.dm_invariant:
            # if all cells have the same massflow, reduce massflow arrays
            #            self.dm = np.zeros(1)
            self._dm_io = np.zeros(1)
            self.dm = self._dm_io[:]

        # save memory address of T
        self._memadd_T = self.T.__array_interface__['data'][0]

        # save all kind of info stuff to dicts:
        # topology info:
        if not hasattr(self, 'info_topology'):
            self.info_topology = dict()

        # IMPORTANT: THIS VARIABLE **MUST NOT BE INHERITED BY SUB-CLASSES**!!
        # If sub-classes are inherited from this part, this bool checker AND
        # the following variables MUST BE OVERWRITTEN!
        # ist the diff function fully njitted AND are all input-variables
        # stored in a container?
        self._diff_fully_njit = True
        self._diff_njit = _pf.pipe1D_diff  # handle to njitted diff function
        # input args are created in simenv _create_diff_inputs method

    def init_part(self, **kwds):
        """Initialize the pipe part. Only to be called internally."""
        # define arguments and errors:
        self._aaei = {  # arguments and errors for init
            'insulation_thickness OR s_ins': (
                'Thermal insulation thickness in [m]. Type: int, float. '
                'Range: X >= 0'
            ),
            'insulation_lambda OR lambda_ins': (
                'Thermal insulation heat conductivity (lambda) '
                'in [[W/(m*K)]]. Type: int, float. Range: X >= 0'
            ),
            'T_init': (
                'Starting temperature distribution in [°C]. If scalar, all '
                'cells will have the same value. If array, array shape must '
                'correlate with the number of grid points. '
                'Type: int, float, np.ndarray. Range: 0 <= X <= 110'
            ),
            'T_amb': (
                'Ambient temperature in [°C]. Type: int, float. '
                'Range: -20 <= X <= 150'
            ),
            'material': (
                'Casing and port material. Pass any value to get more '
                'information on supported materials. Type: str.'
            ),
            'pipe_specs': (
                'Size specifications of pipe and ports. Pass any value to get '
                'more information on supported values. Type: dict.'
            ),
        }
        # check for arguments:
        self._print_arg_errs(self.constr_type, self.name, self._aaei, kwds)

        # check for insulation:
        # =============================================================================
        #         err_str = (
        #             self._base_err +
        #             self._arg_err.format('insulation_thickness OR s_ins') +
        #             'The insulation thickness has to be given in [m] with '
        #             '`insulation_thickness=X` OR `s_ins=X`, where X is an integer or '
        #             'float value >=0.')
        #         assert 'insulation_thickness' in kwds or 's_ins' in kwds, err_str
        # =============================================================================
        self._s_ins = (
            kwds['s_ins'] if 's_ins' in kwds else kwds['insulation_thickness']
        )
        assert isinstance(self._s_ins, (int, float)) and self._s_ins >= 0, (
            self._base_err
            + self._arg_err.format('s_ins')
            + self._aaei['insulation_thickness OR s_ins']
        )
        self._s_ins = float(self._s_ins)  # make sure it is float!

        # =============================================================================
        #         err_str = (
        #             self._base_err +
        #             self._arg_err.format('insulation_lambda OR lambda_ins') +
        #             'The heat conductivity (lambda) value of the insulation has '
        #             'to be given in [W/(m*K)] with `insulation_lambda=X` OR '
        #             '`lambda_ins=X`, where X is an integer or float value >= 0.')
        #         assert 'insulation_lambda' in kwds or 'lambda_ins' in kwds, err_str
        # =============================================================================
        self._lam_ins = (
            kwds['lambda_ins']
            if 'lambda_ins' in kwds
            else kwds['insulation_lambda']
        )
        assert (
            isinstance(self._lam_ins, (int, float)) and self._lam_ins >= 0
        ), self._aaei['insulation_lambda OR lambda_ins']
        self._lam_ins = float(self._lam_ins)  # make sure it is float

        # assert material and pipe specs (after defining port names to avoid
        # name getting conflicts):
        # =============================================================================
        #         err_str = (
        #             self._base_err +
        #             self._arg_err.format('pipe_specs, material') +
        #             '`pipe_specs` and `material` have to be given.')
        #         assert 'material' in kwds and 'pipe_specs' in kwds, err_str
        # =============================================================================
        # get material pipe specs for diameter etc.:
        self._get_specs_n_props(**kwds)
        # assert that pipe specs were given for all ports together:
        err_str = (
            self._base_err
            + self._arg_err.format('pipe_specs')
            + 'For a part of type pipe, the same `pipe_specs` must be given for '
            'all ports by specifying \'all\' as first-level-key!'
        )
        assert 'all' in kwds['pipe_specs'], err_str

        # assert and get initial temperature:
        # =============================================================================
        #         err_str = (
        #             self._base_err +
        #             self._arg_err.format('T_init') +
        #             'The initial temperature `T_init=X` in [°C] has to be given, '
        #             'where X is a single float or integer value or as an array with '
        #             'shape (' + str(self.num_gp) + ',).')
        #         assert 'T_init' in kwds, err_str
        # =============================================================================
        self._T_init = kwds['T_init']
        assert isinstance(self._T_init, (float, int, np.ndarray)), (
            self._base_err
            + self._arg_err.format('T_init')
            + self._aaei['T_init']
        )
        # set init values to T array:
        if isinstance(self._T_init, (int, float)):
            self.T[:] = float(self._T_init)
            self._T_init = float(self._T_init)
        else:
            # assert that correct shape is given and if yes, save to T
            assert self._T_init.shape == self.T.shape, (
                self._base_err
                + self._arg_err.format('T_init')
                + '`T_init` shape not matching part shape resp. number of '
                'grid points'
            )
            self.T[:] = self._T_init

        # geometry/grid definition
        self._d_i = float(self.info_topology['all_ports']['pipe_specs']['d_i'])
        self._d_o = float(self.info_topology['all_ports']['pipe_specs']['d_o'])
        self._r_i = self._d_i / 2
        self._r_o = self._d_o / 2
        # total RADIUS from center of the pipe to the outer radius of the
        # insulation:
        self._r_total = self._d_o / 2 + self._s_ins
        # precalculated log parameters for heat conduction calculation:
        # factor for wall lambda value referred to r_i
        self._r_ln_wll = self._r_i * np.log(self._r_o / self._r_i)
        # factor for insulation lambda value referred to r_i
        self._r_ln_ins = self._r_i * np.log(self._r_total / self._r_o)
        # factor for outer alpha value referred to r_i
        self._r_rins = self._r_i / self._r_total
        # thickness of the wall:
        self._s_wll = self._r_o - self._r_i
        # cross section area and volume of cell:
        self.A_cell = self._A_fld_own_p
        self.V_cell = self.A_cell * self.grid_spacing
        # surface area of pipe wall per cell (fluid-wall-contact-area):
        self._A_shell_i = np.pi * self._d_i * self.grid_spacing
        # outer surface area of pipe wall:
        self._A_shell_o = np.pi * self._d_o * self.grid_spacing
        # outer surface area of pipe including insulation:
        self._A_shell_ins = np.pi * self._r_total * 2 * self.grid_spacing
        # cross section area of shell:
        self._A_shell_cs = np.pi / 4 * (self._d_o ** 2 - self._d_i ** 2)
        # shell volume (except top and bottom cover):
        self._V_shell_cs = self._A_shell_cs * self.length

        # get remaining constant UA value for top-bottom wall heat conduciton:
        self._UA_tb_wll = self._A_shell_cs / self.grid_spacing * self._lam_wll

        # calculate m*cp for the wall PER CELL with the material information:
        self._mcp_wll = (
            self._cp_wll * self._rho_wll * self._V_shell_cs / self.num_gp
        )

        # get ambient temperature:
        self._chk_amb_temp(**kwds)

        # save shape parameters for the calculation of heat losses.
        self._vertical = False
        # get flow length for a horizontal cylinder for Nusselt calculation:
        self._flow_length = _pf.calc_flow_length(
            part_shape='cylinder',
            vertical=self._vertical,
            diameter=self._r_total * 2,
        )

        # construct list of differential input argument names IN THE CORRECT
        # ORDER!!!
        # regex to remove strings: [a-zA-Z_]*[ ]*=self.
        self._input_arg_names_sorted = [
            '_T_ext',
            '_T_port',
            '_T_s',
            '_T_amb',
            'ports_all',  # temperatures
            '_dm_io',
            '_dm_top',
            '_dm_bot',
            '_dm_port',
            'res_dm',  # flows
            '_cp_T',
            '_lam_T',
            '_rho_T',
            '_ny_T',
            '_lam_mean',
            '_cp_port',
            '_lam_port_fld',  # mat. props.
            '_mcp',
            '_rhocp',
            '_lam_wll',
            '_lam_ins',
            '_mcp_wll',
            '_ui',  # material properties.
            '_alpha_i',
            '_alpha_inf',  # alpha values
            '_UA_tb',
            '_UA_tb_wll',
            '_UA_amb_shell',
            '_UA_port',
            '_UA_port_wll',  # UA values
            '_port_own_idx',
            '_port_link_idx',  # indices
            'grid_spacing',
            '_port_gsp',
            '_port_subs_gsp',
            '_d_i',
            '_cell_dist',  # lengths
            '_flow_length',
            '_r_total',
            '_r_ln_wll',
            '_r_ln_ins',
            '_r_rins',  # lengths
            'A_cell',
            'V_cell',
            '_A_shell_i',
            '_A_shell_ins',
            '_A_p_fld_mean',  # areas and vols
            '_process_flows',
            '_vertical',
            '_step_stable',  # bools
            'part_id',
            '_stability_breaches',
            '_vN_max_step',
            '_max_factor',  # misc.
            'stepnum',  # step information
            'dT_cond',
            'dT_adv',
            'dT_total',  # differentials
        ]

        #       # FAST CALLS WITH REDUCED NUMBER OF ARGUMENTS:
        #       # (speed gain negligible, but reducing traceability of errors)
        #        self._input_arg_names_fast = [
        #            '_T_ext', '_T_amb', 'ports_all',  # temperatures
        #            '_dm_io', '_dm_top', '_dm_bot', '_dm_port', 'res_dm',  # flows
        #            '_lam_wll', '_lam_ins', '_mcp_wll',  # material properties.
        #            '_alpha_inf',  # alpha values
        #            '_UA_tb_wll', '_UA_amb_shell',
        #            '_UA_port_wll',  # UA values
        #            '_port_own_idx', '_port_link_idx',  # indices
        #            'grid_spacing', '_port_gsp', '_port_subs_gsp',
        #            '_d_i', '_cell_dist',  # lengths
        #            '_flow_length', '_r_total', '_r_ln_wll', '_r_ln_ins',
        #            '_r_rins',  # lengths
        #            'A_cell', 'V_cell', '_A_shell_i',
        #            '_A_shell_ins', '_A_p_fld_mean',  # areas and vols
        #            '_process_flows', '_vertical', '_step_stable',  # bools
        #            'part_id', '_stability_breaches', '_vN_max_step',
        #            '_max_factor',  # misc.
        #            'stepnum',  # step information
        #            'dT_cond', 'dT_adv', 'dT_total',  # differentials
        #            ]

        # input args when using closures (HUGE compile time, but faster exec)
        # for calls
        #        self._input_arg_names_calls = [
        #            '_T_ext', '_T_port', '_T_s', '_T_amb', 'ports_all',  # temperatures
        #            '_dm_io', '_dm_top', '_dm_bot', '_dm_port', 'res_dm',  # flows
        #            '_cp_T', '_lam_T', '_rho_T', '_ny_T', '_lam_mean', '_cp_port', '_lam_port_fld',  # mat. props.
        #            '_mcp', '_rhocp', '_ui',  # material properties.
        #            '_alpha_i', '_alpha_inf',  # alpha values
        #            '_UA_tb', '_UA_amb_shell', '_UA_port', '_UA_port_wll',  # UA values
        #            '_process_flows', '_step_stable',  # bools
        #            '_stability_breaches', '_vN_max_step', '_max_factor',  # misc.
        #            'stepnum',  # step information
        #            'dT_cond', 'dT_adv', 'dT_total',  # differentials
        #            ]
        #       # for constructing the closure
        #        self._input_arg_names_closure = [
        #            '_lam_wll',
        #            '_lam_ins', '_mcp_wll',
        #            '_UA_tb_wll',
        #            '_port_own_idx', '_port_link_idx',  # indices
        #            'grid_spacing', '_port_gsp',
        #            '_port_subs_gsp', '_d_i',
        #            '_cell_dist', '_flow_length',
        #            '_r_total',
        #            '_r_ln_wll', '_r_ln_ins',
        #            '_r_rins', '_s_wll', '_s_ins',  # lengths
        #            'A_cell', 'V_cell', '_A_shell_i',
        #            '_A_shell_ins', '_A_p_fld_mean',  # areas and vols
        #            '_vertical', 'part_id']
        #
        #       # input args for using homogeneous tuples/list (complicated, since
        #       # there is no numba support for unpacking these INSIDE the jitted funs)
        #        self._iaf1d = [
        #            '_T_ext', '_T_port', '_T_s', '_T_amb', 'ports_all',  # temperatures
        #            '_dm_io', '_dm_top', '_dm_bot', '_dm_port',
        #            '_cp_T', '_lam_T', '_rho_T', '_ny_T', '_lam_mean', '_cp_port',
        #            '_lam_port_fld',  # mat. props.
        #            '_mcp', '_rhocp',
        #            '_ui',  # material properties.
        #            '_alpha_i', '_alpha_inf',  # alpha values
        #            '_UA_tb', '_UA_amb_shell', '_UA_port',
        #            '_UA_port_wll',
        #            '_vN_max_step', '_max_factor',
        #            'dT_cond', 'dT_adv', 'dT_total',
        #            '_port_gsp', '_port_subs_gsp', '_cell_dist', '_A_p_fld_mean'
        #            ]  # float 1d
        #        self._iaf2d = ['res_dm']  # float 2d
        #        self._iai1d = [
        #            '_port_own_idx', '_port_link_idx',
        #            '_stability_breaches', 'stepnum']   # int32 1d
        #        self._iab1d = ['_process_flows',
        #                       '_step_stable']  # bool array
        #        self._iaf = [
        #            '_lam_wll', '_lam_ins', '_mcp_wll', '_UA_tb_wll',
        #            'grid_spacing', '_d_i',
        #            '_flow_length', '_r_total', '_r_ln_wll', '_r_ln_ins', '_r_rins',
        #            'A_cell', 'V_cell', '_A_shell_i', '_A_shell_ins',
        #            ]  # floats

        #        # STRUCTURED ARRAYS ONLY FOR FLOATS, THE REST AS LIST/TUPLE:
        #        self._input_arg_names_recarr = [
        #            '_lam_wll', '_lam_ins',
        #            '_mcp_wll', '_UA_tb_wll',
        #            'grid_spacing', '_d_i', '_flow_length',
        #            '_r_total', '_r_ln_wll', '_r_ln_ins',
        #            '_r_rins',  # lengths
        #            'A_cell', 'V_cell', '_A_shell_i',
        #            '_A_shell_ins'
        #            ]
        #        self._input_arg_names_rest = [
        #            '_T_ext', '_T_port', '_T_s', '_T_amb', 'ports_all',  # temperatures
        #            '_dm_io', '_dm_top', '_dm_bot', '_dm_port', 'res_dm',  # flows
        #            '_cp_T', '_lam_T', '_rho_T', '_ny_T', '_lam_mean', '_cp_port',
        #            '_lam_port_fld',  # mat. props.
        #            '_mcp', '_rhocp',
        #            '_ui',  # material properties.
        #            '_alpha_i', '_alpha_inf',  # alpha values
        #            '_UA_tb', '_UA_amb_shell', '_UA_port',
        #            '_UA_port_wll',  # UA values
        #            '_port_own_idx', '_port_link_idx',  # indices
        #            '_port_gsp', '_port_subs_gsp',
        #            '_cell_dist',  # lengths
        #            '_A_p_fld_mean',  # areas and vols
        #            '_process_flows', '_step_stable',  # bools
        #            '_stability_breaches', '_vN_max_step',
        #            '_max_factor',  # misc.
        #            'stepnum',  # step information
        #            'dT_cond', 'dT_adv', 'dT_total',  # differentials
        #            ]

        #        # STRUCTURED ARRAYS FOR ALMOST ALL ARRAYS, EXCEPT FOR EXTERNAL VIEWS:
        #        # arguments for full recarray version (all except views):
        #        self._input_arg_names_fullrecarr = [
        #            '_T_port', '_T_s', '_T_amb',  # temperatures
        #            '_dm_top', '_dm_bot', '_dm_port',  # flows
        #            '_lam_T', '_rho_T', '_ny_T', '_cp_port',
        #            '_lam_port_fld',  # mat. props.
        #            '_mcp', '_rhocp', '_lam_wll', '_lam_ins',
        #            '_mcp_wll', '_ui',  # material properties.
        #            '_alpha_i', '_alpha_inf',  # alpha values
        #            '_UA_tb_wll', '_UA_amb_shell', '_UA_port',
        #            '_UA_port_wll',  # UA values
        #            '_port_own_idx', #'_port_link_idx',  # indices
        #            'grid_spacing', '_port_gsp', #'_port_subs_gsp',
        #            '_d_i', '_cell_dist',  # lengths
        #            '_flow_length', '_r_total', '_r_ln_wll', '_r_ln_ins',
        #            '_r_rins',  # lengths
        #            'A_cell', 'V_cell', '_A_shell_i',
        #            '_A_shell_ins', '_A_p_fld_mean',  # areas and vols
        #              # bools
        #            'part_id', '_stability_breaches',  # misc.
        #            'dT_cond', 'dT_adv', 'dT_total',  # differentials
        #            ]
        #
        #        self._input_arg_names_full_Tsize = [  # arrays of T size
        #            'T_s', 'dm_top', 'dm_bot',
        #            'lam_T', 'rho_T', 'ny_T', 'mcp', 'rhocp', 'ui', 'alpha_i',
        #            'alpha_inf',
        #            'UA_amb_shell', 'cell_dist', 'dT_cond', 'dT_adv', 'dT_total',
        #            ]
        #
        #        self._input_arg_names_full_psize = [  # arrays of port size
        #            'T_port', 'dm_port',
        #            'cp_port', 'lam_port_fld',
        #            'UA_port', 'UA_port_wll',
        #            'port_own_idx', 'port_gsp', 'A_p_fld_mean'
        #            ]
        #
        #        # arguments which are views to other arrays and thus need to be
        #        # excluded from full recarrays:
        #        self._input_arg_names_views = [
        #            '_T_ext',
        #            'ports_all',  # temperatures
        #            '_dm_io', 'res_dm'
        #            '_cp_T', '_lam_mean', '_UA_tb',
        #            '_port_link_idx',
        #            '_port_subs_gsp',
        #            '_step_stable',  # bools
        #            '_vN_max_step',
        #            '_max_factor',  # misc.
        #            '_process_flows', '_vertical',
        #            'stepnum',  # step information
        #            ]
        #
        #        ra1 = []  # for floats and (1,) shape arrays
        #        ra2 = []  # for arrays of port size
        #        ra5 = []  # for arrays of T size
        #        for arg in self._input_arg_names_fullrecarr:
        #            if type(self.__dict__[arg]) == np.ndarray:
        #                shp = self.__dict__[arg].shape
        #                if shp[0] == 1:
        #                    ra1.append((arg.lstrip('_'), self.__dict__[arg].dtype))
        #                elif arg.lstrip('_') in self._input_arg_names_full_psize:
        #                    ra2.append((arg.lstrip('_'), self.__dict__[arg].dtype))
        #                elif arg.lstrip('_') in self._input_arg_names_full_Tsize:
        #                    ra5.append((arg.lstrip('_'), self.__dict__[arg].dtype))
        #            else:
        #                ra1.append((arg.lstrip('_'), type(self.__dict__[arg])))
        #
        #        self._ra1 = np.zeros((1, ), dtype=ra1)
        #        shp2 = self._T_port.shape
        #        self._ra2 = np.zeros(shp2, dtype=ra2)
        #        shp5 = self.T.shape
        #        self._ra5 = np.zeros(shp5, dtype=ra5)
        #
        #        for arg in self._input_arg_names_fullrecarr:
        #            if (arg not in ['_port_link_idx', '_port_subs_gsp']):
        #                if type(self.__dict__[arg]) == np.ndarray:
        #                    shp = self.__dict__[arg].shape
        #                    if shp[0] == 1:
        #                        self._ra1[arg.lstrip('_')] = self.__dict__[arg]
        #                    elif arg.lstrip('_') in self._input_arg_names_full_psize:
        #                        self._ra2[arg.lstrip('_')] = self.__dict__[arg]
        #                    elif arg.lstrip('_') in self._input_arg_names_full_Tsize:
        #                        self._ra5[arg.lstrip('_')] = self.__dict__[arg]
        #                else:
        #                    self._ra1[arg.lstrip('_')] = self.__dict__[arg]
        #                # also return view to recarr for this arg, if arg is array:
        #                if type(self.__dict__[arg]) == np.ndarray:
        #                    shp = self.__dict__[arg].shape
        #                    if shp[0] == 1:
        #                        self.__dict__[arg] = self._ra1[arg.lstrip('_')][:]
        #                    elif arg.lstrip('_') in self._input_arg_names_full_psize:
        #                        self.__dict__[arg] = self._ra2[arg.lstrip('_')][:]
        #                    elif arg.lstrip('_') in self._input_arg_names_full_Tsize:
        #                        self.__dict__[arg] = self._ra5[arg.lstrip('_')][:]
        #
        #        # reconstr views:
        #        self.T = self._T_ext[1:-1]  # memview of T_ext
        #        self._T_top = self._T_ext[:-2]  # memview of T_ext
        #        self._T_bot = self._T_ext[2:]
        #        self._cp_top = self._cp_T[:-2]  # view for top and bot grid
        #        self._cp_bot = self._cp_T[2:]

        # set initialization to true:
        self.initialized = True

    def _reset_to_init_cond(self):
        self._dm_io[:] = 0
        self.T[:] = self._T_init

    def _get_flow_routine(
        self, port, parent_port=None, subnet=False, **kwargs
    ):
        """
        Returns the massflow calculation routine for the port of the current
        part to the topology construction. The massflow calculation routine has
        to look like:

        routine = (memory_view_to_target_port,
                   operation_id,
                   memory_view_to_port1, memory_view_to_port2, ...)

        with target_port being the port which has to be calculated and port1
        and port2 being the other/source ports which **don't** have to be
        calculated with this routine! These source ports **must be given**
        when the routine is called.

        Parameters:
        -----------
        port : string
            Port name of the port which shall be calculated (target port).

        """

        # get topology connection conditions (target index, source part/port
        # identifiers, source index and algebraic sign for passed massflow):
        trgt_idx, src_part, src_port, src_idx, alg_sign = self._get_topo_cond(
            port, parent_port
        )

        # if no port has been solved yet:
        if self._cnt_open_prts == 2:
            # Pipes without massflow affecting behaviour will always only make
            # a memory view to the parent part's massflow cell, thus no
            # operation is required at all, IF THE PIPE'S IN PORT IS CONNECTED
            # TO src_part's OUT PORT (or vice versa)!
            # otherwise operation_id -1 is needed to get the negative of the
            # value.

            # get operation id depending on the direction of positive massflow
            # through the connected ports:
            if alg_sign == 'positive':
                # if positive, only using a memory view
                operation_id = -99
            else:
                # if negative, a negative copy has to be made
                operation_id = -1

            if operation_id == -99:
                # create a memory view AND pass back a dummy tuple (for the
                # easy information lookup dict) which will be deleted later on.
                # This requires EXTREME CAUTION, since constructing this view
                # to another array KILLS all views to this array! Currently it
                # should be ok, since other parts will only access it AFTER the
                # view has been reassigned, but this is prone to errors!
                self.dm = self._models.parts[src_part]._dm_io.reshape(-1)[
                    src_idx
                ]
                # also link the dm I/O array to the new dm address:
                self._dm_io = self.dm[:]
                # create dummy tuple to return later:
                op_routine = (
                    self._dm_io.reshape(-1)[trgt_idx],
                    operation_id,
                    self._models.parts[src_part]._dm_io.reshape(-1)[src_idx],
                )
            else:
                # if port and parent port are of the same massflow sign
                # character, this port has to be calculated by getting the
                # negative of the parent port's value.
                # construct memory view of self.dm to dm_io (CAUTION! Only
                # works for dm_invariant AND if dm_io address does not change!)
                self.dm = self._dm_io.reshape(-1)[:]
                # add operation instructions to tuple (memory view to target
                # massflow array cell, operation id and memory view to the
                # source port's massflow array cell:
                op_routine = (
                    self._dm_io.reshape(-1)[trgt_idx],
                    operation_id,
                    self._models.parts[src_part]._dm_io.reshape(-1)[src_idx],
                )
        else:
            # return empty tuple to calling function if only the last port is
            # left, since Pipe has the same massflow in all cells
            operation_id = -99
            op_routine = ()

        # update solved ports list and counter stop break:
        self._solved_ports.append(port)
        self._cnt_open_prts = self.port_num - len(self._solved_ports)
        # this stays always False for 3w Valve!
        self.break_topology = False
        # remove part from hydr_comps if completely solved:
        if self._cnt_open_prts == 0:
            self._models._hydr_comps.remove(self.name)

        # save topology parameters to dict for easy information lookup:
        net = 'Subnet' if subnet else 'Flownet'
        operation_routine = (
            'Negative of sum'
            if operation_id == -1
            else 'Sum'
            if operation_id == 1
            else 'Pass on value'
            if operation_id == 0
            else 'Multiplication with port factor'
            if operation_id == 3
            else 'Directly passed by memory view'
        )
        src_part = src_part if src_part is not None else self.name
        source_ports = src_port
        # add port dict for current port and fill it:
        if port not in self.info_topology:
            self.info_topology[port] = dict()
        self.info_topology[port].update(
            {
                'Net': net,
                'Massflow': self._dm_io.reshape(-1)[:],
                'Calculation routine': operation_routine,
                'Source part': src_part,
                'Source port(s)': source_ports,
                'Connected part': (
                    self._models.port_links[self.name + ';' + port].split(';')[
                        0
                    ]
                ),
                'Connected port': (
                    self._models.port_links[self.name + ';' + port].split(';')[
                        1
                    ]
                ),
                'Parent pump/part': kwargs['parent_pump'],
                'Pump side': kwargs['pump_side'],
            }
        )

        return op_routine

    """
    Optimierung:
        - _port_values_to_idx ca. 3.5% weniger Zeit
        - views für top/bot spart 0.5% Zeit
        - views für cp top/bot spart 1%!! Obwohl Arrays so klein und dadurch
          +2 Zellen gleich deutlicher Mehraufwand für get_cp!
          ---> Function call overhead >> Eigentliche Ausführung
        - process ports a: 1.2%, insgesamt *1.0617 oder ca weniger
        - process ports b: 1.4%, insgesamt *1.0642 oder ca. 1.0575
        - process ports loop: 1.8%, insgesamt *1.068
        - process ports loop 2: ca. 15% weniger Zeit!!! insgesamt 1.206-1.213
        - get water props: ca. 3-4% weniger Zeit! insgesamt 1.238-1.241
        (- lam mean: ca. 0.6% weniger Zeit. insgesamt 1.244)
        - lam mean and UA tb: 6.5% weniger Zeit! insgesamt 1.306
        - cell_temp_props: 4.5% weniger Zeit! insgesamt 1.351
        - proc flows: 2%  langsamer! sollte aber später besser werden.
                      insgesamt 1.334. update flow net ca. 20ms langsamer
        - diese bisherigen packen in pipe_diff: weitere 35%! insgesamt 1.686!
    """

    def get_diff(self, timestep):
        """
        This function just calls a jitted calculation function.

        """

        #        (self.dT_total, self._process_flows, self._models._step_stable,
        #         self._models._vN_max_step, self._models._max_factor) = pipe1D_diff(
        #            T_ext=self._T_ext, T_port=self._T_port, T_s=self._T_s,
        #            T_amb=self._T_amb, ports_all=self._models.ports_all,  # temperatures
        #            dm_io=self._dm_io, dm_top=self._dm_top, dm_bot=self._dm_bot,
        #            dm_port=self._dm_port, res_dm=self.res_dm,  # flows
        #            cp_T=self._cp_T, lam_T=self._lam_T, rho_T=self._rho_T,
        #            ny_T=self._ny_T, lam_mean=self._lam_mean, cp_port=self._cp_port,
        #            lam_port_fld=self._lam_port_fld, mcp=self._mcp, rhocp=self._rhocp,
        #            lam_wll=self._lam_wll, lam_ins=self._lam_ins,
        #            mcp_wll=self._mcp_wll, ui=self._ui,  # material properties
        #            alpha_i=self._alpha_i, alpha_inf=self._alpha_inf,  # alpha values
        #            UA_tb=self._UA_tb, UA_tb_wll=self._UA_tb_wll,
        #            UA_amb_shell=self._UA_amb_shell, UA_port=self._UA_port,
        #            UA_port_wll=self._UA_port_wll,  # UA values
        #            port_own_idx=self._port_own_idx, port_link_idx=self._port_link_idx,  # indices
        #            grid_spacing=self.grid_spacing, port_gsp=self._port_gsp,
        #            port_subs_gsp=self._port_subs_gsp, d_i=self._d_i,
        #            cell_dist=self._cell_dist, flow_length=self._flow_length,  # lengths
        #            r_total=self._r_total, r_ln_wll=self._r_ln_wll,
        #            r_ln_ins=self._r_ln_ins, r_rins=self._r_rins,  # lengths
        #            A_cell=self.A_cell, V_cell=self.V_cell, A_shell_i=self._A_shell_i,
        #            A_shell_ins=self._A_shell_ins, A_p_fld_mean=self._A_p_fld_mean,  # areas and vols
        #            process_flows=self._process_flows, vertical=self._vertical,
        #            step_stable=self._models._step_stable,  # bools
        #            part_id=self.part_id, stability_breaches=self._stability_breaches,
        #            vN_max_step=self._models._vN_max_step,
        #            max_factor=self._models._max_factor,  # misc
        #            stepnum=self.stepnum, timestep=timestep,  # step information
        #            dT_cond=self.dT_cond, dT_adv=self.dT_adv  # differentials
        #            )

        #        pipe1D_diff(
        #            T_ext=self._T_ext, T_port=self._T_port, T_s=self._T_s,
        #            T_amb=self._T_amb, ports_all=self._models.ports_all,  # temperatures
        #            dm_io=self._dm_io, dm_top=self._dm_top, dm_bot=self._dm_bot,
        #            dm_port=self._dm_port, res_dm=self.res_dm,  # flows
        #            cp_T=self._cp_T, lam_T=self._lam_T, rho_T=self._rho_T,
        #            ny_T=self._ny_T, lam_mean=self._lam_mean, cp_port=self._cp_port,
        #            lam_port_fld=self._lam_port_fld, mcp=self._mcp, rhocp=self._rhocp,
        #            lam_wll=self._lam_wll, lam_ins=self._lam_ins,
        #            mcp_wll=self._mcp_wll, ui=self._ui,  # material properties
        #            alpha_i=self._alpha_i, alpha_inf=self._alpha_inf,  # alpha values
        #            UA_tb=self._UA_tb, UA_tb_wll=self._UA_tb_wll,
        #            UA_amb_shell=self._UA_amb_shell, UA_port=self._UA_port,
        #            UA_port_wll=self._UA_port_wll,  # UA values
        #            port_own_idx=self._port_own_idx, port_link_idx=self._port_link_idx,  # indices
        #            grid_spacing=self.grid_spacing, port_gsp=self._port_gsp,
        #            port_subs_gsp=self._port_subs_gsp, d_i=self._d_i,
        #            cell_dist=self._cell_dist, flow_length=self._flow_length,  # lengths
        #            r_total=self._r_total, r_ln_wll=self._r_ln_wll,
        #            r_ln_ins=self._r_ln_ins, r_rins=self._r_rins,  # lengths
        #            A_cell=self.A_cell, V_cell=self.V_cell, A_shell_i=self._A_shell_i,
        #            A_shell_ins=self._A_shell_ins, A_p_fld_mean=self._A_p_fld_mean,  # areas and vols
        #            process_flows=self._process_flows, vertical=self._vertical,
        #            step_stable=self._models._step_stable,  # bools
        #            part_id=self.part_id, stability_breaches=self._stability_breaches,
        #            vN_max_step=self._models._vN_max_step,
        #            max_factor=self._models._max_factor,  # misc
        #            stepnum=self.stepnum,  # step information
        #            dT_cond=self.dT_cond, dT_adv=self.dT_adv, dT_total=self.dT_total,  # differentials
        #            timestep=timestep)
        _pf.pipe1D_diff(*self._input_args, timestep)

        #        pipe1D_diff_structarr(*self.rest, self.strct_flt, self._vertical,
        #                              self.part_id, timestep)

        #        pipe1D_diff_fullstructarr(
        #            self._T_ext,
        #            self.ports_all,  # temperatures
        #            self._dm_io, self.res_dm,
        #            self._cp_T, self._lam_mean, self._UA_tb,
        #            self._port_link_idx,
        #            self._port_subs_gsp,
        #            self._step_stable,  # bools
        #            self._vN_max_step,
        #            self._max_factor, self._process_flows, self._vertical,  # misc.
        #            self.stepnum, self._ra1, self._ra2, self._ra5, timestep)

        #        pipe1D_diff_fast(*self._input_args_fast, timestep)

        #        pipe1D_diff(*self._input_args_list, timestep)
        #        pipe1D_diff(**self._input_args_dict, timestep=timestep)

        #        self._diff_clsd(**self._input_args_call, timestep=timestep)

        #        if self.name is 'RohrVL_Ventil' and self._dm_io[0] != 0:
        #            raise ValueError

        # ---> data class
        #        (self._process_flows, self._models._step_stable,
        #         self._models._vN_max_step) = pipe1D_diff_cls(
        #            data_cls=self._data_cls, ports_all=self._models.ports_all,
        #            process_flows=self._process_flows,
        #            step_stable=self._models._step_stable, vN_max_step=self._models._vN_max_step,
        #            stepnum=self.stepnum, timestep=timestep,  # step information
        #            )
        # ---> named tuple
        #        (self._process_flows, self._models._step_stable,
        #         self._models._vN_max_step, self._models._max_factor) = pipe1D_diff_nt(
        #            data_nt=self._data_nt, ports_all=self._models.ports_all,
        #            process_flows=self._process_flows,
        #            step_stable=self._models._step_stable, vN_max_step=self._models._vN_max_step,
        #            max_factor=self._models._max_factor,
        #            stepnum=self.stepnum, timestep=timestep,  # step information
        #            )

        return self.dT_total
