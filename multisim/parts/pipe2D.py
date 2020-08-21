# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:29:51 2018

@author: Johannes
"""


import numpy as np

from .. import simenv as _smnv
from .. import precomp_funs as _pf


class Pipe2D(_smnv.Models):
    """
    type: Single Pipe with a second dimension storing the wall temperatures.

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
        print(
            'using ny_T for alpha i calc: is it ok to only use fluid temp.'
            'ny_T or should I use a mean-temp ny_T_mean of '
            'T_mean = (T_fld + T_wll)/2 ?'
        )

        self.constr_type = 'Pipe2D'
        base_err = (  # define leading base error message
            'While adding {0} `{1}` to the simulation '
            'environment, the following error occurred:\n'
        ).format(self.constr_type, str(name))
        arg_err = (  # define leading error for missing/incorrect argument
            'Missing argument or incorrect type/value: {0}\n\n'
        )
        self._base_err = base_err  # save to self to access it in methods
        self._arg_err = arg_err  # save to self to access it in methods

        # save set of known kwargs to be able to show errors if unknown kwargs
        # or typos have been passed:
        #        known_kwargs = set(('length', 'grid_points', 'material', 'pipe_specs'))
        # assert if kwargs dict exists:
        err_str = (
            'No additional arguments have been passed to `add_part()` '
            'for pipe `' + name + '`! Pass at least '
            'the pipe length in [m] with `length=X` as an integer or '
            'float value >0 to get additional information which '
            'arguments are required.'
        )
        assert kwargs, err_str
        # assert that only known kwargs have been passed (set comparison must
        # be empty):
        #        err_str = ('The following arguments passed to `add_part()` for '
        #                   'Pipe2D ' + name + ' were not understood:'
        #                   '\n' + ', '.join(set(kwargs.keys()) - known_kwargs) + '\n'
        #                   'Please check the spelling!')
        #        assert set(kwargs.keys()) - known_kwargs == set(), err_str
        # assert that all required arguments have been passed:
        err_str = (
            'The pipe length has to be passed to its '
            '`add_part()` method in [m] with `length=X`!'
        )
        assert (
            'length' in kwargs
            and isinstance(kwargs['length'], (int, float))
            and kwargs['length'] > 0
        ), err_str
        self.length = kwargs['length']
        err_str = (
            'The number of grid points for pipe `'
            + name
            + '` has to be passed to its `add_part()` method as an '
            'integer value >0 with `grid_points=X`.'
        )
        assert (
            'grid_points' in kwargs
            and type(kwargs['grid_points']) == int
            and kwargs['grid_points'] > 0
        ), err_str
        self.num_gp = kwargs['grid_points']

        # ---> start part construction:
        #        super().__init__()
        self.name = name
        self.part_id = self._models.num_parts - 1

        # save smallest possible float number for avoiding 0-division:
        self._tiny = self._models._tiny
        self._models = master_cls

        # get grid spacing. other geometry definition will be done after
        # getting ports and pipe specs.
        self.grid_spacing = self.length / self.num_gp
        # set array for each cell's distance from the start of the pipe:
        self._cell_dist = np.arange(
            self.grid_spacing / 2, self.length, self.grid_spacing
        )

        # ---> preallocate temperature and property grids:
        # first column: fluid temp., 2.: wall temp., 3.: surface temp.
        self._T_ext = np.zeros((self.num_gp + 2, 3), dtype=np.float64)
        #        self.T = np.zeros((self.num_gp, 2), dtype=np.float64)
        self.T = self._T_ext[1:-1]
        # preallocate views to temperature grid for upper and lower cells
        self._T_top = self._T_ext[:-2]
        self._T_bot = self._T_ext[2:]
        # preallocate temperature grid for ports:
        self._T_port = np.zeros(self.T.shape[0])
        # preallocate lambda grids and alpha grid for heat conduction with the
        # smallest possible value to avoid zero division:
        self._lam_T = np.zeros(self.num_gp)
        self._lam_mean = np.zeros(
            (self.num_gp - 1, 3)
        )  # mean value between cells
        self._alpha_i = np.full_like(self._lam_T, 3.122)
        self._alpha_inf = np.full_like(self._lam_T, 3.122)
        # preallocate fluid heat capacity grids:
        self._cp_T = np.zeros(self.num_gp + 2)  # extended array
        self._cp_top = self._cp_T[:-2]  # view for top and bot grid
        self._cp_bot = self._cp_T[2:]
        self._cp_port = np.zeros_like(self._T_port) + 1
        # preallocate fluid density grids:
        self._rho_T = np.zeros(self.num_gp)
        self._rhocp = np.zeros_like(self._rho_T)  # volume specific heat cap.
        # preallocate kinematic viscosity grid:
        self._ny_T = np.zeros(self.num_gp)
        # cell mass specific inner energy grid:
        self._ui = np.zeros_like(self.num_gp)
        # heat capacity of fluid AND wall:
        self._mcp = np.zeros_like(self.num_gp)
        # preallocate U*A grids:
        # heat cond. in axial direction. ax. heat cond. in ins. is set to 0
        self._UA_tb = np.zeros((self.T.shape[0] + 1, self.T.shape[1]))
        self._UA_port = np.zeros_like(self._T_port)
        #        self._UA_amb_shell = np.zeros_like(self.T)

        self._UA_radial = np.zeros_like(self.T)
        #        self._UA_fld_wll = np.zeros(self.T.shape[0])
        #        self._UA_wll_amb = np.zeros(self.T.shape[0])
        # preallocate mass flow grids:
        self.dm = np.zeros(self.T.shape[0])
        self._dm_cell = np.zeros_like(self.T[:, 0])  # cell center flow
        self._dm_top = np.zeros_like(self.T[:, 0])  # pos. flow from top
        self._dm_bot = np.zeros_like(self.T[:, 0])  # pos. flow from bottom
        self._dm_port = np.zeros_like(self._T_port)  # inflow through prts
        self._dm_io = np.zeros_like(self._T_port)  # array for I/O flow
        # this I/O flow array MUST NOT BE CHANGED by anything else than
        # _update_FlowNet() method!

        # preallocate result grid with one row. An estimate of total rows will
        # be preallocated before simulation start in initialize_sim. massflow
        # grid is preallocated in set_initial_cond:
        self.res = np.zeros((1,) + self.T.shape)
        self.res_dm = np.zeros((2, 1))
        # separate result grids for conduction and advection to make the code
        # less complicated:
        self.dT_cond = np.zeros_like(self.T)
        self.dT_adv = np.zeros_like(self.T[:, 0])

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
        # is pos. Since a TES may need a cumulative sum over all ports to get
        # the massflow of the last port, massflow is ALWAYS positive when
        # inflowing!
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
        self._A_wll_conn_p = np.zeros_like(self._T_port)
        self._A_fld_conn_p = np.zeros_like(self._T_port)
        self._port_gsp = np.zeros_like(self._T_port)
        self._lam_wll_conn_p = np.full_like(self._T_port, 1e-2)
        self._lam_port_fld = np.zeros_like(self._T_port) + 1
        self._lam_fld_own_p = np.full_like(self._T_port, 0.6)
        self._UA_port_fld = np.zeros_like(self._T_port)
        # preallocate list to mark ports which have already been solved in
        # topology (to enable creating subnets)
        self._solved_ports = list()
        # preallocate grids for von Neumann stability checking:
        self._vN_diff = np.zeros(4)
        #        self._vN_dm = np.zeros(1)  # this is a scalar for dm_invariant=True
        self._trnc_err = 0
        # add truncation error cell weight to weight or disable the trunc. err.
        # calculation in adaptive steps for the part (weight has to be a single
        # float or integer) or for specific cells (weight has to be an array
        # of the shape of self.T):
        self._trnc_err_cell_weight = 1
        # counter for how many times THIS part has breached von Neumann stab.
        self._stability_breaches = np.zeros(1).astype(int)  # array for view
        # preallocate grids for precalculated stuff: REPLACED WITH LOCAL VARS!
        #        self.__rhocp = np.zeros_like(self.T)
        #        self.__cpT = np.zeros_like(self.T)

        # set if type has to be solved numeric:
        self.solve_numeric = True
        # if part can be solved explicitly (otherwise a time-costly implicit
        # solving will be used):
        self.solve_explicitly = True
        # if port arrays shall be collapsed to amount of ports to improve speed
        self.collapse_arrays = False
        self._collapsed = False  # bool checker if already collapsed
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
        self._diff_fully_njit = False
        # self._diff_njit = pipe1D_diff  # handle to njitted diff function
        # input args are created in simenv _create_diff_inputs method

    def init_part(self, **kwargs):
        # check if kwargs given:
        err_str = (
            'No parameters have been passed to the `add_part()` method'
            ' of Pipe2D `' + self.name + '`! To get '
            'additional information about which parameters to pass, '
            'at least the insulation thickness in [m] has to be given '
            '(0 is allowed) with \'insulation_thickness=...\'.'
        )
        assert kwargs, err_str

        # check for insulation:
        err_str = (
            'The insulation thickness of Pipe2D has '
            'to be passed to its `add_part()` method in [m] with '
            '`insulation_thickness=X`!'
        )
        assert 'insulation_thickness' in kwargs, err_str
        self._s_ins = kwargs['insulation_thickness']

        err_str = (
            'The lambda value of Pipe2D insulation has to be '
            'passed to its `add_part()` method in '
            '[W/(m*K)] with `insulation_lambda=X`!'
        )
        assert 'insulation_lambda' in kwargs, err_str
        self._lam_ins = kwargs['insulation_lambda']

        # assert material and pipe specs (after defining port names to avoid
        # name getting conflicts):
        err_str = (
            '`pipe_specs` and `material` for Pipe2D '
            + self.name
            + ' must be passed to its `init_part()` method!'
        )
        assert 'material' in kwargs and 'pipe_specs' in kwargs, err_str
        # get material pipe specs for diameter etc.:
        self._get_specs_n_props(**kwargs)
        # assert that pipe specs were given for all ports together:
        err_str = (
            'For a part of type Pipe2D, the same `pipe_specs` must be '
            'given for all ports by specifying \'all\' as '
            'first-level-key!'
        )
        assert 'all' in kwargs['pipe_specs'], err_str

        # assert and get initial temperature:
        err_str = (
            'The initial temperature `T_init` in [°C] has to be passed '
            'to the `add_part()` method of part ' + self.name + ' as '
            'a single float or integer value or as an array with shape '
            '(' + str(self.num_gp) + ',).'
        )
        assert 'T_init' in kwargs, err_str
        self._T_init = kwargs['T_init']
        assert isinstance(self._T_init, (float, int, np.ndarray)), err_str
        # set init values to T array:
        if type(self._T_init) != np.ndarray:
            self.T[:] = self._T_init
        else:
            # assert that correct shape is given and if yes, save to T
            assert self._T_init.shape == self.T.shape, err_str
            self.T[:] = self._T_init

        # geometry/grid definition
        self._d_i = self.info_topology['all_ports']['pipe_specs']['d_i']
        self._d_o = self.info_topology['all_ports']['pipe_specs']['d_o']
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
        #        self._UA_tb_wll = self._A_shell_cs / self.grid_spacing * self._lam_wll

        # calculate m*cp for the wall PER CELL with the material information:
        self._mcp_wll = (
            self._cp_wll * self._rho_wll * self._V_shell_cs / self.num_gp
        )
        # calculate U*A heat conduction from top/bottom cells for the wall with
        # the material information (except for the first/last cell):
        self._UA_tb[1:-1, 1] = (
            self._A_shell_cs * self._lam_wll / self.grid_spacing
        )

        # get ambient temperature:
        self._chk_amb_temp(**kwargs)

        # save shape parameters for the calculation of heat losses.
        self._vertical = False
        # get flow length for a horizontal cylinder for Nusselt calculation:
        self._flow_length = _pf.calc_flow_length(
            part_shape='cylinder',
            vertical=self._vertical,
            diameter=self._r_total * 2,
        )

        # adjust arrays depending on bools set:
        if self.dm_invariant:
            # if all cells have the same massflow, reduce massflow arrays
            self.dm = np.zeros(1)
            self._dm_io = np.zeros(1)

        # reflect global ports array:
        #        self.ports_all = self._models.ports_all  # not working, since it is changed afterwards

        #        # list of local class instance variables to pass to data class:
        #        self._calc_att_list = [
        #                '_PipeOpt_T_ext', '_T_port', '_T_s',
        #                '_T_amb',  # 'ports_all',  # temperatures
        #                '_dm_io', '_dm_top', '_dm_bot',
        #                '_dm_port', 'res_dm',  # flows
        #                '_cp_T', '_lam_T', '_rho_T',
        #                '_ny_T', '_lam_mean', '_cp_port',
        #                '_lam_port_fld', '_mcp', '_rhocp',
        #                '_lam_wll', '_lam_ins',
        #                '_mcp_wll', '_ui',  # material properties
        #                '_alpha_i', '_alpha_inf',  # alpha values
        #                '_UA_tb', '_UA_tb_wll',
        #                '_UA_amb_shell', '_UA_port',
        #                '_UA_port_wll',  # UA values
        #                '_port_own_idx', '_port_link_idx',  # indices
        #                'grid_spacing', '_port_gsp',
        #                '_port_subs_gsp', '_d_i',
        #                '_cell_dist', '_flow_length',  # lengths
        #                '_r_total', '_r_ln_wll',
        #                '_r_ln_ins', '_r_rins',  # lengths
        #                'A_cell', 'V_cell', '_A_shell_i',
        #                '_A_shell_ins', '_A_p_fld_mean',  # areas and vols
        #                '_vertical',
        #                #'_step_stable',  # bools
        #                'part_id',  # misc
        #                'dT_cond', 'dT_adv'  # differentials
        #                ]
        #
        #        self.new_list = []
        #        for lmnt in self._calc_att_list:
        #            if lmnt[0] == '_':
        #                self.new_list.append(lmnt[1:])
        #            else:
        #                self.new_list.append(lmnt)

        # set initialization to true:
        self.initialized = True

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
            else ('Multiplication ' 'with port factor')
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

    def get_diff(self, timestep):
        # ---> get temperature arrays
        # get temperatures of connected ports:
        self._port_vals[:] = self._models.ports_all[self._port_link_idx]
        # save port values to ports array:
        """
        TO DO: once this is migrated to numba, this needs to be rewritten to
        use a for loop (fastest way is a for loop in numba!)
        """
        self._T_port[self._port_own_idx] = self._port_vals
        # get new values for top and bot grids:
        self._T_top[1:] = self.T[:-1]
        self._T_bot[:-1] = self.T[1:]

        self._process_flows = _pf._process_flow_invar(
            process_flows=self._process_flows,
            dm_io=self._dm_io,
            dm_top=self._dm_top,
            dm_bot=self._dm_bot,
            dm_port=self._dm_port,
            stepnum=self.stepnum,
            res_dm=self.res_dm,
        )

        _pf.water_mat_props_ext(
            T_ext=self._T_ext[:, 0],
            cp_T=self._cp_T,
            lam_T=self._lam_T,
            rho_T=self._rho_T,
            ny_T=self._ny_T,
        )

        # ---> get thermodynamic properties of water
        # get them for port temperatures:
        _pf.get_cp_water(self._T_port, self._cp_port)
        _pf.get_lambda_water(self._T_port, self._lam_port_fld)
        #        get_lambda(self.T, self._lam_fld_own_p)  # done by indexing
        # for bottom and top temps
        #        self._cp_top[1:] = self._cp_T[:-1]
        #        self._cp_bot[:-1] = self._cp_T[1:]
        #        self._lam_top[1:] = self._lam_T[:-1]
        #        self._lam_bot[:-1] = self._lam_T[1:]
        # for own ports in shape of _T_port:
        self._lam_fld_own_p[self._port_own_idx] = self._lam_T[
            self._port_own_idx
        ]

        # get mean lambda of the path between two fluid cell centers:
        self._lam_mean = (
            2
            * self._lam_T[:-1]
            * self._lam_T[1:]
            / (self._lam_T[:-1] + self._lam_T[1:])
        )

        # ---> calculate U values
        # for conduction between current cell and cell on top (first cell 0):
        #        self._UA_top[1:] = (
        #                self.A_cell / (self.grid_spacing / (2 * self._lam_T[1:])
        #                               + self.grid_spacing / (2 * self._lam_top[1:])))
        #        # for conduction between current cell and cell at bottom (last cell 0):
        #        self._UA_bot[:-1] = (
        #                self.A_cell / (self.grid_spacing / (2 * self._lam_T[:-1])
        #                               + self.grid_spacing / (2 * self._lam_bot[:-1])))
        # for conduction between current fluid cell and cell on top or bottom
        # (first and last cell is 0, to enable using a full view of this array
        # for heat flow calculation):
        self._UA_tb[1:-1, 0] = self.A_cell / self.grid_spacing * self._lam_mean
        # for conduction between current cell and cells at ports:
        self._UA_fld_ports[:] = self._A_p_fld_mean / (
            +(self._port_gsp / (2 * self._lam_port_fld))
            + (self.grid_spacing / (2 * self._lam_fld_own_p))
        )
        # parallel circuit of heat conduction through wall and fluid of ports:
        #        self._UA_port[:] = self._UA_fld_ports + self._UA_port_wll
        # for conduction between current cell and ambient:
        # get inner alpha value between fluid and wall from nusselt equations:
        _pf.pipe_alpha_i_wll_sep(
            self.dm,
            self.T,
            self._rho_T,
            self._ny_T,
            self._lam_T,
            self.A_cell,
            self._d_i,
            self._cell_dist,
            self._alpha_i,
        )
        # get resulting UA from the fluid to the mean radius of the wall:
        _pf.UA_fld_wll(
            self._A_shell_i,
            self._r_i,
            self._r_o,
            self._alpha_i,
            self._lam_wll,
            self._UA_fld_wll,
        )
        # get UA from the mean radius of the wall to the ambient:
        _pf.UA_wll_ins_amb(
            self._A_shell_i,
            self._r_i,
            self._r_o,
            self._r_ln_ins,
            self._r_rins,
            self._alpha_inf,
            self._lam_wll,
            self._lam_ins,
            self._UA_wll_amb,
        )

        # precalculate values which are needed multiple times:
        # volume specific heat capacity:
        rhocp = self._rho_T * self._cp_T
        # heat capacity of fluid (wall heat cap. is const. -> precalculated!):
        self._mcp_fld = self.V_cell * rhocp
        # mass specific inner energy (only used in advection, thus only the
        # fluid part of T is used):
        ui = self._cp_T * self.T[:, 0]

        # ---> check for L2/von Neumann stability
        # for diffusion:
        # save von Neumann stability values for cells by multiplying the cells
        # relevant total x-gridspacing with the maximum UA-value (this gives a
        # substitue heat conduction to get a total diffusion coefficient) and
        # the inverse maximum rho*cp value (of all cells! this may result in a
        # worst-case-result with a security factor of up to about 4.2%) to get
        # the substitute diffusion coefficient and then mult. with step and
        # div. by gridspacing**2 (not **2 since this is cut out with mult. by
        # it to get substitute diffusion from UA) and save to array:
        self.__rhocpmax = rhocp.max()
        self._vN_diff[0] = (
            (self._UA_tb.max() / self.__rhocpmax)
            * timestep
            / self.grid_spacing
        )
        #        self._vN_diff[1] = ((self._UA_bot.max() / self.__rhocpmax)
        #                            * timestep / self.grid_spacing)
        # for the next two with non-constant gridspacing, find max of UA/gsp:
        self._vN_diff[1] = (
            (self._UA_port / self._port_subs_gsp).max()
            / self.__rhocpmax
            * timestep
        )
        self._vN_diff[2] = (
            self._UA_amb.max() / self._r_total / self.__rhocpmax * timestep
        )
        # for massflow:
        # get maximum cfl number (this is the von Neumann stability condition
        # for massflow through cells), again with total max. of rho to get a
        # small security factor for worst case:
        self.__Vcellrhomax = self.V_cell * self._rho_T.max()
        self._vN_dm = abs(self.dm)[0] * timestep / self.__Vcellrhomax

        # get maximum von Neumann stability condition values:
        vN_diff_max = self._vN_diff.max()
        #        vN_dm_max = max(self._vN_dm)  # not needed since dm_invariant=True
        # get dividers for maximum stable timestep to increase or decrease
        # stepsize:
        vN_diff_mult = vN_diff_max / 0.5
        vN_dm_mult = self._vN_dm / 1
        # get biggest divider:
        vN_div_max = max(vN_diff_mult, vN_dm_mult)
        # check if any L2 stability conditions are violated:
        if vN_div_max > 1:
            # only do something if von Neumann checking is active, else just
            # print an error but go on with the calculation:
            if self._models._check_vN:
                # if not stable, set stable step bool to False
                self._models._step_stable = False
                self._stability_breaches += 1
                # calculate required timestep to make this part stable with a
                # security factor of 0.95:
                vN_max_step = timestep / vN_div_max * 0.95
                # if this is the smallest step of all parts needed to make all
                # parts stable save it to maximum von Neumann step:
                if self._models._vN_max_step > vN_max_step:
                    self._models._vN_max_step = vN_max_step
            else:
                print(
                    '\nVon Neumann stability violated at step '
                    + str(self.stepnum)
                    + ' and part '
                    + self.name
                    + '!'
                )
                raise ValueError
        elif self._models._max_speed:
            # else get new maximum timestep for this part if maximum speed
            # option is set:
            raise ValueError(
                'Checking with other parts is missing! This '
                'should only get the max error of the weakest '
                'part!'
            )
            timestep /= vN_div_max

        """
        TODO: Temporarily set port heat conduction to 0!
        """
        #        self._UA_port[:] = 0
        #        if not (abs((self._UA_top * (self._T_top - self.T)
        #                + self._UA_bot * (self._T_bot - self.T)).sum()) == 0 or
        #            abs((self._UA_top * (self._T_top - self.T)).sum()
        #                + (self._UA_bot * (self._T_bot - self.T)).sum()) == 0):
        #            raise ValueError

        # ---> CALCULATE DIFFERENTIALS
        # calculate heat transfer by conduction
        #        self.dT_cond[:] = (
        #                (+ self._UA_top * (self._T_top - self.T)
        #                 + self._UA_bot * (self._T_bot - self.T)
        #                 + self._UA_port * (self._T_port - self.T)
        #                 + self._UA_amb * (self._T_amb - self.T))
        #                / self._mcp)
        # calculate heat transfer by conduction for the fluid:
        self.dT_cond[:, 0] = (
            +self._UA_tb[:-1, 0] * (self._T_top[:, 0] - self.T[:, 0])
            + self._UA_tb[1:, 0] * (self._T_bot[:, 0] - self.T[:, 0])
            + self._UA_fld_ports * (self._T_port - self.T[:, 0])
            + self._UA_fld_wll * (self.T[:, 1] - self.T[:, 0])
        ) / self._mcp_fld
        # calculate heat transfer by conduction for the wall:
        self.dT_cond[:, 1] = (
            +self._UA_tb[:-1, 1] * (self._T_top[:, 1] - self.T[:, 1])
            + self._UA_tb[1:, 1] * (self._T_bot[:, 1] - self.T[:, 1])
            + self._UA_port_wll * (self._T_port - self.T[:, 1])
            + self._UA_wll_amb * (self._T_amb - self.T[:, 1])
        ) / self._mcp_wll
        # calculate heat transfer by advection for the fluid:
        self.dT_adv[:] = (
            +self._dm_top * (self._cp_top * self._T_top[:, 0] - ui)
            + self._dm_bot * (self._cp_bot * self._T_bot[:, 0] - ui)
            + self._dm_port * (self._cp_port * self._T_port - ui)
        ) / self._mcp_fld

        # sum all up:
        self.dT_total = self.dT_cond
        self.dT_total[:, 0] += self.dT_adv

        return self.dT_total
