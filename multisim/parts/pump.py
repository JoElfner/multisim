# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Aug 2017 2018
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as _plt

from ..simenv import SimEnv
from .. import precomp_funs as _pf


class Pump(SimEnv):
    """
    type: Pump class.
    One pump is required in each contiguous seaction of the model environment
    which is not separated from other sections by a hydraulic compensator like
    a thermal energy storage.
    The pumps determine the mass flows in these contiguous sections.
    The mass flow is calculated after each timestep and intermediate step
    depending on the given control algorithm and the measured values in the
    specified measuring port.

    The Pump class does not contain a differential method as it only passes the
    values of the part connected to its 'in'-port to its 'out'-port and the
    values of the part connected to its 'out'-port to its 'in'-port. Thus it is
    not involved in solving the equations using the specified solver algorithm.
    """

    def __init__(self, name, master_cls, **kwargs):
        self._models = master_cls

        self.constr_type = 'Pump'  # define construction type
        base_err = (  # define leading base error message
            'While adding {0} `{1}` to the simulation '
            'environment, the following error occurred:\n'
        ).format(self.constr_type, str(name))
        arg_err = (  # define leading error for missing/incorrect argument
            'Missing argument or incorrect type/value: {0}\n\n'
        )
        self._base_err = base_err  # save to self to access it in controllers
        self._arg_err = arg_err  # save to self to access it in controllers

        #        super().__init__()
        self.name = name
        self._unit = '[kg/s]'  # unit of the actuator
        self.part_id = self._models.num_parts - 1

        #        # array defining the minium and maximum massflow for the pump:
        # done in init now!
        #        self.dm_range = np.array((dm_min, dm_max))
        # save smallest possible float number for avoiding 0-division:
        self._tiny = self._models._tiny

        # even though this part is not using numeric solving, number of
        # gridpoints are specified anyways:
        self.num_gp = 2
        # preallocate grids:
        self.T = np.zeros(2, dtype=np.float64)
        self._T_init = np.zeros_like(self.T)  # init temp for resetting env.
        # preallocate T ports array (in Pump only used for dimension checking)
        self._T_port = np.zeros_like(self.T)
        #        self.dm = np.zeros(1)
        #        self.U = np.zeros(2)
        # preallocate grids for port connection parameters
        # cross section area of wall of connected pipe, fluid cross section
        # area of, gridspacing and lambda of wall of connected pipe
        self._A_wll_conn_p = np.zeros_like(self._T_port)
        self._A_fld_conn_p = np.zeros_like(self._T_port)
        self._port_gsp = np.full_like(self._T_port, self._tiny)
        self._lam_wll_conn_p = np.full_like(self._T_port, self._tiny)
        self._lam_port_fld = np.full_like(self._T_port, self._tiny)

        # port_definition (first and last array element):
        self.port_num = 2
        # Index to own value array to get values of own ports, meaning if I
        # index a FLATTENED self.T.flat with self._port_own_idx, I need to
        # get values accoring to the order given in self.port_names.
        # That is, this must yield the value of the cell in self.T, which is
        # belonging to the port 'in':
        # self.T.flat[self._port_own_idx[self.port_names.index('in')]]
        self._port_own_idx = np.array((0, self.T.shape[0] - 1), dtype=np.int32)
        self._port_own_idx_2D = self._port_own_idx  # save for compatibility
        """port_array"""
        self.port_ids = np.array((), dtype=np.int32)
        # save port names
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
        # preallocate list to mark ports which have already been solved in
        # topology (to enable creating subnets)
        self._solved_ports = list()

        # preallocate massflow grid with port_num. An estimate of total rows
        # will be preallocated before simulation start in initialize_sim:
        self.res_dm = np.zeros((2, self.port_num))

        # set if type has to be solved numeric:
        self.solve_numeric = False
        # if port arrays shall be collapsed to amount of ports to improve speed
        self.collapse_arrays = False
        self._collapsed = False  # bool checker if already collapsed
        # determine if part is treated as hydraulic compensator
        self.hydr_comp = False
        # if part can be a parent part of a primary flow net:
        self._flow_net_parent = True
        # add each flow channel of part to hydr_comps (will be removed once its
        # massflow solving method is completely integrated in flow_net.
        # remaining parts except real hydr comps will be used to generate an
        # error):
        self._models._hydr_comps.add(self.name)
        # if the topology construction method has to stop when it reaches the
        # part to solve more ports from other sides before completely solving
        # the massflow of it. This will be set to false as soon as only one
        # port to solve is remaining:
        self.break_topology = False
        # count how many ports are still open to be solved by topology. If
        # break topology is True, this is used to set it to False if 1 is
        # reached.
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
        # bool check if massflow is given for the entire program run:
        self.dm_given = False
        # if the part CAN BE controlled by the control algorithm:
        self.is_actuator = True
        # if the part HAS TO BE controlled by the control algorithm:
        self.control_req = True
        # if the part needs a special control algorithm (for parts with 2 or
        # more controllable inlets/outlets/...):
        self.actuator_special = False
        # initialize bool if control specified:
        self.ctrl_defined = False
        # if the parts get_diff method is solved with memory views entirely and
        # thus has arrays which are extended by +2 (+1 at each end):
        self.enlarged_memview = False
        # if the part has a special plot method which is defined within the
        # part's class:
        self.plot_special = True

        # save initialization status:
        self.initialized = False

        # save memory address of T
        self._memadd_T = self.T.__array_interface__['data'][0]

        # preallocate massflow grid:
        if self.dm_invariant:
            self.dm = np.zeros(1)
        else:
            self.dm = np.zeros(self.port_num)
        # and also preallocate grid for massflow through ports:
        if not self.hydr_comp:
            # if part is no hydraulic compensator, dm ports grid is simply a
            # memory view to massflow grid
            self._dm_port = self.dm[:]
            self._dm_io = self.dm[:]
        else:
            # if part is a hydraulic compensator, dm ports is separate from dm
            self._dm_port = np.zeros_like(self.T)
            self._dm_io = np.zeros_like(self.T)
        # set array where the CV is set to:
        if self.is_actuator:
            self._actuator_CV = self.dm[:]  # set array to be controlled
            self._actuator_CV_name = 'massflow'  # set description
        # save memory address of dm
        self._memadd_dm = self.dm.__array_interface__['data'][0]

        # save all kind of info stuff to dicts:
        # topology info:
        self.info_topology = dict()

        # IMPORTANT: THIS VARIABLE **MUST NOT BE INHERITED BY SUB-CLASSES**!!
        # If sub-classes are inherited from this part, this bool checker AND
        # the following variables MUST BE OVERWRITTEN!
        # ist the diff function fully njitted AND are all input-variables
        # stored in a container?
        self._diff_fully_njit = False
        # self._diff_njit = pipe1D_diff  # handle to njitted diff function
        # input args are created in simenv _create_diff_inputs method

    def init_part(self, *, start_massflow, **kwargs):
        """
        Initialize pump with specifications, material and initial conditions.
        """

        # get material properties and pipe specifications:
        self._get_specs_n_props(**kwargs)

        # gridspacing is saved in an array of length port_num to save the
        # gridspacing of connected parts for heat flux calculation. this array
        # is pre-filled with an estimate of 1.1 times the DN outer diameter but
        # will be overwritten and filled by get_port_connections() method with
        # connected part values, if any numeric parts are connected.
        # therefore get the info topology key:
        if 'in' in self.info_topology:
            key = 'in'
        else:
            key = 'all_ports'
        self.grid_spacing = np.full_like(
            self._T_port, self.info_topology[key]['pipe_specs']['d_o'] * 1.1
        )

        # for nonnumeric parts moved to initialize sim
        #        self.T = self._get_ports()

        # set starting massflow:
        self.dm[0] = start_massflow
        self._dm_init = start_massflow  # save for resetting

        # if pump has to be controlled (default) and thus is NOT set to static,
        # it needs a lower and upper limit for the values to set:
        if 'no_control' not in kwargs or (
            'no_control' in kwargs and kwargs['no_control'] is False
        ):
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
            # set limits to array:
            self._lims = np.array(  # set limits to array
                [kwargs['lower_limit'], kwargs['upper_limit']],
                dtype=np.float64,
            )
            self._llim = self._lims[0]  # also save to single floats
            self._ulim = self._lims[1]  # also save to single floats
            # check if lower limit is less than upper limit:
            assert self._lims[0] < self._lims[1], err_str
        # if part does not need control (static or given values):
        elif 'no_control' in kwargs and kwargs['no_control'] is True:
            # if part is static:
            if 'dm_const' in kwargs:
                # check for correct type:
                err_str = (
                    self._base_err
                    + self._arg_err.format('dm_const')
                    + 'If the part was set to static with `dm_const=X`, X has '
                    'to be either a single float or integer value. To set '
                    'array values over a predefined timespan, use '
                    '`dm_given=value_array` instead.'
                )
                assert isinstance(kwargs['dm_const'], (int, float)), err_str
                self.dm[0] = kwargs['dm_const']
                self._dm_init = kwargs['dm_const']  # backup for resetting
            elif 'time_series' in kwargs:
                # check for correct type:
                err_str = (
                    self._base_err
                    + self._arg_err.format('time_series')
                    + 'If the part is set with predefined values over a '
                    'timespan, `time_series=X` has to be given. `X` has to '
                    'be a Pandas Series with the index column filled with '
                    'timestamps which have to outlast the simulation '
                    'timeframe. The massflow to set has to be given in '
                    'the first column (index 0). To set a constant massflow, '
                    'use `dm_const` instead.'
                )
                assert isinstance(
                    kwargs['time_series'], (pd.Series, pd.DataFrame)
                ), err_str
                assert isinstance(
                    kwargs['time_series'].index, pd.DatetimeIndex
                ), err_str
                assert isinstance(
                    kwargs['time_series'].index.values, np.ndarray
                ), err_str
                assert (
                    kwargs['time_series'].index.values.dtype
                ) == 'datetime64[ns]', err_str
                self.dm_given = True
                self._models.assign_boundary_cond(
                    time_series=kwargs['time_series'],
                    open_port=None,
                    part=self.name,
                    variable_name='dm',
                    array_index=0,
                )
            else:
                # else raise error
                err_str = (
                    self._base_err
                    + self._arg_err.format('dm_const OR dm_given')
                    + 'If `no_control=True` is set, the massflow in [kg/s] has '
                    'either to be given with `dm_const` as a constant '
                    'massflow or with `dm_given` as time dependent '
                    'Pandas Series.'
                )
                assert 'dm_const' in kwargs or 'dm_given' in kwargs, err_str
            self.control_req = False
            self.ctrl_defined = True
        else:
            err_str = (
                'An error during the initialization of '
                + self.name
                + ' occurred! Please check the spelling and type of all '
                'arguments passed to the parts `init_part()`!'
            )

        # construct list of differential input argument names IN THE CORRECT
        # ORDER!!!
        # regex to remove strings: [a-zA-Z_]*[ ]*=self.
        self._input_arg_names_sorted = [
            'ports_all',
            '_port_link_idx',
            'T',
            'res',
            'res_dm',
            'dm',
            'stepnum',
        ]

        # update init status:
        self.initialized = True

    def _reset_to_init_cond(self):
        self.dm[0] = self._dm_init

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

        # get port index slice of target port to create memory view:
        trgt_idx = self._get_topo_cond(port)

        # if part is the starting point of a net (this part as a pump is ALWAYS
        # the starting point of a primary flow net!) OR this part is hitting
        # itself again in the topology (circular net):
        if parent_port is None:
            # for pumps there is no operation id, since they will always
            # be the parent part of the whole net and will thus define the nets
            # massflow, won't have another parent part and won't need any
            # operation routine!
            pass
        elif self.name == kwargs['parent_pump']:
            return ()
        # pump not at start of net:
        else:
            # this will only raise an error and then make the topology analyzer
            # break:
            err_str = (
                'Pump ' + self.name + ' was added to a flow network '
                'where another pump is already existing. There must '
                'not be two pumps in the same flow network!'
            )
            raise TypeError(err_str)

        # set all to fully solved since Pump only has one massflow cell
        #        self._solved_ports = self.port_names[:]
        #        self._cnt_open_prts = 0
        #        # this stays always False for Pump!
        #        self.break_topology = False
        #        # remove part from hydr_comps if completely solved:
        #        if self._cnt_open_prts == 0:
        #            self._models._hydr_comps.remove(self.name)
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
        operation_routine = 'Primary massflow defining part of flow net'
        parent_part = self.name

        # add port dict for current port and fill it:
        if port not in self.info_topology:
            self.info_topology[port] = dict()
        self.info_topology[port].update(
            {
                'Net': net,
                'Massflow': self._dm_io.reshape(-1)[trgt_idx],
                'Calculation routine': operation_routine,
                'Source part': parent_part,
                'Source port(s)': 'No source ports',
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

    def solve(self, timestep):
        """
        Solves the Pump model. That means getting values from connected ports,
        inverting them to pass them on without changing them and saving the
        resulting vector to the temperature array.
        """

        #        self.res_dm[self.stepnum, 0] = self.dm[0]
        #
        #        # also quite fast, like numba
        # #        self.T[:] = self._port_getter(self._models.ports)[::-1]
        #        # numba-version fast speed (57ms)
        #        _get_p_arr_pump(self._models.ports_all, self._port_link_idx, self.T)
        #        self.res[self.stepnum] = self.T
        # medium speed (about 100ms)
        #        self.T[:] = self._port_arrgetter(self._models.ports_all)[::-1]
        # by far the slowest (2x time compared to numba)
        #        self.T = self._models.ports_all[self._models.plinks_arr[self.port_ids]][::-1]

        _pf.solve_pump(
            ports_all=self._models.ports_all,
            port_link_idx=self._port_link_idx,
            T=self.T,
            res=self.res,
            res_dm=self.res_dm,
            dm=self.dm,
            stepnum=self.stepnum,
        )

    def _process_cv(self, ctrl_inst):
        self.dm[:] = (
            ctrl_inst.cv
            if self._llim <= ctrl_inst.cv <= self._ulim
            else self._ulim
            if ctrl_inst.cv > self._ulim
            else self._llim
        )

    def draw_part(self, axis, timestep, draw, animate=False):
        """
        Draws the current part in the plot environment, using vector
        transformation to rotate the part drawing.
        """

        # get part start and end position from plot info dict:
        pos_start = self.info_plot['path'][0]['start_coordinates']
        pos_end = self.info_plot['path'][0]['end_coordinates']
        # get direction vector from info dict:
        vec_dir = self.info_plot['path'][0]['vector']

        # get part rotation angle from the drawing direction vector (vector
        # from part start to part end in drawing):
        rot_angle = self._models._angle_to_x_axis(vec_dir)
        # get length of part:
        vec_len = np.sqrt((vec_dir * vec_dir).sum())
        # vector to line-circle intersection at top and bottom of the drawing
        # (not rotated):
        vec_top = np.array([vec_len / 2, vec_len / 2])
        vec_bot = np.array([vec_len / 2, -vec_len / 2])
        # rotate these vectors:
        vec_top = self._models._rotate_vector(vec_top, rot_angle)
        vec_bot = self._models._rotate_vector(vec_bot, rot_angle)
        # construct top and bottom points:
        pos_top = pos_start + vec_top
        pos_bot = pos_start + vec_bot
        # construct x- and y-grid for lines (from top point to end point to bot
        # point):
        x_grid = np.array([pos_top[0], pos_end[0], pos_bot[0]])
        y_grid = np.array([pos_top[1], pos_end[1], pos_bot[1]])

        # only draw if true
        if draw:
            # construct circle around midpoint of start and pos:
            circ = _plt.Circle(
                tuple((pos_start + np.asarray(pos_end)) / 2),
                radius=np.sqrt((vec_dir * vec_dir).sum()) / 2,
                facecolor='None',
                edgecolor=[0, 0, 0],
                linewidth=self.info_plot['path_linewidth'],
                zorder=5,
                animated=animate,
            )
            # add circle to plot
            axis.add_patch(circ)
            # add lines to plot
            axis.plot(
                x_grid,
                y_grid,
                color=[0, 0, 0],
                linewidth=self.info_plot['path_linewidth'],
                zorder=5,
                animated=animate,
            )
            # construct name and massflow string:
            txt = (r'{0} \n$\dot{{m}} = $ {1:.3f} $\,$kg/s').format(
                self.name, np.round(self.res_dm[timestep][0], 3)
            )

            # construct name and massflow string constructor for animation
            if animate:
                txt_constr = (
                    self.name + r'\n$\dot{{m}} = $' + r'{0:6.3f}$\,$kg/s'
                )
                # get view to part of array where to get text from:
                arr_view = self.res_dm[:, 0:1]
            # get offset vector depending on rotation of pump to deal with
            # none-quadratic form of textbox to avoid overlapping. only in the
            # range of +/-45° of pos. and neg. x-axis an offset vec length of
            # -20 is allowed, else -30:
            offset = (
                -20
                if (
                    0 <= rot_angle <= 45 / 180 * np.pi
                    or 135 / 180 * np.pi <= rot_angle <= 225 / 180 * np.pi
                    or rot_angle >= 315 / 180 * np.pi
                )
                else -40
            )
            # get text offset from bottom point of pump by vector rotation:
            txt_offset = tuple(
                self._models._rotate_vector(np.array([0, offset]), rot_angle)
            )
            ann = axis.annotate(
                txt,
                xy=(pos_bot),
                xytext=txt_offset,
                textcoords='offset points',
                ha='center',
                va='center',
                animated=animate,
            )

            if animate:
                ann.set_animated(True)
                return [[ann, txt_constr, arr_view]]
