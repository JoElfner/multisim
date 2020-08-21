# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:08:14 2017

@author: elfner
"""

import numpy as np
import pandas as pd

from .. import simenv as _smnv
from .. import precomp_funs as _pf


class MixingValve(_smnv.Models):
    """
    type: MixingValve class.
    The MixingValve **mixes or separates** a flow. The flow on the 2-end-side
    is mixed/separated by the factors n1 and n2, with **n1 + n1 = 1** and
    **n1 >= 0** and **n2 >= 0**.
    When mixing the temperatures and mass flows of the respective streams are
    mixed by the rule of *dm_out = dm_in1 + dm_in2*.
    When separating one stream is separated into two streams with the
    same temperature and the massflows *dm_in = n1*dm_out1 + n2*dm_out2*.
    The resulting flow of mixing/separating is calculated after each timestep
    and intermediate step depending on the given control algorithm and the
    measured values in the specified measuring port.

    The MixingValve class does not contain a differential method as it only
    passes the values of the part connected to its 'in'-port(s) to its
    'out'-port(s) and the values of the part connected to its 'out'-port(s) to
    its 'in'-port(s) and only applying the mixing/separating. Thus it is
    not involved in solving the equations using the specified solver algorithm.

    Parameters:
    -----------
    name: string
        Name of the part.
    mix_or_sep: string, default: 'mix'
        Specifies if the MixingValve is supposed to mix or separate strings.
        It can be set to 'mix' for mixing or 'sep' for separating. When 'mix'
        is set, there are two inlet ports 'in1' and 'in1' and one outlet port
        'out' which have to be connected. When 'sep' is set there is one inlet
        port 'in1' two outlet ports 'out1' and 'out2' which have to be
        connected.
    """

    def __init__(self, name, master_cls, mix_or_split='mix', **kwargs):
        self._models = master_cls

        self.constr_type = 'Valve_3w'  # define construction type
        base_err = (  # define leading base error message
            'While adding {0} `{1}` to the simulation '
            'environment, the following error occurred:\n'
        ).format(self.constr_type, str(name))
        arg_err = (  # define leading error for missing/incorrect argument
            'Missing argument or incorrect type/value: {0}\n\n'
        )
        self._base_err = base_err  # save to self to access it in methods
        self._arg_err = arg_err  # save to self to access it in methods

        self.name = name
        self._unit = '[%]'  # unit of the actuator
        self.part_id = self._models.num_parts - 1

        self.kind = mix_or_split
        # save smallest possible float number for avoiding 0-division:
        self._tiny = self._models._tiny

        # even though this part is not using numeric solving, number of
        # gridpoints are specified anyways:
        self.num_gp = 3
        # preallocate grids:
        self.T = np.zeros(3, dtype=np.float64)
        self._T_init = np.zeros_like(self.T)  # init temp for resetting env.
        # preallocate T ports array (here only used for dimension checking)
        self._T_port = np.zeros_like(self.T)
        self.dm = np.zeros(3)
        #        self.U = np.zeros(3)
        # preallocate grids for port connection parameters
        # cross section area of wall of connected pipe, fluid cross section
        # area of, gridspacing and lambda of wall of connected pipe
        self._A_wll_conn_p = np.zeros_like(self._T_port)
        self._A_fld_conn_p = np.zeros_like(self._T_port)
        self._port_gsp = np.full_like(self._T_port, self._tiny)
        self._lam_wll_conn_p = np.full_like(self._T_port, self._tiny)
        self._lam_port_fld = np.full_like(self._T_port, self._tiny)

        # port_definition  (first, second and last array element):
        self.port_num = 3
        # Index to own value array to get values of own ports, meaning if I
        # index a FLATTENED self.T.flat with self._port_own_idx, I need to
        # get values accoring to the order given in self.port_names.
        # That is, this must yield the value of the cell in self.T, which is
        # belonging to the port 'in':
        # self.T.flat[self._port_own_idx[self.port_names.index('in')]]
        self._port_own_idx = np.array(
            (0, 1, self.T.shape[0] - 1), dtype=np.int32
        )
        self._port_own_idx_2D = self._port_own_idx  # save for compatibility
        """port_array"""
        self.port_ids = np.array((), dtype=np.int32)
        # set to read-only to avoid manipulation, same for port_name by using
        # tuple:
        #        self._port_own_idx.flags.writeable = False
        # preallocate port values to avoid allocating in loop:
        self._port_vals = np.zeros(self.port_num)
        # preallocate list to mark ports which have already been solved in
        # topology (to enable creating subnets)
        self._solved_ports = list()
        # port setup depending on mixer or separator valve
        # mixing or separating factors for each port are saved in the dict
        # port_factors, with the factor 1 being a tuple (can't be changed!):
        if mix_or_split == 'mix':
            self.port_names = tuple(('A', 'B', 'AB'))
            # set massflow characteristics for ports: in means that an
            # inflowing massflow has a positive sign, out means that an
            # outflowing massflow is pos.
            self.dm_char = tuple(('in', 'in', 'out'))
            self.pf_arr = np.array(
                [0.5, 0.5, 1], dtype=np.float64  # port in1  # port in2
            )  # port out
        elif mix_or_split == 'split':
            self.port_names = tuple(('A', 'B', 'AB'))
            # set massflow characteristics for ports: in means that an
            # inflowing massflow has a positive sign, out means that an
            # outflowing massflow is pos.
            self.dm_char = tuple(('out', 'out', 'in'))
            self.pf_arr = np.array(
                [0.5, 0.5, 1], dtype=np.float64  # port out1  # port out2
            )  # port in
        else:
            err_str = 'mix_or_split has to be set to \'mix\' or\'split\'!'
            raise ValueError(err_str)
        # make dict for easy lookup of portfactors with memory views:
        self.port_factors = dict(
            {
                'A': self.pf_arr[0:1],
                'B': self.pf_arr[1:2],
                'AB': self.pf_arr[2:3],
            }
        )
        # construct partname+portname to get fast access to own ports:
        dummy_var = list(self.port_names)
        for i in range(self.port_num):
            dummy_var[i] = self.name + ';' + dummy_var[i]
        self._own_ports = tuple(dummy_var)

        # preallocate result grids with one row. An estimate of total rows will
        # be preallocated before simulation start in initialize_sim. massflow
        # grid is preallocated in set_initial_cond:
        self.res = np.zeros((1, self.port_num))
        self.res_dm = np.zeros((2, self.port_num))

        # set if type has to be solved numeric:
        self.solve_numeric = False
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
        self.affect_dm = True
        # if the massflow (dm) has the same value in all cells of the part
        # (respectively in each flow channel for parts with multiple flows):
        self.dm_invariant = False
        # if the part has multiple separated flow channels which do NOT mix
        # (like a heat exchanger for exampe):
        self.multiple_flows = False
        # bool checker if flows were updated in update_flownet to avoid
        # processing flows in get_diff each time (array for referencing):
        self._process_flows = np.array([True])
        # if the part CAN BE controlled by the control algorithm:
        self.is_actuator = True
        self._actuator_CV = self.pf_arr[:]  # set array to be controlled
        self._actuator_CV_name = 'port_opening'
        # if the part HAS TO BE controlled by the control algorithm:
        self.control_req = True
        # if the part needs a special control algorithm (for parts with 2 or
        # more controllable inlets/outlets/...):
        self.actuator_special = True
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
            # if there is the same massflow everywhere in the part
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

    def init_part(self, *, port_A_init, **kwargs):
        """
        Initialize 3-way valve with specifications, material and initial
        conditions. Initial condition for a 3-way valve is the relative port
        opening of port A in values from 0...1.
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

        # delete used kwargs:
        del kwargs['material']
        del kwargs['pipe_specs']

        # assert in and out values
        err_str = (
            'Only values `0 <= port_A_init <= 1` are allowed as '
            'initial values for mixing or splitting valves!'
        )
        assert 0 <= port_A_init <= 1, err_str
        # set starting values to port factors:
        if self.kind == 'mix':
            self.pf_arr[0] = port_A_init
            self.pf_arr[1] = 1 - port_A_init
        else:
            #            self.pf_arr[1] = port_A_init
            #            self.pf_arr[2] = 1 - port_A_init
            """ TODO: change to same idx mix split"""
            self.pf_arr[0] = port_A_init
            self.pf_arr[1] = 1 - port_A_init
        self._pf_init = port_A_init  # backup for resetting

        #        # if set to steady state:
        #        if kwargs:
        #            if 'set_steadystate' in kwargs:
        #                assert (type(kwargs['set_steadystate']) ==
        #                        bool), ('\'set_steadystate\' can only be True or '
        #                                'False!')
        #                self.ctrl_defined = kwargs['set_steadystate']

        # if valve has to be controlled (default) and thus is NOT set to
        # static, it needs a lower and upper limit for the values to set:
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
            self._lims = np.array(  # set limits to array
                [kwargs['lower_limit'], kwargs['upper_limit']],
                dtype=np.float64,
            )
            self._llim = self._lims[0]  # also save to single floats
            self._ulim = self._lims[1]  # also save to single floats
            assert 0 <= self._lims[0] < self._lims[1] <= 1, (
                err_str + ' For Valve_3w limits are additionally restricted '
                'to `0 <= lower_limit < upper_limit <= 1`.'
            )
        # if part does not need control (static or given values):
        elif 'no_control' in kwargs and kwargs['no_control'] is True:
            # if part is static:
            if 'const_val' in kwargs:
                # check for correct type:
                err_str = (
                    'If valve ' + self.name + ' is set to static with '
                    '`const_val=array`, array has to be a 1D numpy '
                    'array with 2 values! To set array values over '
                    'a predefined timespan, use `val_given=time_array` '
                    'instead!'
                )
                assert type(kwargs['const_val']) == np.ndarray and kwargs[
                    'const_val'
                ].shape == (2,), err_str
                self.pfarr[0:2] = kwargs['const_val']
                raise ValueError('with const val reset to init not working')
                # delete used kwargs to enable checking at the end:
                del kwargs['const_val']
            elif 'val_given' in kwargs:
                # check for correct type:
                err_str = (
                    'If valve ' + self.name + ' is set with predefined '
                    'values over a timespan, `val_given=time_array` '
                    'has to be given! `time_array` has to be a Pandas '
                    'Series with the index column filled with '
                    'timestamps which have to outlast the simulation '
                    'timeframe! The valve setting to set has to be '
                    'given in the first column (index 0) for branch A '
                    'and in the second column (index 1) for branch B. '
                    'To set a constant valve opening, use `const_val` '
                    'instead!'
                )
                err_str = (
                    'A check for pandas series needs to be here,'
                    'also checking the timestamp! The check for the '
                    'correct duration of the timestamp needs to be '
                    'done during sim init!'
                )
                assert (
                    type(kwargs['val_given']) == pd.core.series.Series
                ), err_str
                raise TypeError('Timeindex etc. not yet defined!!!')
                # delete used kwargs to enable checking at the end:
                del kwargs['val_given']
                self.val_given = True
                self.control_req = False
                self.ctrl_defined = True
            else:
                err_str = (
                    'If `no_control=True` is defined for valve '
                    + self.name
                    + ', the valve opening has either to be'
                    ' given with `const_val` as a constant opening or '
                    'with `val_given` as time dependent Panda Series!'
                )
                assert (
                    'const_val' not in kwargs and 'val_given' not in kwargs
                ), err_str
        else:
            err_str = (
                'An error during the initialization of '
                + self.name
                + ' occurred! Please check the spelling and type of all '
                'arguments passed to the parts `set_initial_cond()`!'
            )

        # construct list of differential input argument names IN THE CORRECT
        # ORDER!!!
        # regex to remove strings: [a-zA-Z_]*[ ]*=self.
        self._input_arg_names_sorted = [
            'ports_all',
            '_port_link_idx',
            '_dm_io',
            'T',
        ]

        # update init status:
        self.initialized = True

    def _reset_to_init_cond(self):
        # set starting values to port factors:
        if self.kind == 'mix':
            self.pf_arr[0] = self._pf_init
            self.pf_arr[1] = 1 - self._pf_init
        else:
            #            self.pf_arr[1] = port_A_init
            #            self.pf_arr[2] = 1 - port_A_init
            """ TODO: change to same idx mix split"""
            self.pf_arr[0] = self._pf_init
            self.pf_arr[1] = 1 - self._pf_init

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

        # 3wValve, no ports solved yet
        if self._cnt_open_prts == 3:
            # The following connection requirement(s) have to be checked:
            # 1: all ports (A, B and AB) of a mixing valve MUST NOT be on
            #    the pressure side of a pump.
            # 2: entry ports (A and B) of a mixing valve MUST NOT be on the
            #    suction side of a pump. This means a mixing valve can only
            #    be solved coming from port AB.
            # 3: all ports (A, B and AB) of a splitting valve MUST NOT be
            #    on the suction side of a pump.
            # 4: exit ports (A and B) of a splitting valve MUST NOT be on
            #    the pressure side of a pump. This means a splitting valve
            #    can only be solved coming from port AB.
            # 5: two parts of the non numeric solving kind MUST NOT be
            #    connected directly. At least one numeric part has to be in
            #    between.
            # check connection requirement(s):
            # prepare error strings:
            err_str1 = (
                'Part ' + self.name + ' is directly connected to '
                'the pressure side of a pump. Mixing valves may '
                'only be connected to the suction side of a pump '
                'with port AB!'
            )
            err_str2 = (
                'Part ' + self.name + ' is connected to the '
                'suction side of a pump with port A or B. '
                'Mixing valves may only be connected to the '
                'suction side of a pump with port AB!'
            )
            err_str3 = (
                'Part ' + self.name + ' is directly connected to the '
                'suction side of a pump. Splitting valves may only be '
                'connected to the pressure side of a pump with port '
                'AB!'
            )
            err_str4 = (
                'Part ' + self.name + ' is connected to the '
                'pressure side of a pump with port A or B. '
                'Splitting valves may only be connected to the '
                'suction side of a pump with port AB!'
            )
            if self.kind == 'mix':
                # assert condition 1:
                assert kwargs['pump_side'] != 'pressure', err_str1
                # assert condition 2:
                assert port == 'AB', err_str2
            else:
                # assert condition 3:
                assert kwargs['pump_side'] != 'suction', err_str3
                # assert condition 4:
                assert port == 'AB', err_str4
            # assert condition 5:
            err_str5 = (
                'Non numeric Part ' + self.name + ' is connected to '
                'non numeric part ' + src_part + '. Two non '
                'numeric parts must not be connected directly! '
                'Insert a numeric part in between to set up a '
                'correct topology!'
            )
            assert self._models.parts[src_part].solve_numeric, err_str5

            # if valve is getting the massflow from another part (then port
            # AB is solved as the first port), it can simply be copied
            # from it: operation id 0 (positive) or - 1 (negative)
            if alg_sign == 'positive':
                operation_id = 0
            else:
                operation_id = -1

            # add operation instructions to tuple (memory view to target
            # massflow array cell, operation id and memory view source port's
            # massflow array cells)
            op_routine = tuple()
            # construct memory view to target massflow array cell and append to
            # op routine tuple
            op_routine += (self._dm_io.reshape(-1)[trgt_idx],)
            # add operation id:
            op_routine += (operation_id,)
            # add memory view to source massflow array cell:
            op_routine += (
                self._models.parts[src_part]._dm_io.reshape(-1)[src_idx],
            )

        else:
            # get massflow calculation routine for the case that port
            # A or B needs to be solved using the massflow from port AB
            # and valve opening (stored in port factors array).
            # operation id of a 3w valve for this case is ALWAYS 3, since
            # AB must be given and A or B can be calculated by multiplying
            # the respective port opening factor with AB. no negative
            # of product needed, since AB positive massflow sign is
            # contrary to A and B
            operation_id = 3
            # get source port index and create memory view to it:
            src1_idx_start = self._port_own_idx[self.port_names.index('AB')]
            src1_idx = slice(src1_idx_start, src1_idx_start + 1)
            # second source "port" index is the index to the port factor
            # array cell of port:
            src2_idx_start = self.port_names.index(port)
            src2_idx = slice(src2_idx_start, src2_idx_start + 1)
            # add operation instructions to tuple (memory view to target
            # massflow array cell, operation id, memory view to the
            # source port's massflow array cell and memory view to the
            # TARGET PORT'S port factor array cell):
            op_routine = (
                self._dm_io.reshape(-1)[trgt_idx],
                operation_id,
                self._dm_io.reshape(-1)[src1_idx],
                self.pf_arr[src2_idx],
            )

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
            else 'Error'
        )
        src_part = src_part if src_part is not None else self.name
        source_ports = (
            tuple(('AB', 'pf_arr[' + port + ']'))
            if operation_id == 3
            else src_port
            if operation_id == 0
            else tuple(set(self.port_names) - set(port))
        )
        # add port dict for current port and fill it:
        if port not in self.info_topology:
            self.info_topology[port] = dict()
        self.info_topology[port].update(
            {
                'Net': net,
                'Massflow': self._dm_io.reshape(-1)[trgt_idx],
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

    def _process_cv(self, ctrl_inst):
        # 3w_valve_direct!
        # n1 value (port A) with clipping to ]llim,ulim[:
        self.pf_arr[0] = (
            self._llim
            if ctrl_inst.cv < self._llim
            else self._ulim
            if ctrl_inst.cv > self._ulim
            else ctrl_inst.cv
        )
        # n2 value (port B):
        self.pf_arr[1] = 1 - self.pf_arr[0]

    def solve(self, timestep):
        """
        Mixing Valve solve method:
        --------------------------
            The mass flow averaged mean of the values of the other parts ports
            connected to the 'in1' and 'in2' ports is passed to the 'out'
            port, taking the arithmetic mean of the in-ports temperatures to
            get approximate material properties at the out port. the value of
            the 'out' port is passed to 'in1' and 'in2' unchanged.
            This is approximately correct, as the temperature values affect the
            heat conduction and in-/ or outflowing massflow of connected parts
            while the valve part itself is approximated as infinitely small
            containing no mass.

        Splitting Valve solve method:
        -----------------------------
            The arithmetic mean of the values of the other parts ports
            connected to the 'out1' and 'out2' ports is passed to the 'in'
            port, while the value of the 'in' port is passed to 'out1' and
            'out2' unchanged.
            This is approximately correct, as the temperature values affect the
            heat conduction and in-/ or outflowing massflow of connected parts
            while the valve part itself is approximated as infinitely small
            containing no mass.
        """

        # save massflow to results grid:
        self.res_dm[self.stepnum] = self._dm_io

        # get kind and then call numba jitted function (calling numba function
        # which also does the selecting is SLOWER!)
        if self.kind == 'mix':
            # numba compiled function to solve mixing (includes getting ports)
            _pf.solve_mix(
                self._models.ports_all,
                self._port_link_idx,
                self._dm_io,
                self.T,
            )
        else:
            # numba compiled function to solve splitting(incl. getting ports)
            _pf.solve_split(
                self._models.ports_all, self._port_link_idx, self.T
            )
        # copy results to results grid:
        self.res[self.stepnum] = self.T

    def draw_part(self, axis, timestep, draw):
        """
        Draws the current part in the plot environment, using vector
        transformation to rotate the part drawing.
        """

        # get and calculate all the information if not drawing (save all to a
        # hidden dict):
        if not draw:
            # create hidden plot dict:
            __pt = dict()
            # get part start position from plot info dict:
            __pt['pos_start'] = self.info_plot['path'][0]['start_coordinates']
            # assert that orientation is in info dict and correct type:
            orient_list = ['left', 'middle', 'right']
            err_str = (
                'For plotting of 3w-valves the orientation of the '
                'valve must be given! Please pass the orientation as '
                '`orientation=\'string\'` to the 3w-valve\'s '
                '`set_plot_shape()`-method with string being one of '
                'the following: ' + str(orient_list)
            )
            assert 'orientation' in self.info_plot, err_str
            assert self.info_plot['orientation'] in orient_list, err_str
            # get orientation:
            __pt['orient'] = self.info_plot['orientation']

            # get direction vector from info dict:
            __pt['vec_dir'] = self.info_plot['path'][0]['vector']

            # get part rotation angle from the drawing direction vector (vector
            # from part start to part end in drawing):
            __pt['rot_angle'] = self._models._angle_to_x_axis(__pt['vec_dir'])
            # get length of part:
            __pt['vec_len'] = np.sqrt(
                (__pt['vec_dir'] * __pt['vec_dir']).sum()
            )

            # construct all drawing vectors for zero-rotation and one port on
            # the left side, one port on the bottom side and one port on the
            # right side (standard orientation 'left'). all vectors start from
            # the center of the part which is given
            # by the end position of vertex_coords.
            # construct left port (upper vertice and lower vertice):
            __pt['vec_l_u'] = np.array([-1, 0.5]) * __pt['vec_len']
            __pt['vec_l_l'] = np.array([-1, -0.5]) * __pt['vec_len']
            # construct right port (upper vertice and lower vertice):
            __pt['vec_r_u'] = np.array([1, 0.5]) * __pt['vec_len']
            __pt['vec_r_l'] = np.array([1, -0.5]) * __pt['vec_len']
            # construct middle port (left vertice and right vertice):
            __pt['vec_m_l'] = np.array([-0.5, -1]) * __pt['vec_len']
            __pt['vec_m_r'] = np.array([0.5, -1]) * __pt['vec_len']

            # get rotation angle due to orientation (to x unit vector (1 0)):
            if __pt['orient'] == 'left':
                # standard rotation
                __pt['orient_angle'] = 0
            elif __pt['orient'] == 'right':
                # flipped standard rotation
                __pt['orient_angle'] = 180 / 180 * np.pi
            elif __pt['orient'] == 'middle':
                # middle port on the left
                __pt['orient_angle'] = -90 / 180 * np.pi
            # get total rotation angle:
            __pt['rot_angle'] += __pt['orient_angle']

            # rotate all vectors:
            __pt['vec_l_u'] = self._models._rotate_vector(
                __pt['vec_l_u'], __pt['rot_angle']
            )
            __pt['vec_l_l'] = self._models._rotate_vector(
                __pt['vec_l_l'], __pt['rot_angle']
            )
            __pt['vec_r_u'] = self._models._rotate_vector(
                __pt['vec_r_u'], __pt['rot_angle']
            )
            __pt['vec_r_l'] = self._models._rotate_vector(
                __pt['vec_r_l'], __pt['rot_angle']
            )
            __pt['vec_m_l'] = self._models._rotate_vector(
                __pt['vec_m_l'], __pt['rot_angle']
            )
            __pt['vec_m_r'] = self._models._rotate_vector(
                __pt['vec_m_r'], __pt['rot_angle']
            )
            # construct all points:
            __pt['pos_center'] = __pt['pos_start'] + __pt['vec_dir']
            __pt['pos_l_u'] = __pt['pos_center'] + __pt['vec_l_u']
            __pt['pos_l_l'] = __pt['pos_center'] + __pt['vec_l_l']
            __pt['pos_r_u'] = __pt['pos_center'] + __pt['vec_r_u']
            __pt['pos_r_l'] = __pt['pos_center'] + __pt['vec_r_l']
            __pt['pos_m_l'] = __pt['pos_center'] + __pt['vec_m_l']
            __pt['pos_m_r'] = __pt['pos_center'] + __pt['vec_m_r']
            # construct x- and y-grid for lines (from center to l_u to l_l to
            # r_u to r_l to center to m_l to m_r to center):
            __pt['x_grid'] = np.array(
                [
                    __pt['pos_center'][0],
                    __pt['pos_l_u'][0],
                    __pt['pos_l_l'][0],
                    __pt['pos_r_u'][0],
                    __pt['pos_r_l'][0],
                    __pt['pos_center'][0],
                    __pt['pos_m_l'][0],
                    __pt['pos_m_r'][0],
                    __pt['pos_center'][0],
                ]
            )
            __pt['y_grid'] = np.array(
                [
                    __pt['pos_center'][1],
                    __pt['pos_l_u'][1],
                    __pt['pos_l_l'][1],
                    __pt['pos_r_u'][1],
                    __pt['pos_r_l'][1],
                    __pt['pos_center'][1],
                    __pt['pos_m_l'][1],
                    __pt['pos_m_r'][1],
                    __pt['pos_center'][1],
                ]
            )

            # replace port coordinates since they are wrong for more complex
            # parts:
            if __pt['orient'] == 'left':
                # get middle and right port coordinates:
                __pt['p1_coords'] = (
                    __pt['pos_center']
                    + (__pt['vec_m_l'] + __pt['vec_m_r']) / 2
                )
                __pt['p2_coords'] = (
                    __pt['pos_center']
                    + (__pt['vec_r_u'] + __pt['vec_r_l']) / 2
                )
            elif __pt['orient'] == 'middle':
                # get left and right port coordinates:
                __pt['p1_coords'] = (
                    __pt['pos_center']
                    + (__pt['vec_l_u'] + __pt['vec_l_l']) / 2
                )
                __pt['p2_coords'] = (
                    __pt['pos_center']
                    + (__pt['vec_r_u'] + __pt['vec_r_l']) / 2
                )
            elif __pt['orient'] == 'right':
                # get left and middle port coordinates:
                __pt['p1_coords'] = (
                    __pt['pos_center']
                    + (__pt['vec_l_u'] + __pt['vec_l_l']) / 2
                )
                __pt['p2_coords'] = (
                    __pt['pos_center']
                    + (__pt['vec_m_l'] + __pt['vec_m_r']) / 2
                )
            # get the free ports (the ports where the position is not coming
            # from):
            free_ports = list(self.port_names)
            free_ports.remove(self.info_plot['auto_connection']['own_port'])
            # now get the free ports depending on invert status:
            if 'invert' not in self.info_plot or not self.info_plot['invert']:
                p1 = free_ports[0]
                p2 = free_ports[1]
            elif self.info_plot['invert']:
                p1 = free_ports[1]
                p2 = free_ports[0]
            # set them to the ports:
            self.info_plot[p1]['coordinates'] = __pt['p1_coords']
            self.info_plot[p2]['coordinates'] = __pt['p2_coords']
            # get the connected part;ports:
            #            p1_conn_p = self._models.port_links[self.name + ';' + free_ports[0]]
            #            p2_conn_p = self._models.port_links[self.name + ';' + free_ports[1]]
            #            # split them up:
            #            p1_conn_part, p1_conn_port = p1_conn_p.split(';')
            #            p2_conn_part, p2_conn_port = p2_conn_p.split(';')
            #            # now run their set plot shape with that new information again:
            #            NetPlotter.set_plot_shape(p1_conn_part, p1_conn_port,
            #                                      self._models.parts[p1_conn_part].
            #                                      info_plot['vertex_coordinates'],
            #                                      linewidth=self._models.parts[p1_conn_part].
            #                                      info_plot['path_linewidth'])
            #            NetPlotter.set_plot_shape(p2_conn_part, p2_conn_port,
            #                                      self._models.parts[p2_conn_part].
            #                                      info_plot['vertex_coordinates'],
            #                                      linewidth=self._models.parts[p2_conn_part].
            #                                      info_plot['path_linewidth'])

            # get annotation text properties:
            # get offset vector depending on rotation of pump to deal with
            # none-quadratic form of textbox to avoid overlapping. only in the
            # range of +/-45° of pos. and neg. x-axis an offset vec length of
            # -20 is allowed, else -30:
            offset = (
                20
                if (
                    0 <= __pt['rot_angle'] <= 45 / 180 * np.pi
                    or 135 / 180 * np.pi
                    <= __pt['rot_angle']
                    <= 225 / 180 * np.pi
                    or __pt['rot_angle'] >= 315 / 180 * np.pi
                )
                else 30
            )
            # get text offset from bottom point of pump by vector rotation:
            __pt['txt_offset'] = tuple(
                self._models._rotate_vector(
                    np.array([0, offset]), __pt['rot_angle']
                )
            )
            __pt['txtA_offset'] = tuple(
                self._models._rotate_vector(
                    np.array([0, offset]), __pt['rot_angle']
                )
            )
            __pt['txtB_offset'] = tuple(
                self._models._rotate_vector(
                    np.array([0, offset]), __pt['rot_angle']
                )
            )

            # finally save hidden dict to self:
            self.__pt = __pt

        # only draw if true:
        if draw:
            # add lines to plot
            axis.plot(
                self.__pt['x_grid'],
                self.__pt['y_grid'],
                color=[0, 0, 0],
                linewidth=self.info_plot['path_linewidth'],
                zorder=5,
            )

            # construct name and massflow strings for ports A and B:
            txt = self.name
            txtA = (
                'A'
                + r'\n$\dot{m} = $'
                + str(self.res_dm[timestep, 0])
                + r'$\,$kg/s'
            )
            txtB = (
                'B'
                + r'\n$\dot{m} = $'
                + str(self.res_dm[timestep, 1])
                + r'$\,$kg/s'
            )

            axis.annotate(
                txt,
                xy=(self.__pt['pos_center']),
                xytext=self.__pt['txt_offset'],
                textcoords='offset points',
                ha='center',
                va='center',
            )
            axis.annotate(
                txtA,
                xy=(self.info_plot['A']['coordinates']),
                xytext=self.__pt['txtA_offset'],
                textcoords='offset points',
                ha='center',
                va='center',
            )
            axis.annotate(
                txtB,
                xy=(self.info_plot['B']['coordinates']),
                xytext=self.__pt['txtB_offset'],
                textcoords='offset points',
                ha='center',
                va='center',
            )

        # construct name and massflow string:


#        txt = (self.name + '\n$\dot{m} = $' + str(self.res_dm[timestep][0])
#               + 'kg/s')
# get offset vector depending on rotation of pump to deal with
# none-quadratic form of textbox to avoid overlapping. only in the
# range of +/-45° of pos. and neg. x-axis an offset vec length of -20
# is allowed, else -30:
#        offset = (-20 if (0 <= rot_angle <= 45/180*np.pi
#                          or 135/180*np.pi <= rot_angle <= 225/180*np.pi
#                          or rot_angle >= 315/180*np.pi) else -30)
#        # get text offset from bottom point of pump by vector rotation:
#        txt_offset = tuple(self._models._rotate_vector(np.array([0, offset]),
#                                                 rot_angle))
#        axis.annotate(txt, xy=(pos_bot),
#                      xytext=txt_offset, textcoords='offset points',
#                      ha='center', va='center')
