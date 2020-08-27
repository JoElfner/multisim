# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Aug 2018
"""

import numpy as np

from .pipe import Pipe
from ..precomp_funs import pipe1D_branched_diff


class PipeWith3wValve(Pipe):
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
        self.constr_type = 'PipeWith3wValve'  # define construction type
        # since this part is a subclass of Pipe, initialize Pipe:
        super().__init__(
            name, master_cls, **kwargs, constr_type=self.constr_type
        )

        # preallocate mass flow grids:
        self.dm = np.zeros_like(self.T)
        self._dm_top = np.zeros_like(self.T)
        self._dm_bot = np.zeros_like(self.T)
        # set dm char so that all ports face inwards (more convenient when
        # adding new ports):
        self.dm_char = tuple(('in', 'in'))

        # add valve to specified location in pipe:
        err_str = (
            self._base_err
            + self._arg_err.format('valve_location')
            + 'The location of the 3 way valve in the pipe has to be given with '
            '`valve_location=X`, where X is an integer cell index in the '
            'range of the shape of the pipe.\n'
            'The valve location index specifies the position of the B-port of '
            'the valve, while the A- and AB-port are specified by the pipe\'s '
            'add part algorithm.'
        )
        assert (
            'valve_location' in kwargs
            and isinstance(kwargs['valve_location'], int)
            and 0 <= kwargs['valve_location'] < self.num_gp
        ), err_str
        self._valve_B_loc = kwargs['valve_location']
        # if ports can be added to this part. set to true for one single port:
        self.can_add_ports = True
        # add B port:
        kwargs['new_ports'] = {'B': [self._valve_B_loc, 'index']}
        self._add_ports(**kwargs)
        self.can_add_ports = False  # disable adding new ports

        # give specific port names:
        self.port_names = tuple(('A', 'B', 'AB'))
        # set massflow characteristics for ports: in means that an
        # inflowing massflow has a positive sign, out means that an
        # outflowing massflow is pos.
        # self.dm_char = tuple(('in', 'in', 'out'))
        # replaced by all-inflowing ports to deal with all kinds of port setups
        self.dm_char = tuple(('in', 'in', 'in'))

        # preallocate massflow calculation factor array:
        self.pf_arr = np.array(
            [0.5, 0.5, 1], dtype=np.float64  # port A  # port B
        )  # port AB
        # make dict for easy lookup of portfactors with memory views:
        self.port_factors = dict(
            {
                'A': self.pf_arr[0:1],
                'B': self.pf_arr[1:2],
                'AB': self.pf_arr[2:3],
            }
        )

        # construct all port shape dependent vars:
        dummy_var = list(self.port_names)
        for i in range(self.port_num):
            dummy_var[i] = self.name + ';' + dummy_var[i]
        self._own_ports = tuple(dummy_var)
        # preallocate port values:
        self._port_vals = np.zeros(self.port_num)
        # preallocate grids for port connection parameters:
        # cross section area of wall of connected pipe, fluid cross section
        # area of, gridspacing and lambda of wall of connected pipe
        Tpshp = self._T_port.shape
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

        # if the topology construction method has to stop when it reaches the
        # part to solve more ports from other sides before completely solving
        # the massflow of it. This will be set to false as soon as only one
        # port to solve is remaining:
        self.break_topology = True
        # count how many ports are still open to be solved by topology. If
        # break topology is True, this is used to set it to False if 1 is
        # reached.
        self._cnt_open_prts = self.port_num
        # determine if part has the capability to affect massflow (dm) by
        # diverting flow through ports or adding flow through ports:
        self.affect_dm = True
        # if the massflow (dm) has the same value in all cells of the part
        # (respectively in each flow channel for parts with multiple flows):
        self.dm_invariant = False

        # if the part CAN BE controlled by the control algorithm:
        self.is_actuator = True
        self._actuator_CV = self.pf_arr[:]  # set array to be controlled
        self._actuator_CV_name = 'port_opening'
        self._unit = '[%]'
        # if the part HAS TO BE controlled by the control algorithm:
        self.control_req = True
        # if the part needs a special control algorithm (for parts with 2 or
        # more controllable inlets/outlets/...):
        self.actuator_special = True
        # initialize bool if control specified:
        self.ctrl_defined = False

        # IMPORTANT: THIS VARIABLE **MUST NOT BE INHERITED BY SUB-CLASSES**!!
        # If sub-classes are inherited from this part, this bool checker AND
        # the following variables MUST BE OVERWRITTEN!
        # ist the diff function fully njitted AND are all input-variables
        # stored in a container?
        self._diff_fully_njit = False
        # self._diff_njit = pipe1D_diff  # handle to njitted diff function
        # input args are created in simenv _create_diff_inputs method

    def init_part(self, start_portA_opening, **kwargs):
        # since this part is a subclass of Pipe, call init_part of Pipe:
        super().init_part(**kwargs)

        # set starting valve opening:
        err_str = (
            self._base_err
            + self._arg_err.format('start_portA_opening')
            + 'The initial valve port A opening has to be set in the range of '
            '`0 <= start_portA_opening <= 1`.'
        )
        assert (
            isinstance(start_portA_opening, (int, float))
            and 0 <= start_portA_opening <= 1
        ), err_str
        self.pf_arr[0] = start_portA_opening
        self.pf_arr[1] = 1 - self.pf_arr[0]
        self._pf_arr_init = self.pf_arr.copy()  # bkp for re-initializing

        # initialize the actuator
        self._initialize_actuator(**kwargs)

        # expand const var to other ports:
        self._actuator_CV[1] = 1 - self._actuator_CV[0]

        # add massflow grid argument to input args at correct position:
        self._input_arg_names_sorted.insert(6, 'dm')

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

        # 3wValve, no ports solved yet or massflow given from other part
        if self._cnt_open_prts == 3:
            # if valve is getting the massflow from another part, it can simply
            # be copied from it: operation id 0 (positive) or - 1 (negative)
            if alg_sign == 'positive':
                operation_id = 0
            else:
                operation_id = -1
            # add operation instructions to tuple (memory view to target
            # massflow array cell, operation id and memory view source port's
            # massflow array cells)
            op_routine = (
                self._dm_io.reshape(-1)[trgt_idx],
                operation_id,
                self._models.parts[src_part]._dm_io.reshape(-1)[src_idx],
            )
        else:
            # get massflow calculation routine for the case that port
            # A or B need to be solved using the massflow from port AB
            # and valve opening (stored in port factors array).
            # operation id of a 3w valve for this case is ALWAYS -3, since
            # AB must be given and A or B can be calculated by multiplying
            # the respective port opening factor with AB. (no )negative
            # of product needed, since AB positive massflow sign is
            # not contrary to A and B
            if port in ('A', 'B') and 'AB' in self._solved_ports:
                operation_id = -3  # before: 3
                # get source index for massflow cell of port AB. If AB is
                # already solved, this will always be the third (last) cell of
                # dm_io:
                src_idx_ab = slice(2, 3)
                # add operation instructions to tuple (memory view to target
                # massflow array cell, operation id, memory view to the
                # source port's massflow array cell and memory view to the
                # TARGET PORT'S port factor array cell):
                op_routine = (
                    self._dm_io.reshape(-1)[trgt_idx],
                    operation_id,
                    self._dm_io.reshape(-1)[src_idx_ab],
                    self.pf_arr[trgt_idx],
                )
            elif port == 'AB' and (
                'A' in self._solved_ports or 'B' in self._solved_ports
            ):
                # if the requested port is AB AND either A OR B have already
                # been solved.
                # operation ID is now -4 --> negative division of the third
                # op_routine element by the fourth:
                # (op_routine[3]/op_routine[4])
                operation_id = -4
                # get solved port (start with looking for A):
                if 'A' in self._solved_ports:
                    src_idx = slice(0, 1)  # port A src index is always cell 0
                else:  # elif 'B' is in solved ports:
                    # port B src index is always cell 1 (CAUTION: This is only
                    # true for the dm_io and pf_arr arrays, NOT for temp.!)
                    src_idx = slice(1, 2)
                # add operation instructions to tuple (memory view to target
                # massflow array cell, operation id, memory view to the
                # source port's massflow array cell and memory view to the
                # TARGET PORT'S port factor array cell):
                op_routine = (
                    self._dm_io.reshape(-1)[trgt_idx],
                    operation_id,
                    self._dm_io.reshape(-1)[src_idx],
                    self.pf_arr[src_idx],
                )
            elif port in ('A', 'B') and 'AB' not in self._solved_ports:
                # this can only be solved by multiplying the other port (A if
                # port=B, else vice-versa) with the target port port factor
                # and dividing by the source port port factor. -> ID 5
                operation_id = 5
                # get solved port (start with looking for A):
                if 'A' in self._solved_ports:
                    src_idx = slice(0, 1)  # port A src index is always cell 0
                else:  # elif 'B' is in solved ports:
                    # port B src index is always cell 1 (CAUTION: This is only
                    # true for the dm_io and pf_arr arrays, NOT for temp.!)
                    src_idx = slice(1, 2)
                # add operation instructions to tuple (memory view to target
                # massflow array cell, operation id, memory view to the
                # source port's massflow array cell, memory view to the
                # TARGET PORT'S port factor array cell AND memory view to the
                # source port's port factor array cell):
                # resulting calculation routine:
                # target = source * trgt_pf / src_pf
                op_routine = (
                    self._dm_io.reshape(-1)[trgt_idx],
                    operation_id,
                    self._dm_io.reshape(-1)[src_idx],
                    self.pf_arr[trgt_idx],
                    self.pf_arr[src_idx],
                )
        # =============================================================================
        #                 raise NotImplementedError(
        #                     'Port {0} is to be solved, with port AB not yet being'
        #                     'solved. This requires multiple calculations, for example '
        #                     '*pf_arr[0] / pf_arr[1] if port==B, which is currently '
        #                     'not supported.'.format(port))
        # =============================================================================

        # update solved ports list and counter stop break:
        self._solved_ports.append(port)
        self._cnt_open_prts = self.port_num - len(self._solved_ports)
        # update break topology:
        #        self.break_topology = True if self._cnt_open_prts > 0 else False
        self.break_topology = False
        # remove part from hydr_comps if completely solved:
        if self._cnt_open_prts == 0:
            self._models._hydr_comps.remove(self.name)

        # save topology parameters to dict for easy information lookup:
        net = 'Subnet' if subnet else 'Flownet'
        operation_routine = (
            'Negative (of sum) of source'
            if operation_id == -1
            else 'Sum'
            if operation_id == 1
            else 'Pass on value'
            if operation_id == 0
            else 'Multiplication with port factor'
            if operation_id == 3
            else 'Division by port factor'
            if operation_id == 4
            else 'Mult/Div with other ports'
            if operation_id == 5
            else 'Error'
        )
        src_part = src_part if src_part is not None else self.name
        source_ports = (
            tuple(('AB', 'pf_arr[' + port + ']'))
            if operation_id == 3
            else src_port
            if operation_id == 0
            else src_port
            if operation_id == -1
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
        # 3w_valve_direct control update method.
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

    def get_diff(self, timestep):
        """
        This function just calls a jitted calculation function. For a pipe
        with a valve this is the same as the branched pipe's differential
        function.

        """

        pipe1D_branched_diff(*self._input_args, timestep)

        return self.dT_total
