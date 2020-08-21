# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:04:04 2018

@author: elfner
"""

import numpy as np

from .pipe import Pipe
from ..precomp_funs import pipe1D_branched_diff


class PipeBranched(Pipe):
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
        self.constr_type = 'PipeBranched'  # define construction type
        # since this part is a subclass of Pipe, initialize Pipe:
        # super(PipeBranched, self).__init__(
        #     name, master_cls, **kwargs, constr_type=self.constr_type)
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

        # if ports can be added to this part:
        self.can_add_ports = True

        # add ports
        self._add_ports(**kwargs)

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
        self.affect_dm = False
        # if the massflow (dm) has the same value in all cells of the part
        # (respectively in each flow channel for parts with multiple flows):
        self.dm_invariant = False

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
        # super(PipeBranched, self).init_part(**kwargs)
        super().init_part(**kwargs)

        # add massflow grid argument to input args at correct position:
        self._input_arg_names_sorted.insert(6, 'dm')

    def _get_flow_routine(  # overwrite pipe routine, since new ports exist
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

        # if more than one port is open to be solved, it will always be solved
        # by passing on the value from the connected parent part
        if self._cnt_open_prts != 1:
            # get topology connection conditions (target index, source
            # part/port identifiers, source index and algebraic sign for passed
            # massflow):
            (
                trgt_idx,
                src_part,
                src_port,
                src_idx,
                alg_sign,
            ) = self._get_topo_cond(port, parent_port)
            # the operation id is always 0 or -1, depending on the
            # algebraic sign of the ports which are connected, when there is
            # more than one port remaining to be solved. This means the value
            # of connected port is either passed on (0) or the negative value
            # of it is used (-1):
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
        # only one port remaining to be solved:
        else:
            # There are two possibilities to get the last port:
            # 1. - Get it from a connected port -> parent_port must be given
            # 2. - Calculate it from the missing massflow through the other
            #      already solved own ports. - > no parent_port needed
            # Method 1 is highly preferred, since it only involves accessing
            # the connected port's massflow array cell, while method 2 requires
            # summing up all other ports.
            if parent_port is not None:
                # Method 1!
                # get topology connection conditions (target index, source
                # part/port identifiers, source index and algebraic sign for
                # passed massflow):
                (
                    trgt_idx,
                    src_part,
                    src_port,
                    src_idx,
                    alg_sign,
                ) = self._get_topo_cond(port, parent_port)
                # passing values is always 0 or -1, depending on pos.
                # massflow directions of connected ports
                if alg_sign == 'positive':
                    # in/out connected
                    operation_id = 0
                else:
                    # in/in or out/out connected
                    operation_id = -1
                # add operation instructions to tuple (memory view to target
                # massflow array cell, operation id and memory view source
                # port's massflow array cells)
                op_routine = (
                    self._dm_io.reshape(-1)[trgt_idx],
                    operation_id,
                    self._models.parts[src_part]._dm_io.reshape(-1)[src_idx],
                )
            else:
                # Method 2!
                # get port index slice of target port to create memory view:
                trgt_idx = self._get_topo_cond(port)
                # operation id for this case is always -1, since the positive
                # massflow direction of all ports is facing inwards
                operation_id = -1
                # add operation instructions to tuple (memory view to target
                # massflow array cell, operation id and memory view to ALL
                # source/other port's massflow array cells of the TES)
                op_routine = (
                    self._dm_io.reshape(-1)[trgt_idx],
                    operation_id,
                )
                # add memory view to source massflow array cell(s)
                # therefore loop over already solved ports
                for slvd_prt in self._solved_ports:
                    # get solved port index as slice to create memory views
                    slvd_prt_idx = self._get_topo_cond(slvd_prt)
                    # add memory view to this ports massflow
                    op_routine += (self._dm_io.reshape(-1)[slvd_prt_idx],)

        # update solved ports list and counter stop break:
        self._solved_ports.append(port)
        self._cnt_open_prts = self.port_num - len(self._solved_ports)
        # set break topology to False if enough ports have been solved:
        self.break_topology = True if self._cnt_open_prts > 0 else False
        # remove part from hydr_comps if completely solved:
        if self._cnt_open_prts == 0:
            self._models._hydr_comps.remove(self.name)

        # save topology parameters to dict for easy information lookup:
        net = 'Subnet' if subnet else 'Flownet'
        operation_routine = (
            'Negative of sum'
            if operation_id == -1
            else 'Pass on value'
            if operation_id == 0
            else 'Error'
        )
        src_part = src_part if self._cnt_open_prts > 1 else self.name
        source_ports = (
            parent_port
            if operation_id == 0
            else tuple(set(self.port_names) - set(port))
        )
        kwargs['pump_side'] = (
            'decoupled from pump by hydraulic compensator'
            if kwargs['pump_side'] == 'sub net - undefined'
            else kwargs['pump_side']
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

    def get_diff(self, timestep):
        """
        This function just calls a jitted calculation function.

        """

        pipe1D_branched_diff(*self._input_args, timestep)

        return self.dT_total
