# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:30:20 2017

@author: elfner
"""

import numpy as np

from .. import simenv as _smnv
from ..precomp_funs import solve_connector_3w  # , get_cp_water


class Connector3w(_smnv.Models):
    """
    type: Connector3w class.
    The Connector3w **mixes or separates** a flow, depending on the pumps of
    the connected flow nets. In contrast to the Valve3w class, the Connector3w
    class has no measure to influence the massflow through its ports, thus it
    will be solved in sub flow nets after the primary flow nets.
    The resulting flow of mixing/separating is calculated after each timestep
    and intermediate step depending on the timestepping of the used solver
    algorithm.

    The Connector3w class does not contain a differential method as it only
    hands over the values of the parts connected to its ports, named
    **A, B and C**, while applying the mixing/separating. Thus it is not
    involved in solving the equations using the specified solver algorithm.

    Parameters:
    -----------
    name: string
        Name of the part.

    """

    def __init__(self, name, master_cls, **kwargs):
        self._models = master_cls

        self.constr_type = 'Connector_3w'  # define construction type
        base_err = (  # define leading base error message
            'While adding {0} `{1}` to the simulation '
            'environment, the following error occurred:\n'
        ).format(self.constr_type, str(name))
        arg_err = (  # define leading error for missing/incorrect argument
            'Missing argument or incorrect type/value: {0}\n\n'
        )
        self._base_err = base_err  # save to self to access it in methods
        self._arg_err = arg_err  # save to self to access it in methods

        #        super().__init__()
        self.name = name
        self.part_id = self._models.num_parts - 1
        # save smallest possible float number for avoiding 0-division:
        self._tiny = self._models._tiny

        # even though this part is not using numeric solving, number of
        # gridpoints are specified anyways:
        self.num_gp = 3
        # preallocate grids:
        # temperature
        self.T = np.zeros(3, dtype=np.float64)
        self._T_init = np.zeros_like(self.T)  # init temp for resetting env.
        # preallocate T ports array (here only used for dimension checking)
        self._T_port = np.zeros_like(self.T)
        # massflow
        self.dm = np.zeros(3, dtype=np.float64)
        # grids for indexing in/outflowing and 0-flow ports:
        self._dm_in = np.zeros(3, dtype=bool)
        self._dm_out = np.zeros(3, dtype=bool)
        self._dm0 = np.zeros(3, dtype=bool)
        # U value??? not needed so far, perhaps for passing on U-values??
        #        self.U = np.zeros(3, dtype=np.float64)
        # cp value:
        self._cp_T = np.full(3, self._tiny, dtype=np.float64)
        self._cp_out = np.full(1, self._tiny, dtype=np.float64)
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
        # port ids to access connected ports
        self.port_ids = np.array((), dtype=np.int32)
        # preallocate port values to avoid allocating in loop:
        self._port_vals = np.zeros(self.port_num)
        # port setup
        self.port_names = tuple(('A', 'B', 'C'))
        # set massflow characteristics for ports: in means that an inflowing
        # massflow has a positive sign, out means that an outflowing massflow
        # is pos.
        self.dm_char = tuple(('in', 'in', 'in'))
        # preallocate list to mark ports which have already been solved in
        # topology (to enable creating subnets)
        self._solved_ports = list()

        # construct partname+portname to get fast access to own ports:
        dummy_var = list(self.port_names)
        for i in range(self.port_num):
            dummy_var[i] = self.name + ';' + dummy_var[i]
        self._own_ports = tuple(dummy_var)

        # preallocate massflow grid with port_num. An estimate of total rows
        # will be preallocated before simulation start in initialize_sim:
        self.res_dm = np.zeros((2, self.port_num))

        #        # assert and get material specs:
        #        err_str = ('`material` must be given and has to be one of the '
        #                   'following:\n' + str(list(self._models._mat_props.index)) +
        #                   '\nNew materials can be added to `mat_props.pickle` if '
        #                   'required.')
        #        assert material in self._models._mat_props.index, err_str
        #
        #        # get size specifications from pipe table:
        #        # assert correct pipe specs:
        #        err_str = ('`pipe_type` must be given and has to be one of the '
        #                   'following:\n'
        #                   + str(list(self._models._pipe_specs.index.levels[0])) +
        #                   '. Other pipe types can be added to `pipe_specs.pickle` '
        #                   'if required.')
        #        assert pipe_type in self._models._pipe_specs.index, err_str
        #        # assert correct DN:
        #        err_str = ('The pipes nominal diameter has to be given like '
        #                   '`DN=\'DN50\'`! If the diameter was not found, the table '
        #                   'with supported pipe specs in `pipe_specs.pickle` can be '
        #                   'extended.')
        #        assert DN in self._models._pipe_specs.loc[pipe_type].index, err_str
        #        # now get specs: cross section area of pipe wall and fluid flow area
        #        self._A_wll = self._models._pipe_specs.loc[pipe_type, DN]['A_wall']
        #        self._A_fld = self._models._pipe_specs.loc[pipe_type, DN]['A_i']

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
        self.break_topology = True
        # count how many ports are still open to be solved by topology. If
        # break topology is True, this is used to set it to False if 1 is
        # reached.
        self._cnt_open_prts = self.port_num
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
        self.is_actuator = False
        # if the part HAS TO BE controlled by the control algorithm:
        self.control_req = False
        # if the part needs a special control algorithm (for parts with 2 or
        # more controllable inlets/outlets/...):
        self.actuator_special = False
        # if the parts get_diff method is solved with memory views entirely and
        # thus has arrays which are extended by +2 (+1 at each end):
        self.enlarged_memview = False
        # if the part has a special plot method which is defined within the
        # part's class:
        self.plot_special = True

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

        # save all kind of info stuff to dicts:
        # topology info:
        self.info_topology = dict()

        # no initialization needed:
        self.initialized = False

    def init_part(self, *, material, pipe_specs, **kwargs):
        """
        Set initial conditions for the this 3w connector part.
        """

        # get material properties and pipe specifications:
        self._get_specs_n_props(material=material, pipe_specs=pipe_specs)

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

        # construct list of differential input argument names IN THE CORRECT
        # ORDER!!!
        self._input_arg_names_sorted = [
            'T',
            'ports_all',
            '_cp_T',
            'dm',
            '_port_link_idx',
            'res',
            'stepnum',
        ]

        # update init status:
        self.initialized = True

    def _reset_to_init_cond(self):
        pass

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

        # get port index of target port to create memory view:
        trgt_idx = self._get_topo_cond(port)

        # if only one port of part is remaining to be solved it will always be
        # solved by using the other OWN ports to solve it (also works if
        # part is the primary part of a subnet):
        if self._cnt_open_prts == 1:
            # for 3w connectors the operation id is always -1 when there is
            # only one port is remaining to be solved. This means the
            # negative value of the sum of the other two ports is used, since
            # massflow is positive for inflowing flow for all ports:
            operation_id = -1

            # get port indices of the source ports to create memory views to
            # these ports. to find source ports loop over port names:
            src1_idx, src2_idx = self._port_own_idx[
                [
                    x
                    for x, y in enumerate(self.port_names)
                    if port != (self.port_names[x])
                ]
            ]
            # make slices out of src indices:
            src1_idx = slice(src1_idx, src1_idx + 1)
            src2_idx = slice(src2_idx, src2_idx + 1)
            # add operation instructions to tuple (memory view to target
            # massflow array cell, operation id and memory views to the
            # other/source port's massflow array cells:
            op_routine = (
                self._dm_io.reshape(-1)[trgt_idx],
                operation_id,
                self._dm_io.reshape(-1)[src1_idx],
                self._dm_io.reshape(-1)[src2_idx],
            )
        # 3w connector has more than one port remaining to be solved, thus
        # getting value from connected part:
        else:
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
            # if 3w connector is not at the start of a net, they just pass on
            # the value of the previous part, thus op id is 0 or -1 if neg. and
            # only one source port is needed.
            if alg_sign == 'positive':
                # if positive, only using a memory view
                operation_id = 0
            else:
                # if negative, a negative copy has to be made
                operation_id = -1

            # add operation instructions to tuple (memory view to target
            # massflow array cell, operation id and memory view to the source
            # port's massflow array cell):
            op_routine = (
                self._dm_io.reshape(-1)[trgt_idx],
                operation_id,
                self._models.parts[src_part]._dm_io.reshape(-1)[src_idx],
            )

        # update solved ports list and counter stop break:
        self._solved_ports.append(port)
        self._cnt_open_prts = self.port_num - len(self._solved_ports)
        # set break topology to False if enough ports have been solved:
        self.break_topology = True if self._cnt_open_prts > 1 else False
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
            src_port
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

        #        # depending on the flow conditions this 3w connector acts as a flow
        #        # mixing or splitting device. This state has to be determined by
        #        # checking the direction of the massflows through the ports.
        #        # A negative sign means that the massflow is exiting through the
        #        # respective port, a positive sign is an ingoing massflow.
        #
        #        # get connected port temperatures:
        #        # get port array:
        #        self.T[:] = self._models.ports_all[self._port_link_idx]
        #        # get cp-values of all temperatures:
        #        get_cp_water(self.T, self._cp_T)
        #
        #        # save bool indices of massflows greater (in) and less (out) than 0:
        #        # (using dm as massflow array only works since it is a view of _dm_io!)
        #        self._dm_in[:] = np.greater(self.dm, 0)
        #        self._dm_out[:] = np.less(self.dm, 0)
        #        # also get the integer indexes of this to use faster indexing:
        #        """
        #        TO DO: check if this is faster using integer index arrays and summing
        #        up the bools right at the beginning!
        #        then self._dm_in will be the integer index and the bool sum will be
        #        named self._dm_in_sum or something like this. construction will be:
        #            self._dm_in_bool[:] = np.greater(self.dm, 0)
        #            self._dm_in = np.where(self._dm_in_bool)  <-- this could be SLOW!
        #            self._dm_in_sum = np.sum(self._dm_in_bool)
        #        """
        #        self._dm_in_int = np.where(self._dm_in)
        #        self._dm_out_int = np.where(self._dm_out)
        #
        #        # if 2 ports > 0 are True, 3w connector is mixer:
        #        if np.sum(self._dm_in) == 2:
        #            # get cp of outflowing massflow:
        #            get_cp_water(np.sum(self.T[self._dm_in])/2, self._cp_out)
        #            # calc T_out by mixing the inflowing massflows (*-1 since outgoing
        #            # massflows have a negative sign):
        #            T_out = (np.sum(self.dm[self._dm_in]
        #                            * self._cp_T[self._dm_in]
        #                            * self.T[self._dm_in])
        #                     / (self._cp_out * -1 * self.dm[self._dm_out]))
        #            # pass on port values by switching temperatures:
        #            # set old T_out to both in-ports
        #            self.T[self._dm_in] = self.T[self._dm_out]
        #            # set calculated T_out to out-port
        #            self.T[self._dm_out] = T_out
        #        # if 2 ports < 0 are True, 3w connector is splitter:
        #        elif np.sum(self._dm_out) == 2:
        #            # no real calculation has to be done here, just switching
        #            # temperatures and passing them on to opposite ports
        #            # calc the temp which will be shown at the inflowing port as a mean
        #            # of the temps of outflowing ports (at in port connected part will
        #            # see a mean value of both temps for heat conduction):
        #            T_in = self.T[self._dm_out].sum() / 2
        #            # pass inflowing temp to outflowing ports:
        #            self.T[self._dm_out] = self.T[self._dm_in]
        #            # pass mean out temp to in port:
        #            self.T[self._dm_in] = T_in
        #        # if one port has 0 massflow, sum of dm_in == 1:
        #        elif np.sum(self._dm_in) == 1:
        #            # get port with 0 massflow:
        #            self._dm0[:] = np.equal(self.dm, 0, subok=False)
        #            # this port 'sees' a mean of the other two temperatures:
        #            self.T[self._dm0] = self.T[~self._dm0].sum() / 2
        #            # the out ports heat flow is dominated by convection, thus it
        #            # only 'sees' the in flow temperature but not the 0 flow temp:
        #            self.T[self._dm_out] = self.T[self._dm_in]
        #            # the in ports heat flow is also dominated by convection, but here
        #            # it is easy to implement the 0-flow port influence, since heat
        #            # flow by convection of part connected to in port is not affected
        #            # by connected temperature, thus also get a mean value:
        #            self.T[self._dm_in] = self.T[~self._dm_in].sum() / 2
        #        # if all ports have 0 massflow:
        #        else:
        #            # here all ports see a mean of the other ports:
        #            # bkp 2 ports
        #            T0 = (self.T[1] + self.T[2]) / 2
        #            T1 = (self.T[0] + self.T[2]) / 2
        #            # save means to port values:
        #            self.T[2] = (self.T[0] + self.T[1]) / 2
        #            self.T[0] = T0
        #            self.T[1] = T1
        #
        #        # save results:
        #        self.res[self.stepnum] = self.T
        solve_connector_3w(
            T=self.T,
            ports_all=self._models.ports_all,
            cp_T=self._cp_T,
            dm=self.dm,
            port_link_idx=self._port_link_idx,
            res=self.res,
            stepnum=self.stepnum,
        )

    def draw_part(self, axis, timestep, draw, animate=False):
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
                'For plotting of 3w-connectors the orientation of the '
                'connector must be given! Please pass the orientation '
                'as `orientation=\'string\'` to the 3w-connectors\'s '
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
            # now run their set plot shape with that new information again:
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
            __pt['txtC_offset'] = tuple(
                self._models._rotate_vector(
                    np.array([1.5 * offset, -0.5 * offset]), __pt['rot_angle']
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
                animated=animate,
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
            txtC = (
                'C'
                + r'\n$\dot{m} = $'
                + str(self.res_dm[timestep, 2])
                + r'$\,$kg/s'
            )

            axis.annotate(
                txt,
                xy=(self.__pt['pos_center']),
                xytext=self.__pt['txt_offset'],
                textcoords='offset points',
                ha='center',
                va='center',
                animated=animate,
            )
            axis.annotate(
                txtA,
                xy=(self.info_plot['A']['coordinates']),
                xytext=self.__pt['txtA_offset'],
                textcoords='offset points',
                ha='center',
                va='center',
                animated=animate,
            )
            axis.annotate(
                txtB,
                xy=(self.info_plot['B']['coordinates']),
                xytext=self.__pt['txtB_offset'],
                textcoords='offset points',
                ha='center',
                va='center',
                animated=animate,
            )
            axis.annotate(
                txtC,
                xy=(self.info_plot['C']['coordinates']),
                xytext=self.__pt['txtC_offset'],
                textcoords='offset points',
                ha='center',
                va='center',
                animated=animate,
            )

            if animate:
                return []
