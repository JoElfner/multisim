# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 11:38:54 2018

@author: elfner
"""

from .pipe import Pipe


class PipeWithPump(Pipe):
    r"""
    type: Single Pipe with an integrated pump.

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
        self.constr_type = 'PipeWithPump'  # define construction type
        # since this part is a subclass of Pipe, initialize Pipe:
        super().__init__(
            name, master_cls, **kwargs, constr_type=self.constr_type
        )

        # if the part CAN BE controlled by the control algorithm:
        self.is_actuator = True
        self._actuator_CV = self.dm[:]  # set array to be controlled
        self._actuator_CV_name = 'massflow'  # set description
        self._unit = '[kg/s]'  # set unit of control variable
        # if the part HAS TO BE controlled by the control algorithm:
        self.control_req = True
        # if the part needs a special control algorithm (for parts with 2 or
        # more controllable inlets/outlets/...):
        self.actuator_special = False
        # initialize bool if control specified:
        self.ctrl_defined = False
        # if part can be a parent part of a primary flow net:
        self._flow_net_parent = True

        # IMPORTANT: THIS VARIABLE **MUST NOT BE INHERITED BY SUB-CLASSES**!!
        # If sub-classes are inherited from this part, this bool checker AND
        # the following variables MUST BE OVERWRITTEN!
        # ist the diff function fully njitted AND are all input-variables
        # stored in a container?
        self._diff_fully_njit = False
        # self._diff_njit = pipe1D_diff  # handle to njitted diff function
        # input args are created in simenv _create_diff_inputs method

    #        err_str = (
    #            self._base_err +
    #            self._arg_err.format('lower_limit, upper_limit') +
    #            'The part was set to be an actuator and need a control with '
    #            '`no_control=False`, thus `lower_limit` and `upper_limit` '
    #            'in {0} have to be passed to clip the controller action on '
    #            'the actuator to the limits.\n'
    #            'The limits have to be given as integer or float values with '
    #            '`lower_limit < upper_limit`.').format(self._unit)
    #        assert 'lower_limit' in kwargs and 'upper_limit' in kwargs, err_str
    #        self._lims = np.array(  # set limits to array
    #                [kwargs['lower_limit'], kwargs['upper_limit']],
    #                dtype=np.float64)
    #        self._llim = self._lims[0]  # also save to single floats
    #        self._ulim = self._lims[1]  # also save to single floats
    #        assert 0 <= self._lims[0] < self._lims[1], (
    #            err_str + ' For HeatedPipe limits are additionally restricted '
    #            'to `0 <= lower_limit < upper_limit`.')

    def init_part(self, *, start_massflow, **kwds):
        """Initialize the part."""
        # since this part is a subclass of Pipe, call init_part of Pipe:
        super().init_part(**kwds)

        # set starting valve opening:
        err_str = (
            self._base_err
            + self._arg_err.format('start_massflow')
            + 'The initial massflow has to be set in the range of '
            '`0 <= start_massflow`.'
        )
        assert (
            isinstance(start_massflow, (int, float)) and 0 <= start_massflow
        ), err_str
        self.dm[0] = start_massflow
        self._dm_init = start_massflow  # bkp for re-initializing

        self._initialize_actuator(**kwds)
        if ('ctrl_required' not in kwds) or kwds['ctrl_required']:
            assert 'maximum_flow' in kwds and isinstance(
                kwds['maximum_flow'], (int, float)
            ), (
                self._base_err
                + self._arg_err.format('maximum_flow')
                + 'The maximum pump flow in [kg/s] must be given. This value '
                'is only used if the controller is set to '
                '`process_cv_mode=\'part_specific\'` to convert the CV value '
                'from the 0-1 range to the part specific 0-1*max_flow range.\n'
                'This value is can be set independently of the lower/upper '
                'limit to enable clipping of the massflow. Clipping will be '
                'applied AFTER converting the range!'
            )
            self._flow_at_max_speed = kwds['maximum_flow']

    def _get_flow_routine(  # overwrite the pipe method with the pump method
        self, port, parent_port=None, subnet=False, **kwargs
    ):
        """
        Return the flow calculation routine for the current part.

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

        Parameters
        ----------
        port : str
            Port name of the port which shall be calculated (target port).

        """
        # get port index slice of target port to create memory view:
        trgt_idx = self._get_topo_cond(port)

        # if part is the starting point of a net (this part as a pipe
        # containing a pump is ALWAYS the starting point of a primary flow
        # net!) OR this part is hitting itself again in the topology
        # (circular net):
        if parent_port is None:
            # for pumps there is no operation id, since they will always
            # be the parent part of the whole net and will thus define the nets
            # massflow, won't have another parent part and won't need any
            # operation routine!
            pass
        elif self.name == kwargs['parent_pump']:
            return ()
        # (pipe with) pump not at start of net:
        else:
            # this will only raise an error and then make the topology analyzer
            # break:
            err_str = (
                'Pipe with pump ' + self.name + ' was added to a flow network '
                'where another pump is already existing. There must '
                'not be two pumps in the same flow network!'
            )
            raise TypeError(err_str)

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

    def _process_cv(self, ctrlr):
        """Convert 0-1 value range to part specific value range."""
        cv = ctrlr.cv * self._flow_at_max_speed
        self.dm[:] = (
            cv
            if self._llim <= cv <= self._ulim
            else self._ulim
            if cv > self._ulim
            else self._llim
        )


#    def get_diff(self, timestep):
#        """
#        This function just calls a jitted calculation function.
#
#        """
#
#        pipe1D_diff(*self._input_args, timestep)
#
#        return self.dT_total
