# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Aug 2017

TODO:
      Für die Betrachtung der lokalen Film-Koeffizienten (alpha/k-Werte) müssen
      die lokalen Nusselt-Gleichungen aus G1 herangezogen werden. Für den
      Einlauf möglicherweise mit hydr./therm. Anlauf nach eq (9) auf Seite 786,
      sonst für laminar eq(3) auf Seite 786 und für turb eq (28) auf Seite 788.
      G1 Abb1 auf Seite 789 zeigt die Bereiche der Nusselt-Gleichungen. Es
      geht hervor, dass der Einsatz der Gleichung für hydr./therm. Anlauf nach
      eq (9) sinnvoll ist.
      Die Berechnung in phex_alpha_i_wll_sep() sollte angepasst werden. Re
      muss kein array sein. Die ganze Berechnung ist einfacher mit if-else
      Abfrage für turb/nicht-turb. phex_alpha_i_wll_sep() sollte dann von einer
      umschließenden Funktion für a und b aufgerufen werden. Das Ganze in eine
      Iteration.
"""

import numpy as np
from ..simenv import SimEnv
from .. import precomp_funs as _pf


class HeatExchanger(SimEnv):
    """
    type: Plate heat exchanger using a NTU-method coupled with capacities for
    transient heat flow calculations.

    Can be added to the simulation environment by using the following method:
        .add_part(PlateHEx, name, volume=..., grid_points=...,
        outer_diameter=..., shell_thickness=...)
    Part creation parameters:
    -------------------------
    PlateHEx : class name
        This parameter can\'t be changed and needs to be passed exactly like
        this.
    name : string
        Plate heat exchanger identifier as string. Needs to be unique.
    length : integer, float
        Thermal energy storage volume in [m^3].

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

        self.constr_type = 'HeatExchanger'  # define construction type
        base_err = (  # define leading base error message
            'While adding {0} `{1}` to the simulation '
            'environment, the following error occurred:\n'
        ).format(self.constr_type, str(name))
        arg_err = (  # define leading error for missing/incorrect argument
            'Missing argument or incorrect type/value: {0}\n\n'
        )
        self._base_err = base_err  # save to self to access it in methods
        self._arg_err = arg_err  # save to self to access it in methods

        # check kind of hex:
        hex_types = ['plate_hex']
        err_str = self._base_err + self._arg_err.format(
            'HEx_type'
        ) + 'The construction type has to be given with `HEx_type=X`, with ' '`X` being one of the following strings:\n' + str(
            hex_types
        )
        assert (
            'HEx_type' in kwargs and kwargs['HEx_type'] in hex_types
        ), err_str
        self.HEx_type = kwargs['HEx_type']  # save type
        if self.HEx_type == 'plate_hex':  # get parameters depending on type
            # assert that all required arguments have been passed:
            err_str = (
                self._base_err
                + self._arg_err.format('passes')
                + 'For a plate heat exchanger the number of passes has to be'
                'given with with `passes=X` as an integer value >0. In the '
                'current implementation, the number of passes on the supply '
                'and demand side are equal. Other configurations may be '
                'implemented later.'
            )
            assert (
                'passes' in kwargs
                and type(kwargs['passes']) == int
                and kwargs['passes'] > 0
            ), err_str
            self.passes = kwargs['passes']
            # get plates
            err_str = (
                self._base_err
                + self._arg_err.format('plates')
                + 'For a plate heat exchanger the number of plates has to be '
                'given with `plates=X` as an integer value >4. The number of '
                'plates has to be an even number for a plate heat exchanger '
                'with one pass on each side. For other numbers of passes, '
                '`(plates - 1)/passes` must be an even number.'
            )
            assert (
                'plates' in kwargs and type(kwargs['plates']) == int
            ), err_str
            if self.passes == 1:  # if one pass plate hex
                assert kwargs['plates'] % 2 == 0, err_str  # even num plates
                self.num_plates = kwargs['plates']  # save value
                err_chn = (  # err string for number of channels
                    self._base_err
                    + self._arg_err.format('max_channels')
                    + 'A one pass plate heat exchanger has `plates/2` '
                    'channels on one side and `plates/2 - 1` channels on '
                    'the other side. Pass the side where `plates/2` '
                    'channels are located with `max_channels=X`, where '
                    '`X=\'supply\'` or `X=\'demand\'`.'
                )
                assert (
                    'max_channels' in kwargs
                    and type(kwargs['max_channels']) == str  # assert type
                ), err_chn
                assert (
                    kwargs['max_channels'] == 'supply'
                    or kwargs['max_channels'] == 'demand'  # assert entry
                ), err_chn
                if kwargs['max_channels'] == 'supply':
                    self._num_channels_sup = int(self.num_plates / 2)
                    self._num_channels_dmd = int(self.num_plates / 2 - 1)
                else:
                    self._num_channels_dmd = int(self.num_plates / 2)
                    self._num_channels_sup = int(self.num_plates / 2 - 1)
            else:  # for >1 passes
                assert ((kwargs['plates'] - 1) / self.passes) % 2 == 0, err_str
                self.num_plates = int(kwargs['plates'])
                self._num_channels_sup = int(  # calc number of channels
                    (self.num_plates - 1) / (self.passes * 2)
                )
                self._num_channels_dmd = self._num_channels_sup
        #        # number of eff. transfer areas is independent of passes:
        #        self._num_A = self.num_plates - 2  # number of eff. transfer areas
        # number of eff. heat transfer areas PER pass:
        self._num_A = int((self.num_plates - 2) / self.passes)
        #        err_str = ('The heat exchanger\'s number of channels on the supply '
        #                   'side has to be passed to its `add_part()` method with '
        #                   '`channels_sup_side=X` as an integer value >1!')
        #        assert ('channels_sup_side' in kwargs and
        #                type(kwargs['channels_sup_side']) == int and
        #                kwargs['channels_sup_side'] >= 1), err_str
        #        self.num_channels_sup = kwargs['channels_sup_side']
        #        err_str = ('The heat exchanger\'s number of channels on the demand '
        #                   'side has to be passed to its `add_part()` method with '
        #                   '`channels_dmd_side=X` as an integer value >1!')
        #        assert ('channels_dmd_side' in kwargs and
        #                type(kwargs['channels_dmd_side']) == int and
        #                kwargs['channels_dmd_side'] >= 1), err_str
        #        self.num_channels_dmd = kwargs['channels_dmd_side']
        #        err_str = ('The heat exchanger\'s number of passes on the supply side '
        #                   'has to be passed to its `add_part()` method with '
        #                   '`passes_sup_side=X` as an integer value >1!')
        #        assert ('passes_sup_side' in kwargs and
        #                type(kwargs['passes_sup_side']) == int and
        #                kwargs['passes_sup_side'] >= 1), err_str
        #        self.num_passes_sup = kwargs['passes_sup_side']
        #        err_str = ('The heat exchanger\'s number of passes on the demand side '
        #                   'has to be passed to its `add_part()` method with '
        #                   '`passes_dmd_side=X` as an integer value >=1!')
        #        assert ('passes_dmd_side' in kwargs and
        #                type(kwargs['passes_dmd_side']) == int and
        #                kwargs['passes_dmd_side'] >= 1), err_str
        #        self.num_passes_dmd = kwargs['passes_dmd_side']

        # set number of gridpoints to 4:
        self.num_gp = 4

        # %% start part construction:
        #        super().__init__()
        self.name = name
        self.part_id = self._models.num_parts - 1
        # save smallest possible float number for avoiding 0-division:
        self._tiny = self._models._tiny

        # %% preallocate temperature and property grids:

        # preallocate temperature grid with index (i, j) where i=0 is top side
        # (sup in, wall top, dmd out), i=1 are the log mean temp differences
        # and i=2 is the bottom side (sup out, wall bot, dmd in) and j is the
        # index for the flows and the wall with j=0 sup, j=1 wall and j=2 dmd:
        self.T = np.zeros((3, 3), dtype=np.float64)
        """"""
        self.T = np.zeros(4, dtype=np.float64)
        Tshp = self.T.shape  # save shape since it is needed often
        self._T_init = np.zeros(Tshp)  # init temp for resetting env.
        #        self.T = np.zeros(3, dtype=np.float64)  # old 1 row T
        # views to supply and demand side:
        """
        self._T_sup = self.T[:, 0]  # view to supply side
        self._T_dmd = self.T[:, 2]  # view to demand side
        """
        self._T_sup = self.T[:2]  # view to supply side
        self._T_dmd = self.T[2:]  # view to demand side
        # preallocate temperature grids for logarithmic mean temperatures
        """

        self._T_lm = self.T[1, :]  # view to log mean row of T
        self._T_wll_lm = self.T[1, 1:2]  # view to log mean of wall
        self._T_sup_lm = self.T[1, 0:1]  # view to log mean of supply side
        self._T_dmd_lm = self.T[1, 2:3]  # view to log mean of demand side

        """
        self._T_mean = np.zeros(2)  # array for arithmetic mean temperatures
        # preallocate temperature grid for ports:
        self._T_port = np.zeros_like(self.T)
        """"""
        self._T_port = np.zeros(4)
        Tpshp = self._T_port.shape  # save shape since it is needed often
        # preallocate view to temperature array for fluid props. calculation:
        """
        self._T_fld = self.T[:, ::2]
        """
        self._T_fld = self.T[:]
        # preallocate view to reshaped temperature array for alpha calculation:
        # (fluid temperature must be in row 0, wall temp. in row 1)
        #        self._T_resh_1 = self.T[:2].reshape((1, 2))[:]  # old for 1 row T
        #        self._T_resh_2 = self.T[2:0:-1].reshape((1, 2))[:]  # old for 1 row T
        """
        self._T_resh_1 = self.T[:, 0:2]
        self._T_resh_2 = self.T[:, 2:0:-1]
        """
        # preallocate lambda grids and alpha grid for heat conduction with the
        # smallest possible value to avoid zero division:
        #        self._lam_T = np.zeros_like(self._T_fld)  # , self._tiny, dtype=np.float64)
        self._lam_mean = np.zeros(2)
        #        self._lam_ports = np.zeros(Tpshp)  # , self._tiny)
        #        self._alpha_i = np.full_like(self._lam_T, 3.122)
        self._alpha_i = np.full(2, 3.122)
        self._alpha_inf = np.full((1,), 3.122)
        # preallocate heat capacity grids:
        #        self._cp_T = np.zeros_like(self._T_fld)
        self._cp_mean = np.zeros(2)
        self._cp_port = np.zeros(Tpshp)
        # preallocate density grids:
        #        self._rho_T = np.zeros_like(self._T_fld)
        self._rho_mean = np.zeros(2)
        # preallocate kinematic viscosity grid:
        #        self._ny_T = np.zeros_like(self._T_fld)
        self._ny_mean = np.zeros(2)
        # preallocate U*A grids:
        self._UA_port = np.zeros(Tpshp)
        self._UA_amb = np.zeros(1)  # casing to amb (assumed having plate temp)
        #        self._UA_fld_wll = np.zeros_like(self._T_fld)  # from fluid to plate
        self._UA_fld_wll = np.zeros(2)  # from fluid to plate
        self._UA_fld_wll_fld = np.zeros(1)  # from fluid to plate to fluid
        # preallocate mass flow grids:
        self.dm = np.zeros(2)
        #        self._dm_cell = np.zeros_like(self.dm)
        #        self._dm_sup = np.zeros_like(self.dm)
        #        self._dm_dmd = np.zeros_like(self.dm)
        self._dm_port = np.zeros(Tpshp)
        #        self._dm_io = np.zeros(Tpshp)  # array for I/O flow
        #        # this I/O flow array MUST NOT BE CHANGED by anything else than
        #        # _update_FlowNet() method!
        #        self._dm_sup = self._dm_io[:, 0]  # view to supply side
        #        self._dm_dmd = self._dm_io[:, 2]  # view to demand side

        self._dm_io = np.zeros(2)  # array for I/O flow, one cell per channel
        #        self._dm_sup = self._dm_io[:, 0]  # view to supply side
        #        self._dm_dmd = self._dm_io[:, 2]  # view to demand side
        self._dm_sup = self._dm_io[:1]  # view to supply side
        self._dm_dmd = self._dm_io[1:]  # view to demand side

        #        self._dm_port = np.zeros((1, self.dm.shape[0]))
        # preallocate result grid with one row. An estimate of total rows will
        # be preallocated before simulation start in initialize_sim. massflow
        # grid is preallocated in set_initial_cond:
        self.res = np.zeros((1,) + self.T.shape)
        self.res_dm = np.zeros((2, 1))
        # separate result grids for conduction and advection to make the code
        # less complicated:
        self.dT_cond = np.zeros(Tshp)
        self.dT_adv = np.zeros_like(2)
        self.dT_total = np.zeros(Tshp)

        # grid spacing is set to zero to disable heat conduction to other
        # parts:
        self.grid_spacing = np.zeros(Tpshp)

        # port definition (first and last array element):
        self.port_num = 4
        # Index to own value array to get values of own ports, meaning if I
        # index a FLATTENED self.T.flat with self._port_own_idx, I need to
        # get values accoring to the order given in self.port_names.
        # That is, this must yield the value of the cell in self.T, which is
        # belonging to the port 'in':
        # self.T.flat[self._port_own_idx[self.port_names.index('in')]]
        self._port_own_idx = np.array(
            [0, 6, 2, 8], dtype=np.int32
        )  # flat 2d index to value array
        self._port_own_idx = np.array([0, 1, 2, 3], dtype=np.int32)
        self._port_own_idx_2D = np.array(
            [0, 6, 2, 8], dtype=np.int32
        )  # flat 2d index to port array
        self._port_own_idx_2D = np.array([0, 1, 2, 3], dtype=np.int32)
        self.port_ids = np.array((), dtype=np.int32)
        # define flow channel names for both sides. used in topology constr.
        self._flow_channels = ('sup', 'dmd')
        # define port_names
        self.port_names = tuple(('sup_in', 'sup_out', 'dmd_in', 'dmd_out'))
        # set massflow characteristics for ports: in means that an inflowing
        # massflow has a positive sign, out means that an outflowing massflow
        # is pos. Since a TES may need a cumulative sum over all ports to get
        # the massflow of the last port, massflow is ALWAYS positive when
        # inflowing!
        self.dm_char = tuple(('in', 'out', 'in', 'out'))
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
        self._port_gsp = np.zeros(Tpshp)  # , self._tiny)
        self._lam_wll_conn_p = np.full_like(self._T_port, 1e-2)
        self._lam_port_fld = np.zeros(Tpshp)  # , self._tiny)
        self._lam_fld_own_p = np.full_like(self._T_port, 0.6)
        self._UA_fld_ports = np.zeros(Tpshp)
        # preallocate list to mark ports which have already been solved in
        # topology (to enable creating subnets)
        self._solved_ports = list()
        # preallocate grids for von Neumann stability checking:
        self._vN_diff = np.zeros(4)
        #        self._vN_dm = np.zeros(1)  # this is a scalar for dm_invariant=True
        self._trnc_err = 0.0
        # add truncation error cell weight to only calculate the error for cell
        # 1,1 where heat conduction is considered:
        self._trnc_err_cell_weight = np.zeros(Tshp)
        """"""
        self._trnc_err_cell_weight = np.zeros(4)
        #        self._trnc_err_cell_weight[1, 1] = 1.
        self._trnc_err_cell_weight = 0.0
        # preallocate grids for precalculated stuff: REPLACED WITH LOCAL VARS!
        #        self.__rhocp = np.zeros(Tshp)
        #        self.__cpT = np.zeros(Tshp)

        # set if type has to be solved numeric:
        self.solve_numeric = False
        # if port arrays shall be collapsed to amount of ports to improve speed
        self.collapse_arrays = True  # DONT SET TO TRUE HERE
        self._collapsed = True  # bool checker if already collapsed
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
        self._cnt_open_prts = self.port_num
        self._port_heatcond = False  # if heatcond. over ports is enabled
        # determine if part has the capability to affect massflow (dm) by
        # diverting flow through ports or adding flow through ports:
        self.affect_dm = False
        # if the massflow (dm) has the same value in all cells of the part
        # (respectively in each flow channel for parts with multiple flows):
        self.dm_invariant = True
        # if the part has multiple separated flow channels which do NOT mix
        # (like a heat exchanger for exampe):
        self.multiple_flows = True
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
        self.plot_special = True

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

    def init_part(self, *, Reynolds_correction=1e-6, **kwargs):
        # check if kwargs given:
        err_str = (
            'No parameters have been passed to the `init_part()` method'
            ' of heat exchanger `' + self.name + '`! To get '
            'additional information about which parameters to pass, '
            'at least the insulation thickness in [m] has to be given '
            '(0 is also allowed) with `insulation_thickness=X`.'
        )
        assert kwargs, err_str

        # check for insulation:
        err_str = (
            'The insulation thickness of a heat exchanger has '
            'to be passed to its `init_part()` method in [m] with '
            '`insulation_thickness=X`!'
        )
        assert 'insulation_thickness' in kwargs, err_str
        self._s_ins = float(kwargs['insulation_thickness'])

        err_str = (
            'The lambda value of the pipe\'s insulation has to be '
            'passed to its `init_part()` method in '
            '[W/(m*K)] with `insulation_lambda=X`!'
        )
        assert 'insulation_lambda' in kwargs, err_str
        self._lam_ins = float(kwargs['insulation_lambda'])

        # assert material and pipe specs (after defining port names to avoid
        # name getting conflicts):
        err_str = (
            '`pipe_specs` and `material` for '
            + self.constr_type
            + ' `'
            + self.name
            + '` must be passed to its `init_part()` '
            'method! Pipe specifications define the connection '
            'parameters to other parts for heat conduction, thus are '
            'also required for parts that are not a pipe.'
        )
        assert 'material' in kwargs and 'pipe_specs' in kwargs, err_str
        # get material pipe specs for diameter etc.:
        self._get_specs_n_props(**kwargs)
        # assert that pipe specs were given for all ports together:
        err_str = (
            'For a part of type heat exchanger, the same `pipe_specs` '
            'must be given for all ports by specifying \'all\' as '
            'first-level-key!'
        )
        assert 'all' in kwargs['pipe_specs'], err_str
        # backup wall heat conductivity for plate material into single array:
        self._lam_plate = self._lam_wll[:1].copy()

        # get geometry:
        self.get_plate_hex_geometry(**kwargs)

        # assert and get initial temperature:
        err_str = (
            'The initial temperature `T_init` must be passed to the '
            '`init_part()` method of part ' + self.name + ' in [°C] as '
            'a single float or integer value or as an array with shape '
            '(' + str(self.num_gp) + ',).'
        )
        assert 'T_init' in kwargs, err_str
        self._T_init = kwargs['T_init']
        assert isinstance(self._T_init, (float, int, np.ndarray)), err_str
        # set init values to T array:
        if type(self._T_init) != np.ndarray:
            self.T[:] = float(self._T_init)
            self._T_init = float(self._T_init)
        else:
            # assert that correct shape is given and if yes, save to T
            assert self._T_init.shape == self.T.shape, err_str
            self.T[:] = self._T_init

        err_str = (
            'If a correction for the Reynolds number to adjust the '
            'flow regime to profiled plates is passed, this correction '
            'value has to be an integer or float value >0!\n'
            'This correction value will be added to the Reynolds '
            'number during calculation of the heat transfer '
            'coefficients.\n'
            'A multiplication factor or power were tested but found '
            'to be instable and difficult to adjust to the specific '
            'heat exchanger and were thus removed.\n'
            'To automaticaly adapt the Reynolds correction to given '
            'heat coefficients of a specific heat exchanger, call '
            'the method '
            '`modenv.parts[\'' + self.name + '\'].find_Reynolds_correction()`.'
        )
        assert (
            isinstance(Reynolds_correction, (int, float))
            and Reynolds_correction > 0
        ), err_str
        self._corr_Re = np.array([Reynolds_correction], dtype=np.float64)

        # ---> geometry/grid definition
        # set array for each cell's distance from the start of the heat
        # exchanger. This is for constant cross section, VDI Wärmeatlas 2013
        # p 101, the arithmetic mean. Since multiple passes means "restarting"
        # the flow after each pass by strong diversions, this value is
        # independent of passes: FALSE! WRONG!
        # set cell dist to length while still using mean Nusselt numbers.
        # As soon as local Nusselt numbers with Simpson integration are
        # implemented, change this to an array with [tiny, length/2, length]
        self._cell_dist = np.array([self.length])
        # pipe connection parameters
        self._d_o = float(self.info_topology['all_ports']['pipe_specs']['d_o'])
        #        self._r_i = self._d_i / 2
        #        self._r_o = self._d_o / 2
        # total RADIUS from center of the pipe to the outer radius of the
        # insulation:
        #        self._r_total = self._d_o / 2 + self._s_ins
        #        # factor for wall lambda value referred to r_i
        #        self._r_ln_wll = self._r_i * np.log(self._r_o / self._r_i)
        #        # factor for insulation lambda value referred to r_i
        #        self._r_ln_ins = self._r_i * np.log(self._r_total / self._r_o)
        #        # factor for outer alpha value referred to r_i
        #        self._r_rins = self._r_i / self._r_total
        #        # thickness of the wall:
        #        self._s_wll = self._r_o - self._r_i
        #        # cross section area and volume of cell:
        #        self.A_cell = self._A_fld_own_p
        #        self.V_cell = self.A_cell * self.grid_spacing
        # surface area of pipe wall per cell (fluid-wall-contact-area):
        #        self._A_shell_i = np.pi * self._d_i * self.grid_spacing
        #        # outer surface area of pipe wall:
        #        self._A_shell_o = np.pi * self._d_o * self.grid_spacing
        #        # cross section area of shell:
        #        self._A_shell_cs = np.pi / 4 * (self._d_o**2 - self._d_i**2)
        #        # shell volume (except top and bottom cover):
        #        self._V_shell_cs = self._A_shell_cs * self.length

        # calculate m*cp for the wall PER CELL with the material information:
        #        self._mcp_wll = (self._cp_wll * self._rho_wll * self._V_shell_cs
        #                         / self.num_gp)

        # get ambient temperature:
        self._chk_amb_temp(**kwargs)

        # save shape parameters for the calculation of heat losses.
        self._vertical = True
        # get flow length for a vertical rectangular for Nusselt calculation:
        self._flow_length = self.length

        # construct list of differential input argument names IN THE CORRECT
        # ORDER!!!
        # regex to remove strings: [a-zA-Z_]*[ ]*=self.
        self._input_arg_names_sorted = [
            'T',
            '_T_port',
            '_T_mean',
            'ports_all',
            'res',
            '_dm_io',
            '_dm_port',
            '_cp_mean',
            '_lam_mean',
            '_rho_mean',
            '_ny_mean',
            '_lam_wll',
            '_alpha_i',
            '_UA_fld_wll',
            '_UA_fld_wll_fld',
            '_port_own_idx',
            '_port_link_idx',
            'A_plate_eff',
            'A_channel',
            '_d_h',
            '_s_plate',
            '_cell_dist',
            '_num_A',
            '_num_channels_sup',
            '_num_channels_dmd',
            '_corr_Re',
            'stepnum',
        ]

        # set initialization to true:
        self.initialized = True

    def _reset_to_init_cond(self):
        self.T[:] = self._T_init

    def get_plate_hex_geometry(self, **kwargs):
        """
        This calculates the plate heat exchanger's geometry depending on the
        passed inputs.
        """

        def __plate_area(length, width, port_diameter, corner_radius=0):
            """
            Calculates the plate heat exchanger's plate area excluding ports
            and the corner radius.
            """
            # port area for 4 ports with (d/2)**2=d**2/4, 4 is cut out:
            Ap = port_diameter ** 2 * np.pi
            # corner radius area:
            Acr = (2 * corner_radius) ** 2 - corner_radius ** 2 * np.pi
            # return total area except for ports and corner radius:
            return length * width - Ap - Acr

        def __plate_thickness(
            diff_m,
            rho_plate,
            A_plate,
            diff_depth=0,
            length=0,
            width=0,
            corner_radius=0,
        ):
            """
            Calculates the plate thickness from the HEX weight increase per
            plate in [kg], the plate area in [m^2] and the plate material
            density in [kg/m^3]. The influence of the increasing HEX depth can
            be calculated by giving the following parameters in [m]: HEX depth
            increase per plate, HEX length, width and corner radius.
            """
            # get hex circumference:
            U = (
                2 * length
                + 2 * width
                - 8 * corner_radius
                + 2 * corner_radius * np.pi
            )
            # get total plate area, assuming circumference plate has the same
            # thickness as the heat transfer area:
            A_plate += diff_depth * U
            # return plate thickness
            return diff_m / (A_plate * rho_plate)

        def __channel_width_by_plate_thickness(diff_depth, plate_thickness):
            """
            Calculate the channel width (distance between two plates) by the
            increase of the HEX depth per plate and the plate thickness, all
            in [m].
            """
            return diff_depth - plate_thickness

        def __channel_width_by_volume(
            V_channel_total, length, width, corner_radius
        ):
            """
            Calculate the channel width in [m] (distance between two plates) by
            use of the TOTAL channel volume in [m^3] and the HEX length, width
            and corner radius. Total means, that the port diameter MUST NOT be
            considered in the calculation of the volume and plate surface area.
            """
            return V_channel_total / __plate_area(
                length, width, 0, corner_radius
            )

        # save specific error strings:
        err_len = (
            self._base_err
            + self._arg_err.format('length')
            + 'The heat exchanger length (NOT including passes and if available '
            'without casing) has to be given in [m] with `length=X`, where X '
            'is an integer or float value >0.'
        )
        err_wdt = (
            self._base_err
            + self._arg_err.format('width')
            + 'The heat exchanger width, excluding the casing if available, '
            'has to given in [m] with `width=X`, where X is an integer or '
            'float value >0. '
            'If width is only available including the casing, this can be '
            'given as well, since the error is negligible in most cases.'
        )
        err_dpt = (
            self._base_err
            + self._arg_err
            + 'The heat exchanger depth must be given. This can be done in one '
            'of the following ways:\n'
            '    - `depth=X`: The total heat exchanger depth/thickness in [m] '
            'with all plates and INcluding the casing but excluding the '
            'ports, with X being an integer or float value > 0.\n'
            '    - `depth_base=X` and `depth_per_plate=Y`: The base depth of '
            'the heat exchanger without any plates or ports and the depth per '
            'added plate, both in [m]. X and Y must be integer or float '
            'values >0.'
        )
        err_Vfl = (
            self._base_err
            + self._arg_err.format('V_channel')
            + 'The heat exchanger fluid volume **per channel** has to be given '
            'in [m^3] with `V_channel=X`, where X is an integer or float '
            'value >0.'
        )
        err_sch = (
            self._base_err
            + self._arg_err.format('channel_width')
            + 'The heat exchanger\'s channel width (plate distance) has to be '
            'given in [m] with `channel_width=X`, where X is an integer or '
            'float value >0.'
        )
        err_sp = (
            self._base_err
            + self._arg_err.format('plate_thickness')
            + 'The heat exchanger\'s plate thickness has to be given in [m] '
            'with `plate_thickness=X`, where X is an integer or float '
            'value >0, if the increase in depth per plate is not given with '
            '`depth_per_plate=X` in [m] or if not otherwise required.'
        )
        err_cr = (
            self._base_err
            + self._arg_err.format('corner_radius')
            + 'The heat exchanger plate\'s corner radius has to be given in [m] '
            'with `corner_radius=X`, where X is an integer or float '
            'value >=0. Can be set to zero.'
        )
        err_pd = (
            self._base_err
            + self._arg_err.format('port_diameter')
            + 'The heat exchanger\'s port diameter has to be given in [m] with '
            '`port_diameter=X`, where X is an integer or float value >0.'
        )
        err_dd = (
            self._base_err
            + self._arg_err.format('depth_per_plate')
            + 'The heat exchanger\'s increase in depth per additional plate '
            'has to given in [m] with `depth_per_plate=X`, where X is an '
            'integer or float value >0, if the plate thickness is not given '
            'with `plate_thickness=X` in [m] or if otherwise required.'
        )
        err_dm = (
            self._base_err
            + self._arg_err.format('mass_per_plate')
            + 'The heat exchanger\'s increase in mass per additional plate has '
            'to be given in [kg] with `mass_per_plate=X`, where X is an '
            'integer or float value >0.'
        )
        err_mw = (
            self._base_err
            + self._arg_err.format('mass_base OR mass_total')
            + 'The heat exchanger\'s dry mass in [kg] (the mass of the empty '
            'heat exchanger) has to be given either with `mass_base=X`, '
            'giving the base mass without any plates or with `mass_total=X`, '
            'giving the total mass including all plates. Both have to be '
            'given as an integer or float value >0.'
        )
        err_a = (
            self._base_err
            + self._arg_err.format('channel_area')
            + 'If the channel cross section area is passed directly to the heat '
            'exchanger, it has to be given as an integer or float value >0 '
            'in [m^2].'
        )
        #
        # get generally needed values:
        assert (
            'length' in kwargs
            and isinstance(kwargs['length'], (int, float))
            and kwargs['length'] > 0
        ), err_len
        assert (
            'width' in kwargs
            and isinstance(kwargs['width'], (int, float))
            and kwargs['width'] > 0
        ), err_wdt
        assert (
            'port_diameter' in kwargs
            and isinstance(kwargs['port_diameter'], (int, float))
            and kwargs['port_diameter'] > 0
        ), err_pd
        if 'channel_area' in kwargs:
            assert (
                isinstance(kwargs['channel_area'], (int, float))
                and kwargs['channel_area'] > 0
            ), err_a

        self.length = float(kwargs['length'])
        self.width = float(kwargs['width'])
        self._d_port = float(kwargs['port_diameter'])
        # calculate flow length:
        self.length_flow = self.length * self.passes

        # assert and get corner radius
        assert (
            'corner_radius' in kwargs
            and isinstance(kwargs['corner_radius'], (int, float))
            and kwargs['corner_radius'] >= 0
        ), err_cr
        self._r_corner = float(kwargs['corner_radius'])

        # get heat exchanger plate area per plate, not including passes:
        if 'A_plate' not in kwargs:  # if not given
            self.A_plate = __plate_area(
                self.length, self.width, self._d_port, self._r_corner
            )
        else:  # if given
            self.A_plate = float(kwargs['A_plate'])
        # get heat exchanger effective area per plate, including passes:
        self.A_plate_eff = self.A_plate * self.passes

        # get depth of hex either directly or by number of plates and plate
        # specific depth:
        assert (
            'depth' in kwargs
            and isinstance(kwargs['depth'], (int, float))
            and kwargs['depth'] > 0
        ) or (
            'depth_base' in kwargs
            and 'depth_per_plate' in kwargs
            and isinstance(kwargs['depth_base'], (int, float))
            and isinstance(kwargs['depth_per_plate'], (int, float))
            and kwargs['depth_base'] > 0
            and kwargs['depth_per_plate'] > 0
        ), err_dpt
        if 'depth_base' in kwargs:  # base depth and depth per plate given
            self._depth_base = float(kwargs['depth_base'])
            self._diff_depth = float(kwargs['depth_per_plate'])
            self.depth = self._depth_base + self._diff_depth * self.num_plates
        else:  # only depth given
            self.depth = float(kwargs['depth'])

        # all other calcs
        if 'channel_width' in kwargs:  # preferred method: by channel width
            # this is the preferred method, since all critical parameters are
            # given directly or require a short calculation path.
            # assert that channel width is correct
            assert (
                isinstance(kwargs['channel_width'], (int, float))
                and kwargs['channel_width'] > 0
            ), err_sch
            self._s_channel = float(kwargs['channel_width'])
            # get channel cross section area:
            if 'channel_area' in kwargs:
                self.A_channel = float(kwargs['channel_area'])
            else:
                self.A_channel = self._s_channel * self.width
            # get plate thickness
            if 'plate_thickness' not in kwargs:
                assert (
                    'depth_per_plate' in kwargs
                    and isinstance(kwargs['depth_per_plate'], (int, float))
                    and kwargs['depth_per_plate'] > 0
                ), err_dd
                self._diff_depth = float(kwargs['depth_per_plate'])
                self._s_plate = self._diff_depth - self._s_channel
            else:
                assert (
                    isinstance(kwargs['plate_thickness'], (int, float))
                    and kwargs['plate_thickness'] > 0
                ), err_sp
                self._s_plate = float(kwargs['plate_thickness'])
            # get channel fluid volume:
            if 'V_channel' in kwargs:
                assert (
                    isinstance(kwargs['V_channel'], (int, float))
                    and kwargs['V_channel'] > 0
                ), err_Vfl
                self.V_ch_fld_tot = float(kwargs['V_channel'])
            else:  # v channel not given -> get it from area and width
                self.V_ch_fld_tot = self._s_channel * self.A_plate
        elif 'mass_per_plate' in kwargs:  # second preferred method
            # this is preferred over gving v channel, since v channel is barely
            # needed for calculations and thus the error is small when
            # calculating it.
            # assert that needed parameters are passed and correct:
            assert (
                'depth_per_plate' in kwargs
                and isinstance(kwargs['depth_per_plate'], (int, float))
                and kwargs['depth_per_plate'] > 0
            ), err_dd
            assert (
                isinstance(kwargs['mass_per_plate'], (int, float))
                and kwargs['mass_per_plate'] > 0
            ), err_dm
            # get values
            self._diff_m = float(kwargs['mass_per_plate'])
            self._diff_depth = float(kwargs['depth_per_plate'])
            # get plate thickness
            if 'plate_thickness' not in kwargs:
                self._s_plate = __plate_thickness(
                    self._diff_m,
                    self._rho_wll,
                    self.A_plate,
                    self._diff_depth,
                    self.length,
                    self.width,
                    self._r_corner,
                )
            else:
                assert (
                    isinstance(kwargs['plate_thickness'], (int, float))
                    and kwargs['plate_thickness'] > 0
                ), err_sp
                self._s_plate = float(kwargs['plate_thickness'])
            # get channel thickness
            self._s_channel = __channel_width_by_plate_thickness(
                self._diff_depth, self._s_plate
            )
            # get channel cross section area:
            if 'channel_area' in kwargs:
                self.A_channel = float(kwargs['channel_area'])
            else:
                self.A_channel = self._s_channel * self.width
            # get total channel fluid volume:
            if 'V_channel' not in kwargs:  # if not given
                self.V_ch_fld_tot = (
                    __plate_area(  # plate area without ports
                        self.length, self.width, 0, self._r_corner
                    )
                    - 2 * self._d_port ** 2 * np.pi / 4  # minus 2 ports
                ) * self._s_channel  # times channel width
            else:  # if given
                self.V_ch_fld_tot = float(kwargs['V_channel'])
        elif 'V_channel' in kwargs:  # least preferred method
            # this is the least preferred method, since most critical
            # parameters are NOT given directly or require a long calculation
            # path.
            # assert that needed parameters are passed and correct:
            assert (
                isinstance(kwargs['V_channel'], (int, float))
                and kwargs['V_channel'] > 0
            ), err_Vfl
            # get values
            self.V_ch_fld_tot = float(kwargs['V_channel'])
            # get channel width
            self._s_channel = __channel_width_by_volume(
                self.V_ch_fld_tot, self.length, self.width, self._r_corner
            )
            # get plate thickness
            if 'plate_thickness' not in kwargs:
                assert (
                    'depth_per_plate' in kwargs
                    and isinstance(kwargs['depth_per_plate'], (int, float))
                    and kwargs['depth_per_plate'] > 0
                ), err_dd
                self._diff_depth = float(kwargs['depth_per_plate'])
                self._s_plate = self._diff_depth - self._s_channel
            else:
                assert (
                    isinstance(kwargs['plate_thickness'], (int, float))
                    and kwargs['plate_thickness'] > 0
                ), err_sp
                self._s_plate = float(kwargs['plate_thickness'])
            # get channel cross section area:
            if 'channel_area' in kwargs:
                self.A_channel = float(kwargs['channel_area'])
            else:
                self.A_channel = self._s_channel * self.width
        else:
            # if none of the required parameters have been passed, raise an
            # error:
            err_str = (
                self._base_err
                + self._arg_err
                + 'To define the plate heat exchanger geometry, one of the '
                'following arguments has to be given:\n'
                '    - `channel_width=X`: The channel width of the plate '
                'heat exchanger (distance between two plates) in [m], where '
                'X is a float or integer value >0. Giving the channel width '
                'is highly recommended over giving the channel volume of mass '
                'per plate.\n'
                '    - `mass_per_plate=X`: The mass of a single heat '
                'exchanger plate (the increase in mass of the heat exchanger '
                'per plate) in [kg], where X is a float or integer value >0. '
                'This method is less recommended than giving the channel '
                'width, but should be preferred over giving the channel '
                'volume.\n'
                '    - `V_channel=X`: The total fluid channel volume '
                '**per channel** in [m^3], as given in the datasheet, where '
                'X is a float or integer value >0.\n'
                '\n'
                'In all cases passing the channel cross section area in [m^2] '
                'with `channel_area=X`, where X is an integer or float value '
                '>0, additionally increases the accuracy of the solution. '
                'Otherwise this will be calculated from the width which may '
                'or may not include the casing.'
            )
            assert False, err_str.format(
                'channel_width OR mass_per_plate OR V_channel AND '
                'channel_area'
            )

        # get hex mass:
        if 'mass_base' in kwargs:
            assert 'mass_per_plate' in kwargs, err_dm
            self._m_wll = (
                float(kwargs['mass_base']) + self._diff_m * self.num_plates
            )
        else:
            assert 'mass_total' in kwargs, err_mw
            self._m_wll = float(kwargs['mass_total'])

        # calculate other general things:
        # hydr. diameter for thin channels (VDI Wärmeatlas G2 2.1 eq(39)):
        self._d_h = self._s_channel * 2
        # outer surface area for heat losses:
        self._A_o = (  # without insulation
            __plate_area(self.length, self.width, 0, self._r_corner) * 2
            + (
                2 * self.length
                + 2 * self.width
                - 8 * self._r_corner
                + self._r_corner * 2 * np.pi
            )
            * self.depth
        )
        self._A_o_ins = __plate_area(  # with insulation
            self.length + 2 * self._s_ins,
            self.width + 2 * self._s_ins,
            0,
            self._r_corner,
        ) * 2 + (
            2 * (self.length + 2 * self._s_ins)
            + 2 * (self.width + 2 * self._s_ins)
            - 8 * self._r_corner
            + self._r_corner * 2 * np.pi
        ) * (
            self.depth + 2 * self._s_ins
        )
        # relation of both for heat loss calculation of alpha_inf:
        self._A_o_rel = self._A_o_ins / self._A_o
        # total effective heat exchanging area (independent of passes):
        self.A_hex_total = self.A_plate * (self.num_plates - 2)
        # shortest distance from center of HEX to the outside for von Neumann:
        self._dist_min = (self.width + 2 * self._s_ins) / 2
        # get mcp wall:
        self._mcp_wll = self._m_wll * self._cp_wll

    def find_Reynolds_correction(self, *, T_in, T_out, P, alpha_target, side):
        # check sides:
        err_str = (
            'To find the Reynolds correction number, the side of '
            'the heat exchanger corresponding to the given '
            'temperatures and power has to be given with '
            '`side=\'supply\'` or `side=\'demand\'`.'
        )
        assert type(side) == str and (
            side == 'supply' or side == 'demand'
        ), err_str
        if side == 'supply':
            num_channels = self._num_channels_sup
        else:
            num_channels = self._num_channels_dmd
        # check for power:
        if type(P) == np.ndarray:
            err_str = (
                'If `P` was passed as an array with different values '
                '`alpha_target` also has to be an array with the same '
                'shape.'
            )
            assert type(alpha_target) == type(P), err_str
            assert alpha_target.shape == P.shape, err_str
            # get shape for all arrays:
            shape = P.shape
        else:
            # set shape to one
            shape = 1
        # get mean temperature for fluid properties:
        T_mean = (T_in + T_out) / 2
        # get fluid properties:
        lam_mean = np.zeros(shape)
        cp_mean = np.zeros(shape)
        rho_mean = np.zeros(shape)
        ny_mean = np.zeros(shape)
        _pf.get_lambda_water(T_mean, lam_mean)
        _pf.get_cp_water(T_mean, cp_mean)
        _pf.get_rho_water(T_mean, rho_mean)
        _pf.get_ny_water(T_mean, ny_mean)
        # get massflow:
        dm = abs(P / ((T_in - T_out) * cp_mean))
        #        # get temperature without wall correction (same temperature for wall
        #        # and fluid) and reshape to correct shape:
        #        T = np.reshape(np.array([T_mean, T_mean]), (1, 2))
        # save alpha to:
        alpha = np.zeros_like(dm)
        start_value = np.ones(1)  # NEVER set this to zero!!!

        from scipy.optimize import minimize  # ,  # newton

        # wrapper function around phex alpha to feed this into minimizer:
        def alpha_wrapper(
            corr_Re,  # alpha_target,
            #  dm, T, rho_mean, ny_mean, lam_mean, A_channel,
            #  d_h, cell_dist, alpha
        ):
            # get alpha value
            _pf.phex_alpha_i_wll_sep(
                dm / num_channels,
                T_mean,
                T_mean,
                rho_mean,
                ny_mean,
                lam_mean,
                self.A_channel,
                self._d_h,
                self._cell_dist,
                corr_Re,
                alpha,
            )
            # get sum of deviations to alpha target:
            return abs(alpha / alpha_target - 1).sum()

        res = minimize(alpha_wrapper, start_value)
        self._corr_Re = res['x']
        print(
            'Reynolds correction was set to: '
            + str(res['x'])
            + ' with the sum of deviation(s) to the `alpha_target value(s)` '
            'of: ' + str(res['fun'])
        )

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

        # check if this side was already solved:
        for sp in self._solved_ports:  # loop over solved ports
            if kwargs['flow_channel'] in sp:  # if found side is already solved
                src_part = self.name  # replace source part with self
                src_idx = self._get_arr_idx(  # replace src idx with solvedport
                    self.name,
                    sp,
                    target='massflow',
                    as_slice=True,
                    flat_index=True,
                )
                alg_sign = 'positive'  # get same value as in solved port

        # For heat exchaners the operation id is always 0 or -1, depending on
        # the algebraic sign of the ports which are connected. This means the
        # value of the connected port is either passed on (0) or the negative
        # value of it is used (-1).

        # get operation id depending on the direction of positive massflow
        # through the connected ports:
        if alg_sign == 'positive':
            # if positive, pass on values:
            operation_id = 0
        else:
            # if negative, a negative copy has to be made
            operation_id = -1

        # add operation instructions to tuple (memory view to target
        # massflow array cell, operation id and memory view source port's
        # massflow array cells):
        op_routine = (
            # memory view to target massflow
            self._dm_io.reshape(-1)[trgt_idx],
            operation_id,  # add operation id
            # memview to source
            self._models.parts[src_part]._dm_io.reshape(-1)[src_idx],
        )

        # update solved ports list and counter stop break:
        self._solved_ports.append(port)
        self._cnt_open_prts = self.port_num - len(self._solved_ports)
        # this stays always False for HEX!
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

        return op_routine  # , op_routine2

    def solve(self, timestep):
        #        if self.name == 'Hex_Zirk':
        #            self.A_plate_eff = self.A_hex_total
        _pf.solve_platehex(
            T=self.T,
            T_port=self._T_port,
            T_mean=self._T_mean,
            ports_all=self.ports_all,
            res=self.res,
            dm_io=self._dm_io,
            dm_port=self._dm_port,
            cp_mean=self._cp_mean,
            lam_mean=self._lam_mean,
            rho_mean=self._rho_mean,
            ny_mean=self._ny_mean,
            lam_wll=self._lam_plate,
            alpha_i=self._alpha_i,
            UA_fld_wll=self._UA_fld_wll,
            UA_fld_wll_fld=self._UA_fld_wll_fld,
            port_own_idx=self._port_own_idx,
            port_link_idx=self._port_link_idx,
            A_plate_eff=self.A_plate_eff,
            A_channel=self.A_channel,
            d_h=self._d_h,
            s_plate=self._s_plate,
            cell_dist=self._cell_dist,
            num_A=self._num_A,
            num_channels_sup=self._num_channels_sup,
            num_channels_dmd=self._num_channels_dmd,
            corr_Re=self._corr_Re,
            stepnum=self.stepnum,
        )

    def draw_part(self, axis, timestep, draw, animate=False):
        return
