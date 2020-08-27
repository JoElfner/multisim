# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Nov 2019
"""

import numpy as np
import sklearn as _skl

from ..simenv import SimEnv
from ..precomp_funs import (
    make_poly_transf_combs_for_nb,
    extract_pca_results,
    condensing_hex_solve,
)


class HEXCondPoly(SimEnv):
    r"""
    type: Plate HEX for water/flue gas heat transfer with condensation.

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

        self.constr_type = 'HEXCondPoly'  # define construction type
        base_err = (  # define leading base error message
            'While adding {0} `{1}` to the simulation '
            'environment, the following error occurred:\n'
        ).format(self.constr_type, str(name))
        arg_err = (  # define leading error for missing/incorrect argument
            'Missing argument or incorrect type/value: {0}\n\n'
        )
        self._base_err = base_err  # save to self to access it in methods
        self._arg_err = arg_err  # save to self to access it in methods

        # assert that all required arguments have been passed
        # regression pipeline
        err_rp = (
            self._base_err
            + self._arg_err.format('regression_pipeline')
            + 'The scikit-learn regression pipeline must be given.'
        )
        assert 'regression_pipeline' in kwargs and isinstance(
            kwargs['regression_pipeline'], _skl.pipeline.Pipeline
        ), err_rp
        self._linreg_mdl = kwargs['regression_pipeline']
        # flow scaling reference value (dV gas in Nm3/h and dm water in kg/s)
        err_fsg = (
            self._base_err
            + self._arg_err.format('max_gas_Nm3h')
            + 'The maximum flue gas flow (wet) in m^3/h at standard conditions '
            '(often described as Nm3/h) must be given.\n\n'
            '**ATTENTION**: The solve function of the condensing HEX '
            'internally expects a fluegas flow in the unit of [Nm3/s]! '
            '`max_gas_Nm3h` is in [Nm3/h] for convenience, since most '
            'datasheets of CHP-plants or boilers supply this value. '
            'Remember to convert the input to the HEX to Nm3/s, **IF** it is '
            'detemined by a supplementary part such as a pump.\n\n'
            'Furthermore, depending on'
            'regression polynome, the maximum water mass flow in [kg/s] has '
            'to be scaled accordingly. F.i. for the included regression '
            'polynome, the maximum water mass flow is `.1kg/s per 70Nm3/h` '
            'resp. `max_water_kgs=1.42857e-3` kg/s per Nm3/h. For a CHP with '
            '70Nm3/h, this results in a maximum water mass flow of .1kg/s. '
            'Slight exceedence of up to 10% will likely still yield good '
            'results, but has not been validated.'
        )
        assert (
            'max_gas_Nm3h' in kwargs
            and isinstance(kwargs['max_gas_Nm3h'], (int, float))
            and (kwargs['max_gas_Nm3h'] > 0.0)
        ), err_fsg
        self._flow_scaling = np.ones(2, dtype=np.float64)  # prealloc array
        self._flow_scaling[1] = kwargs['max_gas_Nm3h']
        if 'max_water_kgs' in kwargs:
            assert (
                'max_water_kgs' in kwargs
                and isinstance(kwargs['max_water_kgs'], (int, float))
                and (kwargs['max_water_kgs'] > 0.0)
            ), err_fsg
            self._flow_scaling[0] = (
                kwargs['max_water_kgs'] * self._flow_scaling[1]
            )
        else:  # else set default value
            self._flow_scaling[0] = (
                1.4285714285714285e-3 * self._flow_scaling[1]
            )
        # convert gas scaling to Nm3/s to allow for low flow values (increases
        # numeric stability if any kind of replacement pump is used and also
        # keeps both values roughly in the same order of magnitude):
        self._flow_scaling[1] /= 3600
        # flow ranges
        err_fgr = (
            self._base_err
            + self._arg_err.format('fluegas_flow_range')
            + 'The flue gas flow range in percentage has to be given, f.i. '
            '(.5, 1.) means that all values from 50% to 100% of the maximum '
            'gas flow are valid. To avoid collision with interpolation of '
            'boundary condition timeseries, a tolerenace should be included. '
            'A value of 10% at the lower bound and 5% at the upper bound is '
            'recommended, f.i. resulting in `(.4, 1.05)`.'
        )
        assert (
            'fluegas_flow_range' in kwargs
            and isinstance(kwargs['fluegas_flow_range'], (tuple, list))
            and (kwargs['fluegas_flow_range'][0] >= 0.0)
            and (
                kwargs['fluegas_flow_range'][0]
                <= kwargs['fluegas_flow_range'][1]
            )
        ), err_fgr
        self._gas_dv_range = np.empty(2, dtype=np.float64)  # prealloc array
        self._gas_dv_range[:] = kwargs['fluegas_flow_range']
        err_wr = (
            self._base_err
            + self._arg_err.format('water_flow_range')
            + 'The water flow range in percentage has to be given, f.i. '
            '(0., 1.) means that all values from 0% to 100% of the maximum '
            'water mass flow are valid. To avoid collision with interpolation '
            'of boundary condition timeseries, a tolerenace should be '
            'included. A value of 10% at the lower bound (if >0) and 5% at '
            'the upper bound is recommended, f.i. resulting in `(0., 1.05)`.'
        )
        assert (
            'water_flow_range' in kwargs
            and isinstance(kwargs['water_flow_range'], (tuple, list))
            and (kwargs['water_flow_range'][0] >= 0.0)
            and (
                kwargs['water_flow_range'][0] <= kwargs['water_flow_range'][1]
            )
        ), err_wr
        self._water_dm_range = np.empty(2, dtype=np.float64)  # prealloc array
        self._water_dm_range[:] = kwargs['water_flow_range']

        # set number of gridpoints to 4:
        self.num_gp = 4

        # %% start part construction:
        #        super().__init__()
        self.name = name
        self.part_id = self._models.num_parts - 1
        # save smallest possible float number for avoiding 0-division:
        self._tiny = self._models._tiny

        # %% preallocate temperature and property grids:

        # preallocate regression X array. must be 2D, thus reshape.
        # order of the 4 values: t_rf_water, t_fg_in, dm_water_scld, dv_fg_scld
        # scld values are scaled to a 0-1 range
        self._X_pred = np.zeros((4,)).reshape(1, -1)
        # generate views to each cell
        self._T_w_in = self._X_pred[:, slice(0, 1)]
        self._T_fg_in = self._X_pred[:, slice(1, 2)]
        self._dm_w_scld = self._X_pred[:, slice(2, 3)]
        self._dv_fg_scld = self._X_pred[:, slice(3, 4)]

        # preallocate temperature grid with index (i, j) where i=0 is top side
        # (sup in, dmd out) and i=1 is the bottom side (sup out, dmd in)
        # and j is the index for the flow sides with j=0 sup and j=1 dmd:
        self.T = np.zeros((2, 2), dtype=np.float64)

        # self.T = np.zeros(4, dtype=np.float64)
        Tshp = self.T.shape  # save shape since it is needed often
        self._T_init = np.zeros(Tshp)  # init temp for resetting env.
        # views to supply and demand side:
        self._T_sup = self.T[:, 0]  # view to supply side
        self._T_dmd = self.T[:, 1]  # view to demand side

        # preallocate temperature grid for ports:
        # self._T_port = np.zeros_like(self.T)

        self._T_port = np.zeros(4)
        Tpshp = self._T_port.shape  # save shape since it is needed often

        # preallocate mass flow grids:
        self.dm = np.zeros(2)

        self._dm_port = np.zeros(Tpshp)
        self._dm_io = np.zeros(2)  # array for I/O flow, one cell per channel
        # views for each side
        self._dm_sup = self._dm_io[:1]  # view to supply side
        self._dm_dmd = self._dm_io[1:]  # view to demand side

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

        # port definition:
        self.port_num = 4
        # =============================================================================
        #         self._port_own_idx = np.array(
        #                 [0, 6, 2, 8], dtype=np.int32)  # flat 2d index to value array
        # =============================================================================
        # Index to own value array to get values of own ports, meaning if I
        # index a FLATTENED self.T.flat with self._port_own_idx, I need to
        # get values accoring to the order given in self.port_names.
        # That is, this must yield the value of the cell in self.T, which is
        # belonging to the port 'in':
        # self.T.flat[self._port_own_idx[self.port_names.index('in')]]
        self._port_own_idx = np.array([0, 2, 1, 3], dtype=np.int32)
        # =============================================================================
        #         self._port_own_idx_2D = np.array(
        #                 [0, 6, 2, 8], dtype=np.int32)  # flat 2d index to port array
        # =============================================================================
        self._port_own_idx_2D = np.array([0, 2, 1, 3], dtype=np.int32)
        self.port_ids = np.array((), dtype=np.int32)
        # define flow channel names for both sides. used in topology constr.
        self._flow_channels = ('water', 'fluegas')
        # define port_names
        self.port_names = tuple(
            ('water_in', 'water_out', 'fluegas_in', 'fluegas_out')
        )
        # and use slightly different definitions for saving to a pandas
        # dataframe, since the required reshaping changes the order
        self._res_val_names = tuple(
            ('water_in', 'fluegas_in', 'water_out', 'fluegas_out')
        )
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
        # =============================================================================
        #         self._trnc_err_cell_weight = np.zeros(Tshp)
        # =============================================================================
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

    def init_part(self, **kwargs):
        """Initialize part. Do stuff which requires built part dict."""
        # check if kwargs given:
        err_str = (
            'No parameters have been passed to the `init_part()` method'
            ' of heat exchanger `' + self.name + '`! To get '
            'additional information about which parameters to pass, '
            'at least the insulation thickness in [m] has to be given '
            '(0 is also allowed) with `insulation_thickness=X`.'
        )
        assert kwargs, err_str

        # assert material and pipe specs (after defining port names to avoid
        # name getting conflicts):
        err_str = (
            '`pipe_specs` and `material` for '
            + self.constr_type
            + ' `'
            + self.name
            + '` must be passed to its `init_part()` method! '
            'Pipe specifications define the connection parameters to other '
            'parts for heat conduction, thus are also required for parts that '
            'are not a pipe.'
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
        if not isinstance(self._T_init, np.ndarray):
            self.T[:] = float(self._T_init)
            self._T_init = float(self._T_init)
        else:
            # assert that correct shape is given and if yes, save to T
            assert self._T_init.shape == self.T.shape, err_str
            self.T[:] = self._T_init

        # get ambient temperature:
        self._chk_amb_temp(**kwargs)

        # extract regression values:
        (
            self._poly_int_comb_idx,
            self._poly_nvars_per_ftr,
        ) = make_poly_transf_combs_for_nb(
            self._linreg_mdl, pipeline=True, poly_step='polynomialfeatures'
        )
        self._pca_mean, self._pca_components = extract_pca_results(
            self._linreg_mdl, pipeline=True, pca_step='pca'
        )
        self._lm_intercept, self._lm_coef = (  # extract lin mdl fit coefs.
            self._linreg_mdl.steps[-1][1].intercept_,
            self._linreg_mdl.steps[-1][1].coef_,
        )

        # construct list of differential input argument names IN THE CORRECT
        # ORDER!!!
        # regex to remove strings: [a-zA-Z_]*[ ]*=self.
        self._input_arg_names_sorted = [
            'T',
            '_T_port',
            'ports_all',
            'res',
            'res_dm',
            '_dm_io',
            '_dm_port',
            '_port_own_idx',
            '_port_link_idx',  # indices
            '_X_pred',
            '_flow_scaling',
            '_poly_int_comb_idx',
            '_poly_nvars_per_ftr',
            '_pca_mean',
            '_pca_components',
            '_lm_intercept',
            '_lm_coef',
            'stepnum',  # step information
        ]

        # set initialization to true:
        self.initialized = True

    def _reset_to_init_cond(self):
        self.T[:] = self._T_init

    def _get_flow_routine(
        self, port, parent_port=None, subnet=False, **kwargs
    ):
        """
        Define massflow calculation routine.

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

    def _check_results(self):
        """Check results for validity."""
        assert np.all(self.res > 0.0) and np.all(
            self.res < 150
        ), 'Temperature range exceeding 0. < theta < 150.'
        assert np.all(
            self.res_dm >= 0.0
        ), 'Invalid water mass flow or flue gas volume flow < 0. found.'
        # not used, since clipping applied, thus checking is invalid
        # assert np.all(self.res_dm[:, 0] < 1.1), (
        #     'Invalid water mass flow > 1.1*max_water_kgs found.')
        assert (
            (self.res_dm[:, 1].max() / self._flow_scaling[1])
            < self._gas_dv_range[1]
        ), 'Invalid flue gas volume flow > fluegas_flow_range found.'

    def solve(self, timestep):
        """Solve the polynome function."""
        condensing_hex_solve(
            T=self.T,
            T_port=self._T_port,
            ports_all=self.ports_all,
            res=self.res,
            res_dm=self.res_dm,
            dm_io=self._dm_io,
            dm_port=self._dm_port,
            port_own_idx=self._port_own_idx,
            port_link_idx=self._port_link_idx,
            X_pred=self._X_pred,
            flow_scaling=self._flow_scaling,
            water_dm_range=self._water_dm_range,
            gas_dv_range=self._gas_dv_range,
            int_comb_idx=self._poly_int_comb_idx,
            nvars_per_ftr=self._poly_nvars_per_ftr,
            pca_mean=self._pca_mean,
            pca_components=self._pca_components,
            lm_intercept=self._lm_intercept,
            lm_coef=self._lm_coef,
            stepnum=self.stepnum,
        )

    def draw_part(self, axis, timestep, draw, animate=False):
        return
