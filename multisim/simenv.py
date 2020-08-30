# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Sep 2017
"""

from collections import OrderedDict as _odict
from collections import namedtuple as _namedtuple
from functools import partial as _partial
import math as _math
import os
import sys
import time as _time_
import traceback as _traceback
import numpy as np
import pandas as pd
import psutil as _psutil
from scipy.optimize import root as _root  # , fixed_point

from .flownet import FlowNet as _FlowNet
from . import precomp_funs as _pf
from .version import version as __version__

__all__ = ['SimEnv']


class SimEnv:
    """
    Simulation model environment.

    Create instance to start building an environment.
    """

    def __init__(self, suppress_printing=False):
        # check variable if simulation environment has been intialized.
        # WARNING: THIS DOES NOT MEAN, THAT THE INSTANCE IS NOT INITIALIZED!
        self._initialized = False
        self._solver_set = False
        self._timeframe_set = False
        self._disksaving_set = False

        # if printing of (most) info should be suppressed:
        self.suppress_printing = suppress_printing

        # get own instance name:
        (
            filename,
            line_number,
            function_name,
            text,
        ) = _traceback.extract_stack()[-2]
        def_name = text[: text.find('=')].strip()
        self._simenv_name = def_name

        # create path to subfolder:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sub_pth = os.path.join(current_dir, 'data_tables')
        #        self._mprop_w = np.load('\data_tables\StoffdatenWasser.npy')
        # MATERIAL PROPERTIES CAN ALSO BE USED FROM DORTMUND DATA BANK OR
        # INDUSTRIAL DIPPR-801 EQUATIONS!
        self._pipe_specs = pd.read_pickle(  # abspath for more robust import
            os.path.abspath(sub_pth + r'/pipe_specs.pkl')
        )
        self._mprop_sld = pd.read_pickle(
            os.path.abspath(sub_pth + r'/material_props.pkl')
        )
        self._hexdb = pd.read_pickle(
            os.path.abspath(sub_pth + r'/heat_exchangers.pkl')
        )

        usage_helper = (
            'Simulation environment {0} created.\n\n'
            'Next steps:\n'
            '    - set disk-saving with `{0}.set_savetodisk()` to save '
            'results to a pandas HDFStore. This must be set for simulations '
            'with many timesteps and numeric cells to avoid overflowing '
            'the memory. More than 50% of the available memory should never '
            'be in use. F.i. 24h simulation with 1s timestep and 100 parts '
            'with 100 cells each results in about 6.4GB of memory. To exclude '
            'specific parts/cells from disksaving to reduce hard disk drive '
            'memory consumption, set the `store_results` parameter for '
            'the `{0}.add_part()` method for relevant parts.\n'
            '    - set the timeframe with `{0}.set_timeframe()`\n'
            '    - set the solver with `{0}.set_solver()`\n'
            '    - add parts with `{0}.add_part()`\n'
            '    - if required, add constant boundary conditions with '
            '`{0}.add_open_port()`\n'
            '    - if required, convert a constant boundary condition to a '
            'time dependent dynamic BC with `{0}.assign_boundary_cond()`. '
            'This also works for controller setpoints.\n'
            '    - connect parts at ports and to boundary conditions with '
            '`{0}.connect_ports()`\n'
            '    - add controllers with `{0}.add_control()`\n'
            '    - initialize the simulation environment with '
            '`{0}.initialize_sim()`\n'
            '    - start the simulation with `{0}.start_sim()`\n'
            '    - once the simulation is finished, use the '
            '`utility_functions` module to add measurements/meters like heat '
            'meters by import the class `Meters` and/or plot the numeric '
            'errors of the simulation with the class `NetPlotter`\n'
            '    - access results and meters as a pandas DataFrame with '
            '`{0}.return_stored_data()` or load results from a previously '
            'completed simulation with `load_sim_results()` from the '
            '`utility_functions` module\n'
        ).format(self._simenv_name)
        if not self.suppress_printing:
            print(usage_helper)

        # define class attributes:
        # parts dict where added parts are stored. Will be filled by method
        # add_parts:
        self.parts = dict()
        # ports dict, stores values at the ports. Will be initialized by
        # methods add_part and add_ports. Values stored will be updated during
        # calculation by use of method update_ports:
        self.ports = dict()
        """ all changes to ports_all have a leading comment with 'ports_all'"""
        # ports array stores values at the ports. Will be initialized by
        # methods _add_part(), _add_open_port and any other method which adds
        # new ports (currently there are no other methods which add ports).
        # These methods change the array memory address, thus memory views
        # MUST NOT be assigned before all of these methods are finished!
        # Assigning memory views after _add_part() and _add_open_port() is
        # safe.
        self.ports_all = np.array(())
        # port_mapping dict stores each port with the key 'part;port' and the
        # value tuple('part', 'port', _port_own_idx). This allows fast access
        # from ports to parts without needing to do slow string-splitting.
        # Filled by add_part method. Used in update_FlowNet method (but only
        # the dict, not the array!)
        self._port_mapping = dict()
        # array version of port mapping for fast access:
        self._port_map_arr = np.empty([1, 4], dtype=np.int32)
        # ports connection dict. Will be filled by method connect_ports:
        self.port_links = dict()
        # array version of port links for fast access in loops:
        self.plinks_arr = np.empty([1], dtype=np.int32)
        # dictionary for boundary conditions, filled in add_open_port() and
        # assign_boundary_cond()
        self.boundary_conditions = dict()
        # tuple for dynamic boundary conditions and array for the time vector.
        # will be filled by assign_boundary_cond()
        self._dyn_BC = tuple()
        self._dyn_BC_timevec = np.array([])
        # solving dict for explicit numeric parts, which contains all parts
        # with their respective differential equations to solve within one
        # timestep:
        self.solve_num_expl = dict()
        # solving dict for implicit numeric parts, which contains all parts
        # with their respective differential equations to solve within one
        # timestep:
        self.solve_num_impl = dict()
        # solving dict for FULLY njitted numeric parts. Will be mutated into
        # tuple for solver iterations.
        self.solve_num_njit = dict()
        # dict for parts which will not be solved numerically, like pumps,
        # valves, etc...
        self.solve_nonnum = _odict()
        # dict for saving flow variables in net topology of primary superior
        # nets. Contains the class FlowNet and is created by get_topology
        # method and will be filled by recursive_topology method.
        # Starting/parent part is always a pump.
        self.sup_net = _odict()
        # dict for saving net topology of sub nets which do not have their own
        # pump or for sub nets which can't be solved in flow nets due to
        # missing information from ports. contains class instances of FlowNet
        # class.
        self.sub_net = _odict()
        # tuple for saving combined net topology which contains the primary
        # superior nets AND the secondary sub nets in one long tuple. The
        # solving order will be the order detected by get topology and
        # recursive topology:
        self.flow_net = tuple()
        # set which contains all parts which are NOT solved by FlowNet! Most
        # probably this are going to be only hydraulic compensators like TES.
        # Created and filled by get_topology method and finalized (removing
        # parts) by recursive_topology method:
        self._hydr_comps = set()
        # list which contains all controls. The controls are also referenced
        # in the dict ctrls for easy user-interface access:
        self.ctrl_l = list()
        self.ctrls = dict()
        # list which contains all actuators. filled in add_part:
        self._actuators = list()
        # current calculation step number (as array for referencing):
        self.stepnum = np.zeros(1, dtype=np.int32)
        # total step number, including resets of the current step number due to
        # saving to disk etc.
        self._total_stepnum = 0
        # number of added parts:
        self.num_parts = 0
        # number of added ports:
        self.num_ports = 0
        # number of connected ports:
        self._num_conn_ports = 0
        # number of added controls:
        self.num_ctrls = 0
        # total number of cells in all numeric parts:
        self._num_cells_tot_nmrc = 0  # all numeric parts
        self._num_cells_tot_nmrc_expl = 0  # explicit parts
        self._num_cells_tot_nmrc_impl = 0  # implicit parts
        # total number of cells in all non numeric parts:
        self._num_cells_tot_nnmrc = 0
        # initialize time vector:
        self.time_vec = np.array([])
        # and total time vector to enable saving to disk:
        self._total_time_vec = np.array([])
        # maximum von Neumann stable step variable (as array for referecing):
        self._vN_max_step = np.array([np.inf])
        # save sensors for plotting to list:
        self.sensors = list()
        # show warnings for not optimal topology:
        self.dm_warn = True
        # dict to save functions to postpone in. These will be executed in
        # initialize_sim:
        self.__postpone = {}
        # dict for all part specific result disk-storage modes:
        self._disk_store_parts = {}
        # dict for utility information to store on disk, like control vars.
        self._disk_store_utility = {}
        # save smallest number to fill unused array cells with to avoid
        # 0-division, nan-errors and false results. np.finfo(float).tiny is
        # the smallest representable float number, using this tends to give
        # underflow errors during calculation. thus a security factor to this
        # number should be used to avoid denormal numbers:
        self._tiny = 1e-30

    def set_timeframe(self, *, timeframe, adaptive_steps, **kwargs):
        """
        Set the timeframe (duration) for the simulation.

        'timeframe' and 'timestep' must be given in [s] OR the number of
        steps to calculate 'num_steps' has to be given as an integer value.
        The remaining one must be set to 0. The value set to 0 will
        be calculated.

        Parameters
        ----------
        timeframe: int, float
            Total timedelta in [s] over which the calculation will run.
        adaptive_steps : bool
            If adaptive stepsizes with embedded Runge-Kutta methods shall be
            used.

        Returns
        -------
        None.

        """
        # get timeframe
        err_str = (
            '`timeframe` has to be a float or integer value giving the '
            'total simulation time in [s] greater than 0!'
        )
        assert isinstance(timeframe, (float, int)) and timeframe > 0, err_str
        self.timeframe = float(timeframe)

        # get parameters for adaptive/non-adaptive steps:
        err_str = (
            '`adaptive_steps` has to be a bool depicting if the solver should '
            'try to minimize the discretization error of the ODEs by using an '
            'embedded Runge-Kutta adaptive stepsize control algorithm.\n'
            'If set to True, the following additonal parameters can be '
            'given:\n'
            '        - `rtol=X` with 0 < X < 1 giving the relative error '
            'tolerance. As a rule of thumb it should at least be set to '
            '`rtol=10**(-(m+1))` where `m` is the number of relevant '
            'digits for the solution. If no value is given, the default '
            '`rtol=1e-3` will be used.\n'
            '        - `atol=X` with 0 < X < 1 giving the absolute error '
            'tolerance. As a rule of thumb it should at least be set to '
            'the first digit, which is insignificant for the solution. If '
            'no value is given, the default `atol=1e-6` will be used.\n'
            '        - `max_stepsize=X` with X > 0 and '
            'max_stepsize > min_stepsize giving the maximum stepsize in '
            '[s] to use for the simulation. If no value is given, the '
            'default of `max_stepsize=100` will be used.\n'
            '        - `min_stepsize=X` with X > 0 and '
            'max_stepsize > min_stepsize giving the minimum stepsize in '
            '[s] to use for the simulation. If no value is given, the '
            'default of `min_stepsize=1e-8` will be used, which is the '
            'smallest possible value to calculate a timeframe of one year '
            'while not falling below the machine epsilon. For one week '
            '1e-9 is sufficient.\n'
            '        - `max_factor=X` with X > 1 giving the maximum '
            'factor by which the timestep can be increased per iteration '
            'when the discretization error is low enough. If no value is '
            'given, the default `max_factor=2` will be used.\n'
            '        - `min_factor=X` with 0 < X < 1 giving the minimum '
            'factor by which the timestep can be decreased per iteration '
            'when the discretization error is too big. If using a controller '
            'with a derivative term, f.e. a PID, values of 0.5 or higher are '
            'recommended. If no D-terms are used, lower values are ok. If no '
            'value is given, the default `min_factor=0.2` will be used.\n'
            '        - `check_vN=bool` with the bool defining if the '
            'algorithm checks if the von Neumann stability conditions for '
            'conduction and advection are satisfied. This is HIGHLY '
            'recommended to avoid instable simulation steps! If not '
            'given, this will be automatically set to True.\n'
            '        - `factor_recovery=X` int, float with 0 < X, '
            'defines the number of consecutive successful steps until '
            'restrictions to the step-size control are lifted after '
            'stability conditions have been violated. This avoids increasing '
            'the step-size too fast when the system-wide error is low after '
            'a stability condition violation in one specific part. '
            'For `n<factor_recovery steps after a stability condition '
            'violation, the step-size increasing factor will be released '
            'gradually with `(n/factor_recovery)^2`. Default is 50.\n'
            '        - `max_speed=bool` defining if the maximum stepsize '
            'which is stable for all parts should be used. This is only '
            'recommended for calculating rough estimates and will be set '
            'to `max_speed=False` as default.\n'
            '        - `safety_factor=X` with 0 < X < 1 defining the '
            'safety factor which reduces all stepsize changes. A lower '
            'factor means more calculations when the differentials do not '
            'change, a higher factor vice-versa. In most cases this '
            'should not be changed! The default of '
            '`safety_factor=0.9` will be used and is a widely approved '
            'value.'
        )
        assert isinstance(adaptive_steps, bool), err_str
        # save values:
        self.adaptive_steps = adaptive_steps
        if adaptive_steps:
            if 'rtol' in kwargs:
                assert isinstance(kwargs['rtol'], float) and (
                    0 < kwargs['rtol'] < 1
                ), ('rtol not correct!\n\n' + err_str)
                self.__rtol = kwargs['rtol']
            else:
                self.__rtol = 1e-3
            if 'atol' in kwargs:
                assert isinstance(kwargs['atol'], float) and (
                    0 < kwargs['atol'] < 1
                ), ('atol not correct!\n\n' + err_str)
                self.__atol = kwargs['atol']
            else:
                self.__atol = 1e-6
            if 'max_stepsize' in kwargs:
                assert (
                    isinstance(kwargs['max_stepsize'], (int, float))
                    and kwargs['max_stepsize'] > 1e-8
                ), ('max_stepsize not correct!\n\n' + err_str)
                self.__max_stepsize = float(kwargs['max_stepsize'])
            else:
                self.__max_stepsize = 100.0
            if 'min_stepsize' in kwargs:
                assert (
                    isinstance(kwargs['min_stepsize'], (int, float))
                    and 0 < kwargs['min_stepsize'] < self.__max_stepsize
                ), ('min_stepsize not correct!\n\n' + err_str)
                self.__min_stepsize = float(kwargs['min_stepsize'])
            else:
                self.__min_stepsize = 1e-8
            if 'max_factor' in kwargs:
                assert (
                    isinstance(kwargs['max_factor'], (int, float))
                    and kwargs['max_factor'] > 1
                ), ('max_factor not correct!\n\n' + err_str)
                self._max_factor = np.array([float(kwargs['max_factor'])])
            else:
                self._max_factor = np.array([2.0])
            if 'min_factor' in kwargs:
                assert isinstance(kwargs['min_factor'], float) and (
                    0 < kwargs['min_factor'] < 1
                ), ('min_factor not correct!\n\n' + err_str)
                self.__min_factor = kwargs['min_factor']
            else:
                self.__min_factor = 0.2
            if 'check_vN' in kwargs:
                assert isinstance(kwargs['check_vN'], bool)(
                    'check_vN not of correct type!\n\n' + err_str
                )
                self._check_vN = kwargs['check_vN']
            else:
                self._check_vN = True
            if 'max_speed' in kwargs:
                assert isinstance(kwargs['max_speed'], bool), (
                    'max_speed not of correct type!\n\n' + err_str
                )
                self._max_speed = kwargs['max_speed']
            else:
                self._max_speed = False
            if 'safety_factor' in kwargs:
                assert isinstance(kwargs['safety_factor'], float) and (
                    0 < kwargs['safety_factor'] < 1
                ), ('safety_factor not correct!\n\n' + err_str)
                self.__safety = kwargs['safety_factor']
            else:
                self.__safety = 0.9
            # after a stability breach, recover full step increasing factor
            # after n steps:
            if 'factor_recovery' in kwargs:
                assert (
                    isinstance(kwargs['factor_recovery'], (int, float))
                    and kwargs['factor_recovery'] > 0
                ), ('factor_recovery not correct!\n\n' + err_str)
                self._fctr_rec_stps = float(kwargs['factor_recovery'])
            else:
                self._fctr_rec_stps = 50.0
            # incremental counter for consecutive successive steps since the
            # last failed step
            self._steps_since_fail = 0
            # also set starting timestep with some value:
            self._new_step = 1.0  # this is the adaptive step starting step
            self.timestep = 1.0  # this is the adaptive step final value
        else:
            # else if constant stepsizes are chosen
            err_str = (
                'If `adaptive_steps=False` is set, the constant '
                'stepsize in [s] has to be given with `timestep=X` '
                'with X > 0!'
            )
            assert (
                'timestep' in kwargs
                and isinstance(kwargs['timestep'], (float, int))
                and kwargs['timestep'] > 0
            ), err_str
            self.timestep = float(kwargs['timestep'])
            # set all other checks to true/false so that the simulation runs
            # smoothly:
            self._max_speed = False
            self._check_vN = False

        # set stepnum to 0 for the start:
        self.stepnum[0] = 0
        # initialize stable step bool checker to True (array for referencing):
        self._step_stable = np.array([True])

        # get an estimate of number of steps with timestep
        self.num_steps = int(np.ceil(timeframe / self.timestep))

        # backup kwargs for resetting environmnent:
        self.__bkp_kwargs_timeframe = kwargs

        # set checker for timeframe set to true:
        self._timeframe_set = True

    def set_disksaving(
        self,
        save=True,
        save_every_n_steps=100000,
        sim_name=None,
        path=r'.\results\\',
        start_date='2019-09-01 00:00:00',
        resample_final=True,
        resample_freq='1s',
        create_new_folder=True,
        overwrite=False,
        complevel=9,
        complib='zlib',
    ):

        self.__save_to_disk = save

        if not save:  # skip if false
            self._disksaving_set = True  # checker if options were treated
            self._save_every_n_steps = 1e6  # max steps without disksaving
            if not self.suppress_printing:
                print('No disksaving set. Maximum no. of steps set to 1e6.')
            return

        sim_name = self._simenv_name if sim_name is None else sim_name
        now = pd.datetime.now()

        self._save_every_n_steps = int(save_every_n_steps)

        self._disk_store = {}
        if start_date != 'infer':
            self._disk_store['start_date'] = pd.to_datetime(start_date)
            # also intermiadte val for disk saving steps
            self._disk_store['curr_step_start_date'] = self._disk_store[
                'start_date'
            ]
            self.__infer_start_date = False
        else:
            self.__infer_start_date = True
            self._disk_store['start_date'] = None
        # path to disk store
        self._disk_store['path'] = os.path.abspath(
            '{path}{iso_date}_{name}.{typ}'.format(
                path=path,
                iso_date=now.isoformat(timespec='seconds').replace(':', '-'),
                name=sim_name,
                typ='h5',
            )
        )

        # get resampling:
        assert isinstance(resample_final, bool)
        if resample_final:
            try:
                pd.Timedelta(resample_freq)
            except ValueError:
                raise ValueError(
                    'Invalid resample frequency `resample_freq={}`.'.format(
                        resample_freq
                    )
                )
        self._disk_store['resample'] = resample_final
        self._disk_store['resample_freq'] = resample_freq

        # set compression (applied during postprocessing after sim)
        self._disk_store['compress'] = True if complevel > 0 else False
        self._disk_store['complevel'] = complevel
        self._disk_store['complib'] = complib

        # check if folder exists:
        if not os.path.isdir(path):
            if not create_new_folder:
                raise FileNotFoundError('Path does not exist!')
            os.makedirs(path)
        # check if file already exists:
        if os.path.exists(self._disk_store['path']) and not overwrite:
            raise FileExistsError(
                'Storage file already existing! Change the file name or set '
                '`overwrite=True`'
            )

        # temporary store for fast uncompressed writing:
        self._disk_store['path_tmp'] = self._disk_store['path'] + '_tmp'
        # create store in context manager to automatically close it afterwards
        with pd.HDFStore(
            self._disk_store['path_tmp'], mode='w', complevel=0
        ) as store:
            self._disk_store['store_tmp'] = store

        # set checker to True
        self._disksaving_set = True

    def set_solver(self, solver, **kwargs):
        """
        Set the solver-method. The solver-method has to be passed to 'solver'
        as a string. The following solver-methods are currently supported:

        Supported solver-methods
        ------------------------
        Heun\'s Method: solver = 'heun'
            Sets the solver to use Heun\'s Method to solve the equation system.
        Runge-Kutta-4 Method: solver = 'rk4'
            Sets the solver to use Runge-Kutta-4 Method (classic
            Runge-Kutta Method) to solve the equation system.
        """

        # assert that the timeframe is already set:
        err_str = (
            'The timeframe of the simulation environmnet has the be '
            'set before setting the solver!'
        )
        assert self._timeframe_set, err_str
        # list of supported solvers: DEPRECATED! ONLY HEUN SUPPORTED NOW
        self._solvers = tuple(
            (
                '_heun',
                '_heun_adaptive',
                '_rk4',
                '_euler_expl_classic',
                '_euler_expl_modified',
                '_midpoint',
            )
        )

        assert solver == 'heun', (
            'Currently only heun solver is supported. Support for all other '
            'solvers has been deprecated. Major solver method refactoring '
            '(outsourcing of the solver steps to a method should be the first '
            'step to make a flexibile implementation with an arbitrary step '
            'number applicable) is required before implementing new solvers.'
        )

        # construct private method name:
        solver = '_' + solver

        # if adaptive, add adaptive part to name:
        if self.adaptive_steps:
            solver += '_adaptive'

        if solver not in self._solvers:
            err_str = (
                'Solver Method not found! Please check spelling or '
                'implement a new solver method.'
            )
            raise NameError(err_str)

        if solver == '_heun_adaptive':
            self.__order = 2
        elif solver == '_rk4(3)_adaptive':
            self.__order = 4
        elif solver == '_dormand_prince':
            self.__order = 5

        # store ref of the selected solver
        self.solver = getattr(self, solver)

        # check if implicit solving is allowed:
        err_str = (
            'The argument `allow_implicit=X` has to be given to '
            '`{0}.set_solver` method, where X is a bool.\n\n'
            'If it is set to True, parts may also be solved implicitly. This '
            'allows for much larger timesteps at the cost of a significant '
            'increase in computational cost and numeric error.\n'
            'Thus only parts with very stiff PDEs, like plate heat '
            'exchangers, should be solved implicitly, by passing '
            '`solve_implicit=True` to their respective `{0}.add_part()` '
            'method. This also introduces an '
            'increase of the numeric error of the solving method of these '
            'parts, as well as a loss of solver order.\n\n'
            'To check if parts may require implicit solving, plot the part '
            'specific numeric errors with '
            '`NetPlotter.plot_errors(simenv={0})` '
            'after fully explicitly solving a relevant range of timesteps. '
            'Parts with outstanding high numeric errors may be worth to be '
            'solved implicitly. But before solving parts implicitly, trying '
            'to decrease the number of grid points (increase the cell volume '
            'and thus the thermal capacity) is recommended.'
        ).format(self._simenv_name)
        assert 'allow_implicit' in kwargs and isinstance(
            kwargs['allow_implicit'], bool
        ), err_str
        self._allow_implicit = kwargs['allow_implicit']
        # set solver to set:
        self._solver_set = True

    def add_part(self, part, name, store_results=True, **kwargs):
        """
        Add a part to the simulation environment.

        Parameters
        ----------
        part : Class
            Reference to part class.
        name : str
            Name of the part.
        store_results : bool, int, slice, list, tuple, optional
            Store the results. True/False saves the results of all/none of the
            part's cells. Slice, int or list/tuple of integers selects specific
            cells of which the results are saved. The default is True.
        **kwargs : various
            Part specific construction argumentes. Each part will check for
            correct arguments and raise errors independently.

        """
        # check if add parts is already in postpone and if not, create it:
        if 'add_part' not in self.__postpone:
            self.__postpone['add_part'] = {}
            self.__postpone['init_part'] = {}  # same for init part

        # save gridpoints for checking and if no gp, set to dummy value:
        if 'grid_points' in kwargs:
            gps = kwargs['grid_points']
        else:
            gps = 1
            # overwrite store results, only allowing True or False
            store_results = True if store_results else False

        # check for correct store_results arg
        assert isinstance(store_results, (bool, int, slice, list, tuple)), (
            'While adding part `{0}`, argument `store_results` is not '
            'correct. Store results must be of type bool, int, slice or '
            'list/tuple.\n'
            '    - If bool, True or False results in saving all or no results '
            'to disk.\n'
            '    - If int, the integer will be used as index for the cell to '
            'store the values from.\n'
            '    - If slice, the values of the cells indexed by the slice '
            'will be stored.\n'
            '    - If list or tuple, values must be integer indices for the '
            'cells of which the values shall be stored'.format(name)
        )
        if isinstance(store_results, (bool, int, slice)):
            self._disk_store_parts[name] = store_results
            if isinstance(store_results, int) and not isinstance(
                store_results, bool
            ):
                assert store_results < gps, (
                    'Elements of `store_results` must be smaller than '
                    'the number of grid points of part `{0}`.'.format(name)
                )

        elif isinstance(store_results, (list, tuple)):
            for elmnt in store_results:
                assert isinstance(
                    elmnt, int
                ), 'Elements of `store_results` list/tuple must be integers.'
                assert elmnt < gps, (
                    'Elements of `store_results` list/tuple must be smaller '
                    'than the number of grid points of part `{0}`.'.format(
                        name
                    )
                )
            self._disk_store_parts[name] = store_results

        # check if part was not already added to postpone dict:
        self._check_isadded(part=name, kind='part')

        # postpone addition of parts to enable unordered sim. env.
        # construction (passing on self is needed, since some vars are saved in
        # the created SimEnv class instance!):
        self.__postpone['add_part'][name] = _partial(
            self._add_part, part, name, **kwargs
        )
        # postpone initialization of parts to enable unordered sim. env.
        # construction:
        self.__postpone['init_part'][name] = kwargs

    def _add_part(self, part, name, **kwargs):
        """
        Construct and add the part to the simulation environment.

        **DO NOT CALL THIS METHOD!** It is only meant to be used internally
        while assembling the simulation environment.

        Add a part to the calculation model by creating an instance of the
        specific part class inside the SimEnv class.

        """
        # assert that timeframe and solver are set before adding any parts:
        err_str = (
            'Before adding any parts to the simulation environment, the '
            'simulation timeframe and solver have to be set with '
            '`{0}.set_timeframe()` and `{0}.set_solver()` methods.'
        ).format(self._simenv_name)
        assert self._solver_set and self._timeframe_set, err_str
        # assert that name is not BC to avoid name conflict with boundary cond.
        err_str = (
            'The name of a part must not be \'BC\' to avoid a name '
            'conflict with boundary conditions!'
        )
        assert name != 'BC', err_str
        assert '/' not in name, (
            'The part name may not include \'BC\', \'\\\', \'/\'. The given '
            'name was {0}'.format(name)
        )

        # increase parts count by one:
        self.num_parts += 1

        # Add part to parts dict:
        self.parts[name] = part(name, self, **kwargs)

        # check if multiple ports have been added to the same cell (this
        # results in port index being a 2D-array with one array for each dim,
        # thus checking if array dimension >= 2) and then save the
        # 1D-version of it if true, else just use it as it is:
        if self.parts[name]._port_own_idx.ndim >= 2:
            # if len(self.parts[name]._port_own_idx) >= 2:
            # extract the 1D-indices from the last column of array:
            port_indices = self.parts[name]._port_own_idx[:, -1]
        else:
            port_indices = self.parts[name]._port_own_idx

        # get ports of part and save them to the containers which deal with
        # ports
        for i in range(len(self.parts[name].port_names)):
            port_key = name + ';' + self.parts[name].port_names[i]
            # preallocate ports dict array:
            self.ports[port_key] = 0
            # fill port_mapping dict with (name, port) tuples:
            self._port_mapping[port_key] = (
                name,
                self.parts[name].port_names[i],
                port_indices[i],
            )
            # construct global ids of ports
            self.parts[name].port_ids = np.append(
                self.parts[name].port_ids, self.num_ports
            )
            # append row to ports array, which can be accessed with port ids
            # where the port value will be stored in and increase port counter:
            self.ports_all = np.append(self.ports_all, 0.0)
            self.num_ports += 1
            # second column can be deleted as it is the same as the index, only
            # for safety checking reasons still in! TODO
            self._port_map_arr = np.vstack(
                [
                    self._port_map_arr,
                    [
                        self.parts[name].part_id,
                        self.parts[name].port_ids[i],
                        port_indices[i],
                        np.zeros(1, dtype=np.int32),
                    ],
                ]
            )
            # enlarge plinks_arr:
            self.plinks_arr = np.append(self.plinks_arr, 0)
        """ports_all version, remaining for backwards compatibility"""
        # delete first row which is only filled with random numbers
        if self.num_parts == 1:
            self._port_map_arr = self._port_map_arr[1:, :]
            self.plinks_arr = self.plinks_arr[1:]
        # end array_version

        # add differential function of numeric parts to the solving dict:
        if self.parts[name].solve_numeric:
            # only a few are fully jitted, these are saved separately
            # SET TO FALSE AS LONG AS NO PARALLEL ITERATION OVER HET TUPLES
            # IS POSSIBLE!!! (and False added)
            if (
                hasattr(self.parts[name], '_diff_fully_njit')
                and getattr(self.parts[name], '_diff_fully_njit')
                and False
            ):
                self.solve_num_njit[name] = self.parts[name]._diff_njit
                # save number of grid points/cells per part
                if self.parts[name].solve_explicitly:
                    # count total cells of explicit numeric parts:
                    self._num_cells_tot_nmrc_expl += self.parts[name].num_gp
                else:
                    # count total cells of implicit numeric parts:
                    self._num_cells_tot_nmrc_impl += self.parts[name].num_gp
            # for parts that need explicit solving (most if not all):
            elif self.parts[name].solve_explicitly:
                self.solve_num_expl[name] = self.parts[name].get_diff
                # count total cells of explicit numeric parts:
                self._num_cells_tot_nmrc_expl += self.parts[name].num_gp
            else:  # for implicit solving of some parts:
                self.solve_num_impl[name] = self.parts[name].diff_impl
                # count total cells of implicit numeric parts:
                self._num_cells_tot_nmrc_impl += self.parts[name].num_gp
            # count total cells of numeric parts:
            self._num_cells_tot_nmrc += self.parts[name].num_gp
        # for parts which are not to be solved numerically, add them to a
        # separate dict:
        elif not self.parts[name].solve_numeric:
            self.solve_nonnum[name] = self.parts[name].solve
            # count total cells of non numeric parts:
            self._num_cells_tot_nnmrc += self.parts[name].num_gp

        # add actuators to actuator list:
        if self.parts[name].is_actuator:
            self._actuators.append(name)

    def _add_ports(self, **kwargs):
        """
        Add new ports to a part.

        This method adds new ports to parts. This only works for specific
        parts, which differential method's can deal with variant flows.
        Furthermore all arrays dealing with port specifications must collapsed,
        meaning that their size exactly matches the number of ports at that
        part.
        Since several views are created after adding parts, this method **must
        be called from within the `add_part()` method.**
        """

        err_cap = (
            self._base_err
            + self._arg_err.format('can_add_ports')
            + 'The part does not support adding new ports.'
        )
        assert self.can_add_ports, err_cap

        # assert that new ports are given:
        err_str = (
            self._base_err
            + self._arg_err.format('new_ports')
            + 'The argument `new_ports=X` has to be given. If adding new ports '
            'is not required, pass `new_ports=None`.\n'
            'If new ports shall be added, X has to be a dict with the port '
            'name(s) as key(s) and the port position(s) and position type(s) '
            'in a list as value(s).\n'
            'The port position can be given as an integer cell index, '
            'or as a float/integer value giving the volume in [m^3] between '
            'the inlet port (index 0 of the value array) and the desired port '
            'location or the distance in [m] between the inlet port and the '
            'desired port location. The chosen kind has to be specified in '
            'the position type either as \'index\' or \'volume\' or '
            '\'distance\'. The resulting key:value-pair-dict has to look '
            'like:\n'
            '`new_ports={\'p1\': [25, \'index\'],\n'
            '            \'p2\': [0.8, \'volume\'],\n'
            '            \'p3\': [2.7, \'distance\']}`'
        )
        assert 'new_ports' in kwargs, err_str
        # get additional ports:
        if kwargs['new_ports'] is not None:
            assert isinstance(kwargs['new_ports'], dict), err_str
            # assert that sub-structure of new ports is composed of lists:
            for value in kwargs['new_ports'].values():
                assert isinstance(value, list), err_str
            # make a DEEP copy of the new ports dict to be able to alter it
            # without sending alterations to the calling functions (and thus
            # avoid errors due when the part-construction is called twice):
            nep_cpy = {key: val[:] for key, val in kwargs['new_ports'].items()}
            # construct dict of currently existing ports with the
            # port index as key and the number of occurences as value to
            # be able to calculate the number of extensions needed for new
            # ports:
            pidx, counter = np.unique(self._port_own_idx, return_counts=True)
            port_occurrence = dict(zip(pidx, counter))

            # loop over new ports. Value contains the new port's index at
            # element 0.
            for key, value in nep_cpy.items():
                # assert that correct position type is given:
                assert value[1] in ('index', 'volume', 'distance'), err_str
                # if port position is given as index:
                if value[1] == 'index':
                    # assert correct position:
                    assert isinstance(value[0], int), err_str
                    # assert that index exists:
                    err_str2 = (
                        self._base_err
                        + self._arg_err.format('new_ports')
                        + 'While adding new port `'
                        + key
                        + '` an error '
                        'occurred:\n'
                        'Index ' + str(value[0]) + ' is out of range for '
                        'the defined grid with '
                        + str(self.num_gp)
                        + ' grid points.'
                    )
                    assert -1 <= value[0] <= (self.num_gp - 1), err_str2
                    # if -1 was given (last element index), get the direct idx:
                    if value[0] == -1:
                        value[0] = self.num_gp - 1
                # if port position is given as volume, calculate index:
                elif value[1] == 'volume':
                    # assert correct type:
                    assert isinstance(value[0], (int, float)), err_str
                    err_str2 = (
                        self._base_err
                        + self._arg_err.format('new_ports')
                        + 'The given volume position at '
                        + str(value[0])
                        + 'm³ for port `'
                        + key
                        + '` must be within the '
                        'interval 0 <= volume_position <= '
                        + str(self._V_total)
                        + 'm³.'
                    )
                    # assert correct position/that volume element exists:
                    assert 0.0 <= value[0] <= self._V_total, err_str2
                    # get index of volume element where to place the port and
                    # save as integer index to value[0]
                    # ((self.V_cell / 2 - 1e-9) makes that the volume position
                    # is always rounded towards the position of the cell
                    # center!)
                    value[0] = round(
                        (value[0] - (self.V_cell / 2 - 1e-9)) / self.V_cell
                    )
                else:  # else if distance is given
                    # assert correct type:
                    assert isinstance(value[0], (int, float)), err_str
                    err_str2 = (
                        self._base_err
                        + self._arg_err.format('new_ports')
                        + 'The given distance position at '
                        + str(value[0])
                        + 'm for port `'
                        + key
                        + '` must be within the '
                        'interval 0 <= length <= ' + str(self.length) + 'm.'
                    )
                    assert 0.0 <= value[0] <= self.length, err_str2
                    # get index of cell where to place the port and save as
                    # integer index to value[0]
                    # round towards lower integer cell so that values on
                    # cell boundaries will be set to the cell with the lower
                    # index:
                    value[0] = round(
                        np.floor(value[0] / self.length * self.num_gp - 1e-9)
                    )
                    value[0] = value[0] if value[0] >= 0 else 0  # clip to zero
                # clip value to the range of cells to avoid erros resulting
                # from float and/or rounding errors:
                value[0] = (
                    self.num_gp - 1
                    if value[0] >= self.num_gp
                    else 0
                    if value[0] < 0
                    else value[0]
                )
                # check if a port at that position already exists:
                if value[0] in self._port_own_idx:
                    # increase counter of port in port_occurrence dict:
                    port_occurrence[value[0]] += 1
                # if port not yet existing, only add it to port_occurence dict,
                # everything else will be done in general in the next line:
                else:
                    port_occurrence[value[0]] = 1

                # append new cell to all arrays which have to cope with ports:
                self._T_port = np.append(self._T_port, 0.0)
                self._dm_port = np.append(self._dm_port, 0.0)
                self._cp_port = np.append(self._cp_port, 0.0)
                self._UA_port = np.append(self._UA_port, 0.0)
                # set dm in-out to port size, to enable also using it on all
                # kind of differently shaped arrays:
                self._dm_io = np.zeros_like(self._T_port)

                # things which can be done in general:
                # now add/insert new port in _port_own_idx
                # find place where to insert:
                idx_to_insert = self._port_own_idx.searchsorted(value[0])
                # insert
                self._port_own_idx = np.insert(
                    self._port_own_idx, idx_to_insert, value[0]
                )
                # expand port massflow characteristics tuple:
                self.dm_char += ('in',)

                # insert port name at correct place in port names::
                self.port_names = (
                    self.port_names[:idx_to_insert]
                    + (key,)
                    + self.port_names[idx_to_insert:]
                )
                # update number of ports:
                self.port_num += 1
        # save indices to 2D array version for backwards-compatibility.
        # TODO: Remove as soon as no more parts use 2D indices!
        self._port_own_idx_2D = self._port_own_idx[:]

    def add_open_port(self, name, constant, temperature):
        """
        Add an open port for assigning boundary conditions.

        **Only temperatures can be set with this method.**

        Parameters
        ----------
        name : str
            Name of the boundary condition. Must start with `'BC_'`.
        constant : bool
            If theBC is constant (True) or transient (False).
        temperature : int, float, np.ndarray, pd.Series, pd.DataFrame
            Temperature to set as BC.

        """
        # check if add open ports is already in postpone and if not, create it:
        if 'add_open_port' not in self.__postpone:
            self.__postpone['add_open_port'] = {}

        # check if open port was not already added to postpone dict:
        self._check_isadded(part=name, kind='open_port')

        # assert that no nan values were given:
        assert not np.isnan(temperature).any() and isinstance(
            temperature, (int, float, np.ndarray, pd.Series, pd.DataFrame)
        ), (
            'Value given to `add_open_port` with `name={0}` contains NaN '
            'values or is not numeric.'.format(name)
        )

        # postpone addition of parts to enable unordered sim. env.
        # construction:
        #        self.__postpone['add_open_port'][name] = (
        #                lambda: self._add_open_port(
        #                        name=name, constant=constant,
        #                        temperature=temperature))
        self.__postpone['add_open_port'][name] = _partial(
            self._add_open_port,
            name=name,
            constant=constant,
            temperature=temperature,
        )

    def _initialize_actuator(self, variable_name='dm', **kwargs):
        """
        This method adds the initialization checks for a part that is an
        actuator.
        """

        # if actuator has to be controlled (default) and thus is NOT set to
        # static, it needs a lower and upper limit for the controller:
        if 'ctrl_required' not in kwargs or (
            'ctrl_required' in kwargs and kwargs['ctrl_required'] is True
        ):
            err_str = (
                self._base_err
                + self._arg_err.format('lower_limit, upper_limit')
                + 'The part is an actuator and thus requires additional '
                'parameters. To disable treating this part as a controlled '
                'actuator, set `ctrl_required=False`. This enables setting a '
                'constant value or a time series to the actuator.\n'
                'To clip the controller action on the actuator to upper and '
                'lower boundaries, the control variable limits have to be '
                'given in {0} with `lower_limit=X` and `upper_limit=Y`.\n'
                'The limits have to be given as integer or float values with '
                '`lower_limit < upper_limit`. X and/or Y may be set to '
                '`+-np.inf` to disable clipping.'
            ).format(self._unit)
            assert 'lower_limit' in kwargs and 'upper_limit' in kwargs, err_str
            assert isinstance(kwargs['lower_limit'], (int, float)), err_str
            assert isinstance(kwargs['upper_limit'], (int, float)), err_str
            self._lims = np.array(  # set limits to array
                [kwargs['lower_limit'], kwargs['upper_limit']],
                dtype=np.float64,
            )
            self._llim = self._lims[0]  # also save to single floats
            self._ulim = self._lims[1]  # also save to single floats
        # if part does not need control (static or given values):
        elif 'ctrl_required' in kwargs and kwargs['ctrl_required'] is False:
            # assert that correct arguments are given:
            err_str = (
                self._base_err
                + self._arg_err.format('const_val OR time_series')
                + 'If `ctrl_required=False` is defined, the actuator\'s '
                'control variable needs to be set manually with:\n'
                '    `const_val`: A constant value for the actuator control '
                'variable in {0}, given as an integer or float value.\n'
                '    `time_series`: A time dependent value which will be set '
                'to the actuator control variable depending on the elapsed '
                'simulation time. Needs to be given as a pandas Series or '
                'DataFrame object with a DatetimeIndex and the values in {0}.'
                '\n\n'
                'Only one of these options may be given.\n'
                'For parts with multiple actuator ports, like a 3 way valve, '
                'the given values will always be assigned to the first '
                'actuator port, for example port A for a 3 way valve.'
            )
            assert 'const_val' in kwargs or 'time_series' in kwargs, err_str
            assert not (
                'const_val' in kwargs and 'time_series' in kwargs
            ), err_str
            # if part is static:
            if 'const_val' in kwargs:
                # check for correct type:
                err_str = (
                    self._base_err
                    + self._arg_err.format('const_val')
                    + '`const_val` type mismatch.'
                )
                assert isinstance(kwargs['const_val'], (int, float)), err_str
                self._actuator_CV[0] = float(kwargs['const_val'])
            elif 'time_series' in kwargs:
                # check for correct type:
                err_str = (
                    self._base_err
                    + self._arg_err.format('time_series')
                    + 'If the part is set with predefined values over a '
                    'timespan with `time_series=X`, `X` has to  be a Pandas '
                    'Series with the index column filled with timestamps '
                    'which have to outlast the simulation timeframe. The '
                    'actuator control variable to set has to be given in '
                    'the first column (index 0). To set a constant actuator '
                    'CV, use `const_val` instead.'
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
                self._models.assign_boundary_cond(  # set dynamic BC
                    time_series=kwargs['time_series'],
                    open_port=None,
                    part=self.name,
                    variable_name=variable_name,
                    array_index=0,
                )
            # set bools
            self.control_req = False
            self.ctrl_defined = True

    def _add_open_port(self, *, name, constant, temperature):
        """
        This method adds an open port to the simulation environment. A constant
        temperature value must be assigned to the open port. The open port must
        be connected to a port of a part will impose an open boundary condition
        for this part.
        The open port can be accessed by its name in the port dictionary
        `ports` and the port mapping dictionary `_port_mapping`.
        To assign a dynamic time dependent boundary condition to open ports,
        the method `modenv.assign_boundary_cond()` must be used.

        Parameters:
        -----------
            name : str
                Name of the port, starting with \'BC_\'
            temperature : float, int
                Constant temperature value in [°C] to assign to the port. Can
                be replaced with a dynamic time dependent value with
                `modenv.assign_boundary_cond()`.
        """

        # check for correct name and temperature
        err_str = (
            'The name of the open port to add to the simulation '
            'environment must be a string and begin with \'BC_\' '
            'to facilitate the lookup of the port in dictionaries!'
        )
        assert isinstance(name, str) and 'BC_' == name[0:3], err_str
        # check if name has not yet been given to another port:
        assert name not in self.ports, (
            name + ' for `add_open_port()` is already existing!'
        )
        assert isinstance(constant, bool), (
            '`constant=' + str(constant) + '` was passed to '
            '`add_open_port(name=' + name + ', ...)`.'
            '`constant` must be a bool value depicting if the boundary '
            'condition is given by a constant value (`constant=True`) or '
            'a dynamic time series (`constant=False`)!'
        )

        if not constant:  # if dynamic BC, construct dummy temperature
            err_str = (
                '`temperature` passed to '
                '`{0}.add_open_port(name=' + name + ', ...)` with '
                '`constant=False` has to be a pandas DataFrame or Series. '
                'If further exceptions occur, the temperature will be '
                'referred to as `time_series`.'
            ).format(self._simenv_name)
            assert isinstance(temperature, (pd.Series, pd.DataFrame)), err_str
            tmprtr = temperature  # backup dynamic temperature
            temperature = 25.0  # dummy const. temp. for dummy const. BC

        # construct constant BC, also for dynamic BC as a base BC to build the
        # dynamic BC:
        err_str = (
            'The constant boundary condition temperature of the open port '
            'must be an integer or float value in [°C].'
        )
        assert isinstance(temperature, (int, float)), err_str
        temperature = float(temperature)  # make sure it is float

        # add BC port to dictionaries:
        self.ports['BoundaryCondition;' + name] = np.array([temperature])
        self._port_mapping[name] = ('BoundaryCondition', name, None)
        # append row to ports array, which can be accessed with port ids
        # where the port value will be stored in and increase port counter:
        self.ports_all = np.append(self.ports_all, 0.0)
        self.num_ports += 1
        # also enlarge plinks_arr:
        self.plinks_arr = np.append(self.plinks_arr, 0)
        # add to boundary conditions dictionary:
        self.boundary_conditions[name] = {
            'type': 'open_port',
            'condition': 'constant',
            'value': temperature,
            'port_id': self.num_ports - 1,
        }
        # and finally add BC value to ports array:
        self.ports_all[self.num_ports - 1] = temperature

        if not constant:  # if dynamic time dependent BC
            # check if dyn BC alread in postpone and if not, create it:
            if 'dyn_BC' not in self.__postpone:
                self.__postpone['dyn_BC'] = {}
            # postpone addition of dynamic boundary conditions to open port to
            # avoid altering ports_all after making memoryviews to it:
            self.__postpone['dyn_BC'][
                name
            ] = lambda: self.assign_boundary_cond(
                time_series=tmprtr, open_port=name
            )
            # update information dict:
            self.boundary_conditions[name]['condition'] = 'dynamic'
            self.boundary_conditions[name]['value'] = tmprtr

    # @staticmethod
    def assign_boundary_cond(
        self,
        time_series,
        open_port=None,
        part=None,
        control=None,
        variable_name=None,
        array_index=None,
    ):
        """
        Assign dynamic boundary conditions to the simulation environment.

        With this method, static boundary conditions like **open ports**,
        **ambient temperatures** or **controller set points** can be converted
        to dynamic time dependent values.

        Boundary conditions assigned with this method have to be timeseries
        in a pandas Series or DataFrame with a DateTimeIndex.

        Parameters
        ----------
        * : TYPE
            DESCRIPTION.
        time_series : TYPE
            DESCRIPTION.
        open_port : TYPE, optional
            DESCRIPTION. The default is None.
        part : TYPE, optional
            DESCRIPTION. The default is None.
        variable_name : TYPE, optional
            DESCRIPTION. The default is None.
        array_index : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        # get name for giving good errors
        err_name = (
            open_port
            if open_port is not None
            else part
            if part is not None
            else control
            if control is not None
            else variable_name
        )
        # check for correct time_series:
        err_str = (
            'Error while assigning boundary condition to: `{0}`\n'
            '`time_series` passed to `assign_boundary_cond()` has to be of '
            'type pd.DataFrame or pd.Series with the values stored in the '
            'column with index 0 and the timestamps stored in the index as '
            'numpy `dtype=\'datetime64[ns]\'`.\n'
            'The index must be **EVENLY SPACED**, monotonic increasing and '
            'without any missing values!!!\n'
            'If this is NOT the first dynamic boundary condition, ALL '
            'subsequently added dynamic boundary conditions MUST be of the '
            'same shape as the time_series of the first dynamic boundary '
            'conditions AND the timestamp in the index must be exactly equal!'
        ).format(err_name)
        assert isinstance(time_series, (pd.Series, pd.DataFrame)), err_str
        assert isinstance(time_series.index, pd.DatetimeIndex), err_str
        assert isinstance(time_series.index.values, np.ndarray), err_str
        assert (getattr(time_series.index, 'freq', None) is not None) or (
            getattr(time_series.index, 'inferred_freq', None) is not None
        ), 'Index not evenly spaced, monotonic or without missing values!'
        assert time_series.index.values.dtype == 'datetime64[ns]', err_str
        # check for matching start dates if it should not be inferred
        if self._disksaving_set and not self.__infer_start_date:
            assert time_series.index[0] == self._disk_store['start_date'], (
                'Start date for disksaving does not match boundary condition '
                'start date. Both must match exactly. If disksaving start '
                'date should be inferred from boundary condition start date, '
                'set `start_date=\'infer\'` for disksaving.'
            )
        elif self.__infer_start_date and self._disksaving_set:
            self._disk_store['start_date'] = time_series.index[0]
            self._disk_store['curr_step_start_date'] = time_series.index[0]

        for_which, for_which_str = (
            (open_port, 'open_port')
            if open_port is not None
            else (part, 'part')
            if part is not None
            else (control, 'control')
        )
        assert not np.isnan(time_series).any(), (
            '`time_series` passed to `assign_boundary_cond()` for `{0}={1}` '
            'may not contain NaN values.'.format(for_which_str, for_which)
        )
        # if a timeseries has already been added:
        if self._dyn_BC_timevec.shape != (0,):
            err_st2 = (
                'Error while assigning boundary condition to: `{0}`\n'
                '\n\n'
                'Another timeseries has already been added as a boundary '
                'condition to this simulation environment. This may also '
                'include timeseries added to actuator parts like a massflow '
                'timeseries for a pump.\n'
                'All timeseries added to a simulation environment must be of '
                'the same shape and must have exactly the same timestamp '
                'index. The timeseries added before has the following shape: '
                + str(self._dyn_BC_timevec.shape)
                + ', first entry: '
                + str(self._dyn_BC_timevec[0])
                + ' and last entry: '
                + str(self._dyn_BC_timevec[-1])
            ).format(err_name)
            assert (
                self._dyn_BC_timevec.shape == time_series.index.values.shape
            ), (err_str + err_st2)
            assert np.all(self._dyn_BC_timevec == time_series.index.values), (
                err_str + err_st2
            )
        else:
            # get timedelta of timeseries in seconds (from nanoseconds):
            self._SimEnv__BC_timedelta = (
                time_series.index.values[1].item()
                - time_series.index.values[0].item()
            ) / 1e9
            err_str = (
                'Error while assigning boundary condition to: `{0}`\n'
                'The index of `time_series` has to cover at least a '
                'timeframe a few seconds longer than the simulation '
                'environment timeframe, to avoid running into '
                'indexing problems!'
            ).format(err_name)
            assert (
                self._SimEnv__BC_timedelta * time_series.index.size
                > self.timeframe
            ), err_str
            # add new timeseries in seconds frequency:
            self._dyn_BC_timevec = time_series.index.values.astype(
                'datetime64[s]'
            )
            # preallocate last step index counter for interpolation:
            self._SimEnv__last_step_t_idx = 0

        # add dynamic BC for an open port
        if open_port is not None and part is None and control is None:
            err_str = (
                '`open_port=' + open_port + '` passed to '
                '`assign_boundary_cond()` was not found!'
            )
            assert open_port in self.boundary_conditions, err_str
            # get open port id
            pid = self.boundary_conditions[open_port]['port_id']
            # get trgt index as slice to open port
            trgt_idx = slice(pid, pid + 1)
            # get view to open port:
            trgt_arr_view = self.ports_all[trgt_idx]
        # add dynamic BC for a part
        elif open_port is None and part is not None and control is None:
            err_str = (
                'Error while assigning boundary condition to: `{0}`\n'
                'While assigning dynamic boundary conditions to a part '
                'specific variable, at least one of the following parameters '
                'was not found in the simulation environment:\n'
                'part={0}, variable_name={1}'.format(part, variable_name)
                + '\n MUST be called after `initialize_sim`, otherwise parts '
                'won\'t be found!.'
            ).format(err_name)
            assert part in self.parts and hasattr(
                self.parts[part], variable_name
            ), err_str
            # check if variable is already an array and if NOT, make it one:
            if not isinstance(
                getattr(self.parts[part], variable_name), np.ndarray
            ):
                # bkp old value:
                bkp_ov = getattr(self.parts[part], variable_name)
                # make array and pass bkp to it:
                setattr(self.parts[part], variable_name, np.array([bkp_ov]))
            err_str = (
                'Error while assigning boundary condition to: `{2}`\n'
                '`array_index={0:G}` passed to `assign_boundary_cond()` is '
                'either not given as an integer or out of range for the array '
                'with shape {1}'.format(
                    array_index,
                    getattr(self.parts[part], variable_name).shape,
                    err_name,
                )
            )
            assert isinstance(array_index, int), err_str
            assert array_index < (
                getattr(self.parts[part], variable_name).shape[0]
            ), err_str
            # create view to array with idx:
            trgt_idx = slice(array_index, array_index + 1)
            trgt_arr_view = getattr(self.parts[part], variable_name)[trgt_idx]
        # add dynamic BC for a controller SET POINT (SP)
        elif open_port is None and part is None and control is not None:
            err_str = (
                'Error while assigning boundary condition to: `{2}`\n'
                'While assigning dynamic boundary conditions to a controller '
                'specific variable with `simenv.assign_boundary_cond()`, '
                'the following controller was not found:\n'
                'control=\'{0}\'\n'
                'Existing controllers: {1}\n'
                '`assign_boundary_cond()` MUST be called after '
                '`initialize_sim`, otherwise no controllers exist!'.format(
                    control, repr(list(self.ctrls.keys())), err_name
                )
            )
            assert control in self.ctrls and hasattr(
                self.ctrls[control], 'sp'
            ), err_str
            # check if variable is already an array and if NOT, make it one:
            if not isinstance(getattr(self.ctrls[control], 'sp'), np.ndarray):
                # bkp old value:
                bkp_ov = getattr(self.ctrls[control], 'sp')
                # make array and pass bkp to it:
                setattr(self.ctrls[control], 'sp', np.array([bkp_ov]))
            # create view to array with idx:
            trgt_idx = slice(0, 1)
            trgt_arr_view = getattr(self.ctrls[control], 'sp')[trgt_idx]
        else:
            raise ValueError(
                'Either part, open_post or control must be given.'
            )

        # add BC to dynamic BC tuple with target array view in the first
        # position and values of time_series in the second position:
        self._dyn_BC += ((trgt_arr_view, time_series.values),)

    def _initialize_dynamic_BC(self):
        """
        This method initializes dynamic boundary conditions by setting the
        first value in the time series to the ports array position.
        """
        for dbc in self._dyn_BC:
            dbc[0][0] = dbc[1][0]  # set first time series element to view

    def __global_init_part(self):
        """
        This methods is called in the part specific init_part()-method and
        initializes the part in the global scope (simulation environment class
        scope).
        In the current implementation this means saving the port statistics
        (arrays and dicts to save port values, port ids, number of ports, port
        mapping and linking arrays) to simulation environment variables.
        Since dicts are ordered since Python 3.6, the order of parts and ports
        is retained. Even though this is not deemed to be necessary for
        the convergence of the solution, this can slightly decrease the
        calculation cost.
        """

    def _chk_amb_temp(self, **kwargs):
        """
        Intenal method to be called from within parts.

        Check if ambient temperature is given to the initialization method.
        """
        err_str = (
            self._base_err
            + self._arg_err.format('T_amb')
            + 'The ambient temperature `T_amb=X` has to be given, where X is '
            'the temperature in [°C] as an integer or float value '
            '`-273.15 < X`.'
        )
        assert (
            'T_amb' in kwargs
            and isinstance(kwargs['T_amb'], (int, float))
            and kwargs['T_amb'] > -273.15
        ), err_str
        self._T_amb = np.array([float(kwargs['T_amb'])])

    def add_control(self, Controls_module, name, **kwargs):
        """
        Add controls to the simulation environment.

        The control type has to be passed via the argument `Controls_module`
        as a Class type. Refer to the `.controls` module to get more
        information.

        Parameters
        ----------
        Controls_module : Class
            DESCRIPTION.
        name : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # check if add control is already in postpone and if not, create it:
        if 'add_ctrl' not in self.__postpone:
            self.__postpone['add_ctrl'] = {}
            self.__postpone['init_ctrl'] = {}  # same for init. ctrl

        # check if control was not already added to postpone dict:
        self._check_isadded(part=name, kind='control')

        # postpone addition of controls to enable unordered sim. env.
        # construction (passing on self is needed, since some vars are saved in
        # the created SimEnv class instance!):
        #        self.__postpone['add_ctrl'][name] = (
        #                lambda: self._add_control(
        #                        Controls_module, name,
        #                        controlled_part, controlled_port, ctrl_algorithm,
        #                        reference_part='none', reference_port='none',
        #                        steady_state=False, **kwargs))
        self.__postpone['add_ctrl'][name] = _partial(
            self._add_control, Controls_module, name, **kwargs
        )
        # postpone initialization of controls to enable unordered sim. env.
        # construction:
        #        self.__postpone['init_ctrl'][name] = (
        #                lambda: self.ctrls[name].init_controller(**kwargs))
        self.__postpone['init_ctrl'][name] = kwargs

    def _add_control(self, Controls_module, name, **kwargs):
        """Postponed partial method doing the internal stuff."""
        # check if number of connected ports equals the number of ports (that
        # is all ports have been connected). Double checking after explicit
        # checking in higher level call postponed method.
        err_str = (
            'Before adding controls all ports must be connected!\n'
            'Use method `{0}.connect_ports()` to connect ports.'
        ).format(self._simenv_name)
        assert self.num_ports == self._num_conn_ports, err_str
        # check for already existing Controls with the same name:
        err_str = 'A control with `name={0}` is already existing.'.format(name)
        assert name not in self.ctrls, err_str

        # Add control to controls list:
        self.ctrl_l.append(Controls_module(name, self, **kwargs))
        # reference control in user-interface controls dict:
        self.ctrls[name] = self.ctrl_l[self.num_ctrls]

        # increase controls count by one:
        self.num_ctrls += 1

    def _update_ports(self, source='results', all_ports=False, nnum=False):
        """Update values of ports. Deprecated? Probably safe to remove."""
        assert_str = (
            'Source for updating port values has to be set to \'result\' to '
            'use the result after each completed calculation step or to '
            '\'intermediate\' to use the intermediate result in between the '
            'calculation steps!'
        )
        assert source == 'results' or source == 'intermediate', assert_str

        # if only specific ports have to be updated:
        if not all_ports:
            # Parts which have to be solved with a numeric method will update
            # their ports after each step, other parts will be updated in the
            # end of this function:
            if self.solve_numeric or nnum:
                if source == 'results':
                    values = self.res[self._models.stepnum[0]]
                elif source == 'intermediate':
                    values = self.T
                # numba version:
                _pf.upd_p_arr(
                    self._models.ports_all,
                    self.port_ids,
                    values,
                    self._port_own_idx,
                )
                if np.any(np.isnan(values)):
                    raise NotADirectoryError
        # update all ports:
        elif all_ports:
            # recursive function call for all non-numeric parts:
            if not nnum:
                for part in self._models.parts:
                    self._models.parts[part]._update_ports(source=source)
            # parts which are NOT solved in a numeric method will be updated
            # AFTER ALL OTHER PARTS!
            else:
                for part in self._models.solve_nonnum:
                    self._models.parts[part]._update_ports(
                        source=source, nnum=True
                    )
        else:
            err_str = '\'all_ports\' only allows boolean values True/False!'
            raise ValueError(err_str)

    def _get_port_connections(self):
        """
        Construct tuples containing part+port identifiers.

        Constructs the tuples which contain the part+port identifier and the
        arrays which contain the indices of the ports connected to the current
        part. The arrays are used to get the temperature from the connected
        ports in the differential calculation.
        Furthermore constant values of the connected ports like the fluid
        and wall area are saved to the parts.
        """
        #        now get the gridspacing arrays filled with this one! memoryviews not
        #        needed here, since gridspacing won't change.
        #        and also get the U value arrays for connected ports from here.
        #        memory views for this are great (are they? changing u values while calculation
        #        is running could be reaaally problematic, since other parts will see
        #        the change before one complete solver step is finished!)!
        #        perhaps go the same way like with
        #        flow nets. they seem to be super-duper-fast!!

        # since non-numeric parts like pumps and 3w-valves need to pass on the
        # (mean) port connection parameters, these need to be connected before
        # numeric parts. thus get the order for looping over the parts:
        loop_order = list()
        for part in self.parts:
            # if numeric part: append at the end
            if self.parts[part].solve_numeric:
                loop_order.append(part)
            else:
                # else if non-numeric part: insert at the beginning
                loop_order.insert(0, part)

        for part in loop_order:
            other_port_key = []
            for i in range(self.parts[part].port_num):
                self_port_key = (
                    self.parts[part].name
                    + ';'
                    + self.parts[part].port_names[i]
                )
                try:
                    other_port_key.append(self.port_links[self_port_key])
                except KeyError:
                    # if a KeyError occures, at least one port has not yet
                    # been connected. Loop over all ports to find the not
                    # connected ports, add this up to a string and raise error:
                    missing_ports = (
                        'The following ports have not been '
                        'connected to any other port:\n'
                    )
                    for part_port in self.ports:
                        pa, po = part_port.split(';')
                        if part_port not in self.port_links:
                            missing_ports += pa + ': ' + po + '\n'
                    missing_ports += (
                        'Use method `{0}.connect_ports()` to connect ports.'
                    ).format(self._simenv_name)
                    raise AssertionError(missing_ports)
                # now also get view to port array in ports dict for easy lookup
                pid = self.parts[part].port_ids[i]
                self.ports[self_port_key] = self.ports_all[pid : pid + 1]
            # construct tuple of connected ports in correct order (following
            # _port_own_idx) for easy lookup
            self.parts[part]._connected_ports = tuple(other_port_key)
            # construct the same as an array version for fast lookup during
            # iterations using integer indices:
            self.parts[part]._port_link_idx = self.plinks_arr[
                self.parts[part].port_ids
            ]
            # now also get all other port connection parameters which are
            # constant during iterations. This includes:
            # - Area of port (inner flow area and cross section area of casing
            #   material). Needed to calculate heat conduction.
            # - Gridspacing of connected parts. Needed to calculate heat
            #   conduction.
            # - Heat conductivity of port casing material (only steel etc.,
            #   insulation will be ignored). Assuming values to not be
            #   temperature dependent. Needed to calculate heat conduction.
            # loop over ports again (could also be done above, but it is
            # clearer here):
            for port in self.parts[part].port_names:
                # get own target port idx where the parameters must be saved:
                trgt_idx = self._get_arr_idx(
                    part, port, target='port_specs', as_slice=False
                )
                # get other part's (source part's) part and port as string:
                src_part, src_port = self._get_conn_port(part + ';' + port)
                # if source part is a BC, get own part as connected part so
                # that all calculated port values are same as the own values:
                if src_part == 'BoundaryCondition':
                    src_part = part
                    src_port = port
                # get wall cross section area, fluid/flow cross section area,
                # gridspacing and lambda of the wall material
                # (grid spacing and lambda are assumed to be constant within
                # source part thus taken from single float. areas are dependent
                # on the type of ports of the part and need to be checked.
                if self.parts[src_part]._port_heatcond:
                    # only get these values if port heat conduction is enabled
                    if not isinstance(
                        self.parts[src_part]._A_wll_own_p, np.ndarray
                    ):
                        # get port values if single float value
                        A_wcp = self.parts[src_part]._A_wll_own_p
                        A_fcp = self.parts[src_part]._A_fld_own_p
                    else:
                        # if ports have different values
                        # get src_idx to the port location:
                        src_idx = self._get_arr_idx(
                            src_part,
                            src_port,
                            target='port_specs',
                            as_slice=False,
                        )
                        # get port values
                        A_wcp = self.parts[src_part]._A_wll_own_p[src_idx]
                        A_fcp = self.parts[src_part]._A_fld_own_p[src_idx]
                    # gridspacing and lambda wall is NOT always constant:
                    if type(self.parts[src_part].grid_spacing) != np.ndarray:
                        gsp_p = self.parts[src_part].grid_spacing
                        lam_wcp = self.parts[src_part]._lam_wll
                    else:
                        gsp_p = self.parts[src_part].grid_spacing[src_idx]
                        lam_wcp = self.parts[src_part]._lam_wll[src_idx]
                else:  # if no port heat conduction, set all to zero
                    A_wcp = 0.0
                    A_fcp = 0.0
                    gsp_p = 0.0
                    lam_wcp = 0.0
                # save all to arrays at trgt_idx position:
                try:
                    self.parts[part]._A_wll_conn_p[trgt_idx] = A_wcp
                except IndexError:
                    raise IndexError(part, trgt_idx, port)
                self.parts[part]._A_fld_conn_p[trgt_idx] = A_fcp
                try:
                    self.parts[part]._port_gsp[trgt_idx] = gsp_p
                except IndexError:
                    raise ValueError(part)
                self.parts[part]._lam_wll_conn_p[trgt_idx] = lam_wcp
                # and also get the sum of the own gridspacing and ports
                # gridspacing for some calculations:
                self.parts[part]._port_subs_gsp = (
                    self.parts[part].grid_spacing + self.parts[part]._port_gsp
                )
            # now for non-numeric parts: backup the connected port information
            # arrays, loop over the ports again and for each port get the mean
            # values of the other ports and save this as the own port values.
            # This is necessary to allow correct calculation of heat conduction
            # from numeric parts over non-numeric parts:
            if not self.parts[part].solve_numeric:
                # backup directly connected values:
                A_wll_conn_bkp = self.parts[part]._A_wll_conn_p.copy()
                A_fld_conn_bkp = self.parts[part]._A_fld_conn_p.copy()
                port_gsp_bkp = self.parts[part]._port_gsp.copy()
                lam_wll_conn_bkp = self.parts[part]._lam_wll_conn_p.copy()
                # loop over the ports:
                for port in self.parts[part].port_names:
                    # get own target port idx where the parameters must be
                    # saved:
                    trgt_idx = self._get_arr_idx(
                        part, port, target='port_specs', as_slice=False
                    )
                    # construct mask for value arrays which excludes the
                    # trgt_idx and only includes real ports:
                    mask = np.zeros(A_wll_conn_bkp.shape, dtype=np.bool)
                    # only consider real ports
                    mask.flat[self.parts[part]._port_own_idx] = True
                    mask[trgt_idx] = False  # leave out current port
                    # save the mean value of the other connected port values
                    # to the current port value:
                    self.parts[part]._A_wll_own_p[trgt_idx] = A_wll_conn_bkp[
                        mask
                    ].mean()
                    self.parts[part]._A_fld_own_p[trgt_idx] = A_fld_conn_bkp[
                        mask
                    ].mean()
                    self.parts[part].grid_spacing[trgt_idx] = port_gsp_bkp[
                        mask
                    ].mean()
                    self.parts[part]._lam_wll[trgt_idx] = lam_wll_conn_bkp[
                        mask
                    ].mean()

    def connect_ports(self, first_part, first_port, scnd_part, scnd_port):
        r"""
        Connect ports of two parts or one port to a boundary condition.

        The port `first_port` of part `first_part` is connected with port
        `scnd_port` of part `scnd_part`.

        Any port can also be connected to an open port **boundary condition**.
        If an open port boundary condition shall be connected to a port, one
        of the parts has to be set to \'BoundaryCondition\' and the
        corresponding port to the the BC name, f.i.
        `scnd_part=\'BoundaryCondition\'`, `scnd_port=\'coldwater_temp\'`.

        Parameters
        ----------
        first_part : str
            Name of the first part.
        first_port : str
            Port identifier of the first part's port.
        scnd_part : str
            Name of the second part.
        scnd_port : str
            Port identifier of the second part's port.

        Returns
        -------
        None.

        """
        # check if connect ports is already in postpone and if not, create it:
        if 'connect_ports' not in self.__postpone:
            self.__postpone['connect_ports'] = {}
        # postpone connecting ports to enable unordered sim. env. construction:
        #        self.__postpone['connect_ports'][first_part + ';' + first_port] = (
        #                lambda: self._connect_ports(
        #                        first_part=first_part, first_port=first_port,
        #                        scnd_part=scnd_part, scnd_port=scnd_port))
        self.__postpone['connect_ports'][
            first_part + ';' + first_port
        ] = _partial(
            self._connect_ports,
            first_part=first_part,
            first_port=first_port,
            scnd_part=scnd_part,
            scnd_port=scnd_port,
        )

    def _connect_ports(self, *, first_part, first_port, scnd_part, scnd_port):
        """
        Postponed partial method of connect_ports, doing the internal stuff.

        Parameters
        ----------
        first_part : str
            Name of the first part.
        first_port : str
            Port identifier of the first part's port.
        scnd_part : str
            Name of the second part.
        scnd_port : str
            Port identifier of the second part's port.

        Returns
        -------
        None.

        """
        # check if parts exist:
        try:
            if first_part != 'BoundaryCondition':
                self._check_ispart(part=first_part)
            if scnd_part != 'BoundaryCondition':
                self._check_ispart(part=scnd_part)
        except AssertionError as e:
            raise AssertionError(
                'While connecting ports, the following error '
                'occurred:\n{0}'.format(e)
            )
        # check if port has already been connected to another port:
        err_str = (
            'While connecting part `{0}` at port `{1}` with part `{2}` at '
            'port `{3}` an error occurred:\n'
            'Port `{4}` of part `{5}` has already been connected to another '
            'port or boundary condition.'
        ).format(first_part, first_port, scnd_part, scnd_port, '{0}', '{1}')

        # check connected parts:
        if (first_part != 'BoundaryCondition') and hasattr(
            self.parts[first_part], '_ports_connected'
        ):
            assert (
                first_port not in self.parts[first_part]._ports_connected
            ), err_str.format(first_port, first_part)
        elif first_part != 'BoundaryCondition':  # create empty list if not yet
            self.parts[first_part]._ports_connected = []
            self.parts[first_part]._ports_unconnected = self.parts[
                first_part
            ].port_names
        if scnd_part == 'p_gasb1_ff':
            breakpoint()
        if (scnd_part != 'BoundaryCondition') and hasattr(
            self.parts[scnd_part], '_ports_connected'
        ):
            assert (
                scnd_port not in self.parts[scnd_part]._ports_connected
            ), err_str.format(scnd_port, scnd_part)
        elif scnd_part != 'BoundaryCondition':  # create empty list if not yet
            self.parts[scnd_part]._ports_connected = []
            self.parts[scnd_part]._ports_unconnected = self.parts[
                scnd_part
            ].port_names

        # find id of first_port of first_part if first part is not set to be a
        # BC:
        if first_part != 'BoundaryCondition':
            try:
                own_port_id = self.parts[first_part].port_ids[
                    self.parts[first_part].port_names.index(first_port)
                ]
            except ValueError:
                err_str = (
                    'While connecting part `{0}` at port `{1}` with part `{2}`'
                    ' at port `{3}` an error occurred:\nPort `{1}` at `{0}` '
                    'was not found!\n The following ports exist at that '
                    'part:\n'.format(
                        first_part, first_port, scnd_part, scnd_port
                    )
                    + str(self.parts[first_part].port_names)
                    + '\nAnd the following ports are not yet connected:\n'
                    + str(set(self.parts[first_part]._ports_unconnected))
                )
                raise ValueError(err_str)
            except KeyError:
                err_str = (
                    'While connecting part `{0}` at port `{1}` with part `{2}`'
                    ' at port `{3}` an error occurred:\nPart `{0}` was not '
                    'found!\n The following parts have been added to the '
                    'simulation environment:\n'.format(
                        first_part, first_port, scnd_part, scnd_port
                    )
                    + str(list(self.parts))
                )
                raise ValueError(err_str)
        else:
            # else if first part is BC:
            err_str = (
                'The boundary condition `{0}` was not '
                'found!\nThe following boundary conditions have been '
                'added to the simulation environment:\n'.format(first_port)
                + str(list(self.boundary_conditions))
            )
            assert first_port in self.boundary_conditions, err_str
            # get ports array index of boundary condition:
            own_port_id = self.boundary_conditions[first_port]['port_id']

        # find id of scnd_port of scnd_part if scnd part is not set to be a
        # BC:
        if scnd_part != 'BoundaryCondition':
            try:
                other_port_id = self.parts[scnd_part].port_ids[
                    self.parts[scnd_part].port_names.index(scnd_port)
                ]
            except ValueError:
                err_str = (
                    'While connecting part `{0}` at port `{1}` with part `{2}`'
                    ' at port `{3}` an error occurred:\nPort `{3}` at `{2}` '
                    'was not found!\nThe following ports exist at that '
                    'part:\n{4}\n'
                    'And the following ports are not yet connected:\n{5}'
                ).format(
                    first_part,
                    first_port,
                    scnd_part,
                    scnd_port,
                    repr(self.parts[scnd_part].port_names),
                    repr(set(self.parts[scnd_part]._ports_unconnected)),
                )
                raise ValueError(err_str)
            except KeyError:
                err_str = (
                    'While connecting part `{0}` at port `{1}` with part `{2}`'
                    ' at port `{3}` an error occurred:\nPart `{2}` was not '
                    'found!\n The following parts have been added to the '
                    'simulation environment:\n'.format(
                        first_part, first_port, scnd_part, scnd_port
                    )
                    + str(list(self.parts))
                )
                raise ValueError(err_str)
        else:
            # else if second part is BC:
            err_str = (
                'The boundary condition `{0}` was not found!\n The following '
                'boundary conditions have been added to the simulation '
                'environment:\n'.format(scnd_port)
                + str(list(self.boundary_conditions))
            )
            assert scnd_port in self.boundary_conditions, err_str
            # get ports array index of boundary condition:
            other_port_id = self.boundary_conditions[scnd_port]['port_id']

        # add links to plinks_arr:
        self.plinks_arr[own_port_id] = other_port_id
        self.plinks_arr[other_port_id] = own_port_id
        # port_links dict is filled (with a tuple) for indexing the connected
        # port to get the values of this port:
        self.port_links[first_part + ';' + first_port] = (
            scnd_part + ';' + scnd_port
        )
        # vice-versa linking:
        self.port_links[scnd_part + ';' + scnd_port] = (
            first_part + ';' + first_port
        )
        # count how many ports have already been connected:
        self._num_conn_ports += 2
        # save connected ports to each part
        if (first_part != 'BoundaryCondition') and hasattr(
            self.parts[first_part], '_ports_connected'
        ):
            self.parts[first_part]._ports_connected.append(first_port)
            self.parts[first_part]._ports_unconnected = tuple(
                set(self.parts[first_part].port_names)
                - set(self.parts[first_part]._ports_connected)
            )
        elif first_part != 'BoundaryCondition':
            self.parts[first_part]._ports_connected = [first_port]
            self.parts[first_part]._ports_unconnected = tuple(
                set(self.parts[first_part].port_names)
                - set(self.parts[first_part]._ports_connected)
            )
        if (scnd_part != 'BoundaryCondition') and hasattr(
            self.parts[scnd_part], '_ports_connected'
        ):
            self.parts[scnd_part]._ports_connected.append(scnd_port)
            self.parts[scnd_part]._ports_unconnected = tuple(
                set(self.parts[scnd_part].port_names)
                - set(self.parts[scnd_part]._ports_connected)
            )
        elif scnd_part != 'BoundaryCondition':
            self.parts[scnd_part]._ports_connected = [scnd_port]
            self.parts[scnd_part]._ports_unconnected = tuple(
                set(self.parts[scnd_part].port_names)
                - set(self.parts[scnd_part]._ports_connected)
            )

    def _check_unconnected_ports(self):
        """Check if any ports are not yet connected."""
        missing_ports = (
            'The following ports have not been connected to any other port:\n'
        )
        unconnected_ports = False
        for part_port in self.ports:
            pa, po = part_port.split(';')
            if part_port not in self.port_links:
                unconnected_ports = True
                missing_ports += pa + ': ' + po + '\n'
        missing_ports += (
            'Use the method `{0}.connect_ports()` to connect ports.\n'
            'To add a static or dynamic temperature boundary condition to a '
            'port, call the function `{0}.add_open_port()` and connect it to '
            'the target part/port with `scnd_part=\'BoundaryCondition\', '
            'scnd_port=X`, where X is the name given to the boundary '
            'condition using `{0}.add_open_port()`.'
        ).format(self._simenv_name)
        if unconnected_ports:
            raise AssertionError(missing_ports)

    def _print_arg_errs(self, part, name, arg_and_err, kwd_args):
        err_base = (
            'While adding {part} `{name}` to the simulation environment, the '
            'following argument error(s) occurred:\n'
        ).format(part=part, name=name)
        err_ma = 'Missing argument or incorrect type/value:\n'
        argerr = '    - {arg}: {err}'
        argerrs = []
        for k, v in arg_and_err.items():
            ks = k.replace(' ', '').split('OR')
            in_kwds = False
            for ks_ in ks:
                if ks_ in kwd_args:
                    in_kwds = True
            if not in_kwds:
                argerrs.append(argerr.format(arg=k, err=v))
        if len(argerrs) >= 1:
            raise ValueError(err_base + err_ma + '\n'.join(argerrs))

    def _get_topology(self):
        """
        Calculate the topology of the flow net.

        This function determines the topology of the pipe-and-part-net around
        a pump to calculate the massflow in this net section. The net section
        is defined as the net, which lies in between two ports to (a) hydraulic
        compensator(s). If there is no hydraulic compensator, the net has to be
        a closed net.
        """
        # find flow net parent parts (currently only pumps) and construct
        # superior flow networks around them:
        for part in self.parts:
            if self.parts[part]._flow_net_parent:
                # create class instance which saves the flow variables:
                self.sup_net[part] = _FlowNet(self, part, sub_net=False)
                # loop over pump section until next hydraulic compensator:
                self.parts[part]._recursive_topology(part)

        # concatenate all superior flow nets in the given order to a tuple
        # which contains all nets:
        for _, net in self.sup_net.items():
            self.flow_net += tuple(net.dm_topo.values())

        # loop over hydr_comps (contains hydraulic compensators AND all other
        # remaining parts which were not yet added to flow nets) to find all
        # parts of sub flow nets. Do that while not all parts with the
        # bool checker break_topology and one open port have been removed.
        i = 1  # part checker
        # make list out of hydr comps set to be able to iterate over it while
        # changing it (ugly but required):
        self._hydr_comps = list(self._hydr_comps)
        while i >= 1:
            # loop over all parts left in hydr_comps
            for part in self._hydr_comps:
                # check if part has broken the topology, is not a hydraulic
                # compensator and is already defined enough by
                # _recursivetopology calls (counter for open ports shows only
                # one port remaining to be solved) to solve it as a parent
                # part of a sub net:
                if (
                    self.parts[part].break_topology
                    and not self.parts[part].hydr_comp
                    and self.parts[part]._cnt_open_prts == 1
                ):
                    # create subnet:
                    self.sub_net[part] = _FlowNet(self, part, sub_net=True)
                    self.parts[part]._recursive_topology(
                        part, sub_net=True, pump_side='sub net - undefined'
                    )
                    # set i to 2 to show that one part was found:
                    i = 2
            # now check if at least one part for sub nets was found during for
            # loop. if so, i will be reset to 1 for the next loop:
            if i == 2:
                i = 1
            else:
                # if no part has been found when looping over hydr_comps, no
                # more subnets can be constructed -> break out of while loop by
                # setting i to 0:
                i = 0
        # make set out of list again:
        self._hydr_comps = set(self._hydr_comps)

        # loop over hydr_comps again, but this time also solve parts which are
        # real hydraulic compensators in sub_nets if they have only one open
        # port left. Do that while not all parts with the
        # bool checker break_topology and one open port have been removed.
        i = 1  # part checker
        # make list out of hydr comps set to be able to iterate over it while
        # changing it (ugly but required):
        self._hydr_comps = list(self._hydr_comps)
        while i >= 1:
            for part in self._hydr_comps:
                # check if part has broken the topology and if only one port to
                # be solved is remaining:
                if (
                    self.parts[part].break_topology
                    and self.parts[part]._cnt_open_prts == 1
                ):
                    # create subnet:
                    self.sub_net[part] = _FlowNet(self, part, sub_net=True)
                    self.parts[part]._recursive_topology(
                        part, sub_net=True, pump_side='sub net - undefined'
                    )
                    # set i to 2 to show that one part was found:
                    i = 2
            # now check if at least one part for sub nets was found during for
            # loop. if so, i will be reset to 1 for the next loop:
            if i == 2:
                i = 1
            else:
                # if no part has been found when looping over hydr_comps, no
                # more subnets can be constructed -> break out of while loop by
                # setting i to 0:
                i = 0
        self._hydr_comps = set(self._hydr_comps)

        # concatenate all sub flow nets in the given order to the tuple
        # which contains all nets:
        for _, net in self.sub_net.items():
            self.flow_net += tuple(net.dm_topo.values())

        # remove all empty tuples and dummy tuples (dummy tuples were created
        # for easy information lookup in dicts and have the operation id -99):
        # create bkp of flow net and clear flow net
        flow_net_bkp = self.flow_net
        self.flow_net = tuple()
        # loop over bkp and fill flow net with "good" values:
        for element in flow_net_bkp:
            # skip empty tuples
            if element != ():
                # skip dummy tuples
                if element[1] != -99:
                    # append tuples as nested tuples
                    self.flow_net += (element,)

        # check parts remaining in hydr_comps if they are really only
        # hydr_comps, otherwise raise error for net creation:
        for part in self._hydr_comps:
            if not self.parts[part].hydr_comp:
                err_str = (
                    'The topology analyzer was not able to solve for the '
                    'massflow of `'
                    + part
                    + '` at port(s):\n'
                    + str(
                        set(self.parts[part].port_names)
                        - set(self.parts[part]._solved_ports)
                    )
                    + '\n\n'
                    'Please check if the part is solvable by following a '
                    'logical order in the flow net. Sometimes adding a '
                    'surrogate pump at the part of the net which could not '
                    'be solved can help. This pump can f.i. use a linked '
                    'controller to the main pump to couple the massflows.'
                )
                raise TypeError(err_str)

    def _recursive_topology(self, parent_pump, sub_net=False, **kwargs):
        """
        Build flow net recursively.

        Recursive function which builds the dictionairies for superior flow
        nets (flow nets with a pump) and sub flow nets (flow nets without a
        pump). These nets are separated by parts, which make it impossible for
        the topology algorithm to determin the massflow in the flow net without
        having knowledge about other flow nets. Typically separating parts are
        three way connectors or valves.
        Superior flow nets are saved in the `sup_net` dict, while sub flow nets
        are saved in the `sub_net` dict. Each net stores additional information
        among which the net's port-wise calculation procedure is stored in
        `dm_topo`.

        """
        # save parent pump to kwargs to make it accessible in sub functions
        kwargs['parent_pump'] = parent_pump
        # even though this should already be done when calling this function,
        # catch already solved parts again for security reasons ;)
        if self.name not in self._models._hydr_comps:
            return  # return to calling function to avoid double solving

        # if nonnum part move current part to the end to sort the solve_nonnum
        # OrderedDict for ordered solving:
        if not self.solve_numeric:
            self._models.solve_nonnum.move_to_end(self.name, last=True)

        # if next_port is passed as argument, port_names has to be reordered so
        # that the loop over the ports starts with next_port (otherwise the
        # loop over the ports would make recursive_topo to hop from part to
        # part without finishing each part). but to retain the order of
        # 'port_names' in the parts (needed for correctly getting and updating
        # port values), port_names is copied to a separate list for the use in
        # recursive_topology:
        port_names = list(self.port_names)
        if 'next_port' in kwargs:
            # next port is 'part;port' -> get only port string:
            next_port = kwargs['next_port'].split(';')[1]
            #  remove next_port from port_names
            port_names.remove(next_port)
            # insert next_port as first element of port_names:
            port_names.insert(0, next_port)

        # loop over all ports in current part
        for port in port_names:
            # if a sub net is constructed and the current port is already
            # solved: skip to next port
            # skip already solved ports
            if port in self._solved_ports:
                continue
            # construct current part + current port name:
            current_port = self.name + ';' + port
            # if current part has multiple flow channels, like a HEX, only
            # ports of the same side can be treated in this recursive_topology
            # call. Thus if current flow channel side can not be found in the
            # current port, skip the current port:
            if self.multiple_flows:
                if kwargs['flow_channel'] not in port:
                    continue

            # create dummy topo_routine to avoid checking for non-existent var
            # and overwrite vars of last loop
            topo_routine = None

            # set own massflow calculation routine:
            # if parent_port is not passed to recursive_topology and it is not
            # a sub_net, the current part is the parent part (== pump), so no
            # parent values will be added:
            if 'parent_port' not in kwargs and not sub_net:
                # construct empty tuple for values in odict for easy lookup:
                self._models.sup_net[parent_pump].dm_topo[
                    current_port
                ] = tuple()
                # set pump port to pump side to let next parts know if they
                # are on the pressure side (pump outlet) or suction side
                # (pump inlet) of the pump:
                kwargs['pump_side'] = (
                    'pressure' if port == 'out' else 'suction'
                )
                # run flow routine on part to create its information dict and
                # check for errors:
                self._get_flow_routine(port, subnet=False, **kwargs)
            # if parent_part is not passed to recursive topology and it is a
            # sub net:
            elif 'parent_port' not in kwargs and sub_net:
                # create empty tuple (parent_pump is the parent part here, even
                # though it is NOT a pump):
                self._models.sub_net[parent_pump].dm_topo[
                    current_port
                ] = tuple()
                # get massflow calculation routine and add to port's tuple:
                topo_routine = self._get_flow_routine(
                    port, subnet=True, **kwargs
                )
                self._models.sub_net[parent_pump].dm_topo[
                    current_port
                ] += topo_routine
                # get pump side from parent part to get specific information if
                # it was specified in the parts get flow routine method:
                kwargs['pump_side'] = self.info_topology[port]['Pump side']
            # if parent port is passed to recursive_topology, add flow values:
            elif 'parent_port' in kwargs and not sub_net:
                # create empty tuple:
                self._models.sup_net[parent_pump].dm_topo[
                    current_port
                ] = tuple()
                # append flow routine:
                topo_routine = self._get_flow_routine(
                    port, subnet=sub_net, **kwargs
                )
                self._models.sup_net[parent_pump].dm_topo[
                    current_port
                ] += topo_routine
                # break out of recursive topology for this part if not enough
                # ports have been solved to go on and thus break topology=True:
                if self.break_topology:
                    return
            elif 'parent_port' in kwargs and sub_net:
                # create empty tuple:
                self._models.sub_net[parent_pump].dm_topo[
                    current_port
                ] = tuple()
                # append flow routine:
                topo_routine = self._get_flow_routine(
                    port, subnet=sub_net, **kwargs
                )
                self._models.sub_net[parent_pump].dm_topo[
                    current_port
                ] += topo_routine
                # break out of recursive topology for this part if not enough
                # ports have been solved to go on and thus break topology=True:
                if self.break_topology:
                    return

            # check that at least machine epsilon has been added to llim or
            # ulim if one of them is zero and op routine includes division:
            if (
                topo_routine is not None
                and len(topo_routine) > 1
                and topo_routine[1] in (-4, 4, 5)
            ):
                # get limits if existing
                llim = getattr(self, '_llim', None)
                ulim = getattr(self, '_ulim', None)
                # check that any existing limits do not equal zero:
                err_lim = (
                    'While solving the topology for part `{0}` at port `{1}` '
                    'the following error occurred:\n'
                    '`{2}=0` found with a topology that requires division!\n'
                    'To solve this error, try the following steps:\n'
                    '  - 1. Try altering the topology of the system. Since '
                    'the topology is analyzed starting with/from pumps, '
                    'setting a pump to another branch usually solves the '
                    'problem. Since this problem mainly arises when a pump is '
                    'directly pushing into or pulling from the A or B port of '
                    'a mixing valve (or alike), which should be frowned upon '
                    'anyways due to realistic technic considerations, a '
                    'refactoring of the topology is recommended. As a last '
                    'measure, surrogate pumps may be introduced.\n'
                    '  - 2. Set a numeric slip value higher than '
                    'machine epsilon to the `{2}`. A good value '
                    'is `{2}>>1.5e-7`. Since the division with small values '
                    'will yield large values, setting a slip value of '
                    '**at least** 0.01 is recommended. Larger values '
                    'increase stability (especially of PID controls) and '
                    'stepsize while reducing the error. As long as the value '
                    'is acceptable: The larger the better.\n'
                )
                if llim is not None and llim == 0.0:
                    raise ValueError(
                        err_lim.format(self.name, port, 'lower_limit')
                    )
                if ulim is not None and ulim == 0.0:
                    raise ValueError(
                        err_lim.format(self.name, port, 'upper_limit')
                    )

            # get other connected port name:
            connected_port = self._models.port_links[current_port]
            # get other connected part name:
            connected_part = connected_port.split(';')[0]
            # if a BC has been reached, skip to next port:
            if connected_part == 'BoundaryCondition':
                continue
            # to avoid going back up in the recursive topology, check if this
            # part's port has already been solved and if yes, skip to next port
            if (
                connected_port.split(';')[1]
                in self._models.parts[connected_part]._solved_ports
            ):
                continue

            # if connected_part is another Pump (not parent_pump), raise Error:
            if self._models.parts[connected_part]._flow_net_parent and (
                self._models.parts[connected_part].name != parent_pump
            ):
                err_str = (
                    'There must not be two pumps in the same ' 'pump-network!'
                )
                raise TypeError(err_str)

            # CALL RECURSIVE TOPOLOGY AGAIN TO DELVE DEEPER INTO THE NET
            # if connected part is still in hydr_comps call recursive function
            # on connected part:
            if connected_part in self._models._hydr_comps:
                # if connected_part is NO part with multiple flow channels:
                if not self._models.parts[connected_part].multiple_flows:
                    # pass parent_pump to use correct flow net
                    # pass current_port as next parent_port to prevent going up
                    # in the tree again and to get values from correct parent
                    # port
                    # pass pump side (in/out) to know in next parts if it is on
                    # the pressure or suction side
                    self._models.parts[connected_part]._recursive_topology(
                        parent_pump,
                        sub_net,
                        pump_side=kwargs['pump_side'],
                        parent_port=current_port,
                        next_port=connected_port,
                    )
                else:
                    # else find flow channel:
                    for fc in self._models.parts[
                        connected_part
                    ]._flow_channels:
                        # if channel is in connected port, channel is found:
                        if fc in connected_port:
                            channel = fc  # save current flow channel
                    # now call next part in recursive topology with flow
                    # channel information:
                    self._models.parts[connected_part]._recursive_topology(
                        parent_pump,
                        sub_net,
                        pump_side=kwargs['pump_side'],
                        parent_port=current_port,
                        next_port=connected_port,
                        flow_channel=channel,
                    )

    def _calc_UAwll_port(self):
        """
        This method calculates the U*A value of the walls of port connections
        of parts. Since material properties of solids are assumed to be
        temperature independent for the used temperature ranges, the U*A values
        of the walls are constant and will be calculated before the iteration
        starts.

        """

        for part in self.parts.values():
            # calculate mean port cross section areas of own ports and
            # connected ports:
            if hasattr(part, '_A_p_wll_mean') and hasattr(
                part, '_A_p_fld_mean'
            ):  # if arrays already exist
                part._A_p_wll_mean[:] = (
                    part._A_wll_own_p + part._A_wll_conn_p
                ) / 2
                part._A_p_fld_mean[:] = (
                    part._A_fld_own_p + part._A_fld_conn_p
                ) / 2
            else:  # if arrays do not yet exist at part
                part._A_p_wll_mean = (
                    part._A_wll_own_p + part._A_wll_conn_p
                ) / 2
                part._A_p_fld_mean = (
                    part._A_fld_own_p + part._A_fld_conn_p
                ) / 2
            # set these values to zero where no port is:
            part._A_p_wll_mean[part._A_wll_conn_p == 0] = 0.0
            part._A_p_fld_mean[part._A_fld_conn_p == 0] = 0.0
            # get (constant) UA value from port-port walls (serial circuit):
            if not hasattr(part, '_UA_port_wll'):  # only create if not exist.
                part._UA_port_wll = np.zeros_like(part._A_p_wll_mean)
            mask = part._A_wll_conn_p != 0  # mask 0-division
            if type(part.grid_spacing) == np.ndarray:  # if only arrays
                part._UA_port_wll[
                    mask
                ] = part._A_p_wll_mean[  # only get non-0-division values
                    mask
                ] / (
                    +(part._port_gsp[mask] / (2 * part._lam_wll_conn_p[mask]))
                    + (part.grid_spacing[mask] / (2 * part._lam_wll[mask]))
                )
            else:  # if own variables are scalars
                part._UA_port_wll[
                    mask
                ] = part._A_p_wll_mean[  # only get non-0-division values
                    mask
                ] / (
                    +(part._port_gsp[mask] / (2 * part._lam_wll_conn_p[mask]))
                    + (part.grid_spacing / (2 * part._lam_wll))
                )

    def _update_FlowNet(self):
        """
        This method updates the massflows of each part and port following the
        operation instructions constructed by `_get_topology()`.

        Therefore it loops over the tuple **flow_net**. Each element in
        **flow_net** is a tuple itself, containing the massflow calculation
        instructions for each port. The calculation instructions are built up
        in the following way, with `X` being the index to the current port:
            - flow_net[X][0]:
                Target array cell to save the calculated massflow to.
            - flow_net[X][1]:
                Operation ID defining the mathematical operation to calculate
                the massflow. The following operations are currently supported:
                    -1:
                        Negative of the sum of the source array cells.
                    1:
                        Sum of the source array cells.
                    2:
                        Difference of the source array cells. Only works if
                        there are two source array cells.
                    3:
                        Multiplication of all source array cells. Currently
                        only works for two array cells.
                    4:
                        Division of source array cells. Only works if there are
                        two source array cells.
            - flow_net[X][2:]:
                Source array cells from which the target massflow will be
                calculated.
        """
        # loop over topology in flow net:
        for port in self.flow_net:
            # calc massflow depending on operation id. tuple buildup for each
            # port is the following:
            # element [1] of the tuple: operation id.
            # element [0]: memory view to target massflow cell (second [0]
            #              directly accesses its value)
            # element [2:]: memory view(s) to source massflow cell(s)
            if port[1] == 0:
                # op id 0: just passing on values! (indirectly accessing [2]
                # is 35% faster than directly accessing [2][0]!)
                port[0][0] = port[2]
            elif port[1] == -1:
                # op id -1: negative value of the sum of source cell(s)!
                # if only one source cell this is just the negative of it.
                # get first massflow (this is slightly faster than setting
                # [0][0] to zero and getting all massflows in loop without
                # skipping first loop!)
                port[0][0] = -port[2][0]
                # iterate over the views to the massflows but continue and
                # don't do anything for the first value (since it has
                # already been assigned):
                i = 0  # checker to ignore first loop
                for dm_view in port[2:]:
                    i += 1
                    # skip first loop:
                    if i == 1:
                        continue
                    port[0][0] -= dm_view[0]
            elif port[1] == 1:
                # op id 1: sum of source cells!
                port[0][0] = np.sum(port[2:])
            elif port[1] == -3:
                # op id -3: negative multiplication of source cells
                port[0][0] = -(port[2][0] * port[3][0])
            elif port[1] == -4:
                # op id -4: negative division of source cells!
                # (before 2, 3 and 4, since it will be applied more often)
                port[0][0] = -(port[2][0] / port[3][0])
            elif port[1] == 3:
                # op id 3: multiplication of source cells!
                # (before 2, since it will be applied more often)
                port[0][0] = port[2][0] * port[3][0]
            elif port[1] == 4:
                # op id 4: division of source cells!
                # (before 2, since it will be applied more often)
                port[0][0] = port[2][0] / port[3][0]
            elif port[1] == 2:
                # op id 1: difference of source cells!
                port[0][0] = port[2][0] - port[3][0]
            elif port[1] == 5:
                # op id 5: multi step multiplication with port factors to get
                # value from side-port to side-port for 3w-valves f.i.
                port[0][0] = port[2][0] * port[3][0] / port[4][0]

        # set bool in all parts that flows need to be updated:
        for part in self.parts.values():
            part._process_flows[0] = True

    def _update_control(self, timestep):
        """
        Update the controls of the simulation environment.
        """
        for controller, _ in enumerate(self.ctrl_l):
            self.ctrl_l[controller].ctrl(timestep)

    def _update_dyn_BC(self, time, timestep):
        """
        Update the dynamic boundary conditions.

        Update BCs by applying interpolation if the current timestep is
        between two steps or has skipped steps.
        """
        # only do this if there is something in dyn_BC:
        if self._dyn_BC:
            # check how many cells will be skipped by the timestep or if
            # interpolation is needed:
            cells = timestep / self.__BC_timedelta
            # new index:
            new_idx = self.__last_step_t_idx + cells
            # get interpolation conditions:
            # get index for new first cells after which the new timestep is:
            #            first_idx = int(round(new_idx))
            first_idx = int(new_idx)
            # get leverage for last two cells mean
            lev = new_idx - int(new_idx)
            # loop over boundary conditions:
            for BC in self._dyn_BC:
                # get the leveraged mean value of the steps between which the
                # current timestep is:
                lev_mean = (
                    BC[1][first_idx] * (1 - lev) + BC[1][first_idx + 1] * lev
                )
                # if less than or equal to one cell in the timestep index was
                # done in the last timestep:
                if cells <= 1:
                    # in this case the leveraged mean value is the new BC
                    # value, thus assign this to the memory view:
                    BC[0][0] = lev_mean
                else:
                    # else if this is not the case, then the time averaged mean
                    # of all jumped cells has to be calculated and to be added
                    # up with the time weighted leveraged mean:
                    BC[0][0] = (
                        BC[1][
                            int(round(self.__last_step_t_idx)) : int(new_idx)
                            + 1
                        ].mean()
                        * (cells - lev)
                        + lev_mean * lev
                    ) / cells
            # save new index as last step t idx:
            self.__last_step_t_idx = new_idx

    # @staticmethod
    def _get_conn_port(self, own_port, as_id=False):
        """
        Returns the part and port connected to 'own_port' either as a strings
        (with 'as_id=False', default) or as integer ids/indices when
        'as_id=True' is set. 'own_port' can also be given either as integer id
        or as a string.
        """

        # if own port given as id:
        if type(own_port) == int:
            # get id of own part:
            own_part_id = self._port_map_arr[own_port][0]
            # loop over parts to find the part and port with this id:
            for part in self.parts:
                if own_part_id == self.parts[part].part_id:
                    # get own part string
                    #                    own_part_str = self.parts[part].name
                    # get own port string as 'part;port' type
                    own_port = self.parts[part]._own_ports[
                        (np.where(self.parts[part].port_ids == own_port)[0][0])
                    ]
                    break

        # get other connected part and port name:
        conn_part, conn_port = self.port_links[own_port].split(';')

        # if as_id is True: return ids
        if as_id:
            conn_part_id = self.parts[conn_part].part_id
            conn_port_id = self.parts[conn_part].port_ids[
                self.parts[conn_part].port_names.index(conn_port)
            ]
            return conn_port_id, conn_part_id
        else:
            # return strings
            return conn_part, conn_port

    # =============================================================================
    #     @staticmethod
    #     def _get_str_or_id(part_id=None, port_id=None,
    #                        part_str=None, port_str=None):
    #         """
    #         Returns the part and/or port string or id, depending on the input type.
    #         If part and/or port were passed as integer id, part and or port
    #         identifier will be returned as string. Vice versa for part and/or port
    #         passed as string.
    #
    #         CAUTION: This is NOT returning the index of the ports, BUT rather the
    #         GLOBAL port ID. This is the unique ID of the port in the array of all
    #         ports added to the simulation environment.
    #         """
    #
    #         # assert correct types:
    #         err_str = ('If part and/or port were passed with `part_id` and/or '
    #                    '`port_id`, they must be passed as integer type!')
    #         assert part_id is None or type(part_id) == int, err_str
    #         assert port_id is None or type(port_id) == int, err_str
    #         err_str = ('If part and/or port were passed with `part_str` and/or '
    #                    '`port_str`, they must be passed as string type!')
    #         assert part_str is None or type(part_str) == str, err_str
    #         assert port_str is None or type(port_str) == str, err_str
    #         err_str = ('If port is passed with `port_str` and no `part_str` is '
    #                    'given, `port_str` needs to look like '
    #                    '`port_str=\'port;part\'`, also containing the part '
    #                    'identifier string!')
    #         if port_str is not None and part_str is None:
    #             assert len(port_str.split(';')) == 2, err_str
    #             # split up:
    #             part_str, port_str = port_str.split(';')
    #
    #         # get string identifier(s) if ids are given:
    #         if port_id is not None or part_id is not None:
    #             if port_id is None:
    #                 # if no port id is given, loop over parts to find part:
    #                 for part in self.parts:
    #                     if part_id == self.parts[part].part_id:
    #                         # only return part name
    #                         return self.parts[part].name
    #             else:
    #                 # if port id is given, part AND port can be found
    #                 if part_id is None:
    #                     # get part_id from port mapping array:
    #                     part_id = self._port_map_arr[port_id, 0]
    #                 # find part name by looping over parts:
    #                 for part in self.parts:
    #                     if part_id == self.parts[part].part_id:
    #                         # get 'part;port' string:
    #                         try:
    #                             partport = (
    #                                 self.parts[part]._own_ports[(
    #                                     np.where(
    #                                         self.parts[part].port_ids == port_id)
    #                                     [0][0])])
    #                         except IndexError:
    #                             # if port id was not found at the part, raise an
    #                             # error:
    #                             err_str = ('Port id ' + str(port_id) + ' was '
    #                                        'not found at the part with id '
    #                                        + str(part_id) + '!')
    #                             raise IndexError(err_str)
    #                         # split up:
    #                         part_str, port_str = partport.split(';')
    #                         # check if id of part and port are refering to the same
    #                         # part:
    # #                        if
    #                         # return both:
    #                         return part_str, port_str
    #         elif part_str is not None or port_str is not None:
    #             # now for the case that the string(s) are (is) given
    #             # if only part_str was given:
    #             if port_str is None:
    #                 # return only the part id:
    #                 return self.parts[part_str].part_id
    #             else:
    #                 # else port str is given
    #                 # get index of port:
    #                 local_idx = self.parts[part_str].port_names.index(port_str)
    #                 # get port id:
    #                 port_id = self.parts[part_str].port_ids[local_idx]
    #                 # return both:
    #                 return self.parts[part_str].part_id, port_id
    # =============================================================================

    # @staticmethod
    def _get_arr_idx(
        self,
        part,
        port,
        target=None,
        as_slice=False,
        flat_index=False,
        cls=None,
    ):
        """
        Returns the index to the array cell of the given port at the given
        part. This only works for actual ports, but not for standard indexes
        to arrays.
        If `as_slice=True` is set, the index will be returned as a slice,
        so that it will always construct a memory view to the position of the
        port.
        The parameter `target` defines which data array shall be indexed.
        Currently \'temperature\', \'massflow\' and \'port_specs\' (port
        specifications) are supported.

        Old:
        (`T_array=True` specifies that the temperature array shall be indexed
        for accessing the cells current temperature. If set to 'False', all
        other port values like massflow or cross section are will be indexed.)

        Parameters:
        -----------
        cls : Class
            If the part to get the indices from has not yet been added to
            SimEnv Class (for example when getting sport specifications while
            calling the class' __init__ method), the class instance has to be
            passed to this method within the `cls`-parameter.
        """

        # assert correct input types:
        # list of indices which are supported:
        sup_idx = ['temperature', 'massflow', 'port_specs']
        err_str = (
            '`target` must be a string which defines the array which '
            'is to be indexed. The following arrays can currently be '
            'indexed with the `_get_arr_idx()`-method:\n' + str(sup_idx)
        )
        assert target in sup_idx, err_str
        err_str = '`as_slice` must be given as a bool value!'
        assert type(as_slice) == bool, err_str

        # get parts dict depending on calling class:
        if self.__class__ != SimEnv:
            # Catch some strange error occurring randomly after import multisim several times...
            try:
                parts_dict = self._models.parts
            except AttributeError as e:
                print(e)
                breakpoint()
        else:
            parts_dict = self.parts

        if part in parts_dict:
            cls = parts_dict[part]
        else:
            assert (
                cls is not None
            ), 'Class must be given when part is not yet added to SimEnv!'

        # if part and port are given as ids, convert them to strings:
        if isinstance(part, int) and isinstance(port, int):
            raise NotImplementedError
            part, port = cls._get_str_or_id(part_id=part, port_id=port)
        elif type(part) != str or type(port) != str:
            err_str = (
                'Part and port must be both be either given as integer ID '
                '(the global port ID out of all ports in the simulation '
                'environmnent, not the local index to the value array!) or '
                'string identifier!'
            )
            raise TypeError(err_str)

        # assert if port exists in part:
        err_str = (
            'The port `{0}` was not found at part `{1}`.\n'
            'This error might also be caused by trying to pass a mix of an '
            'all-ports-identifier and port names to a function which can take '
            'multiple ports as parameters, for example when defining port '
            'specifications.'
        ).format(port, part)
        assert port in cls.port_names, err_str

        if target == 'temperature':  # for temperature
            if cls._T_port.ndim > 1:  # if 2D or 3D array
                if not flat_index:  # if unflattened index is requested
                    unflat_idx = np.unravel_index(
                        cls._port_own_idx_2D, cls._T_port.shape
                    )
                    idx_ax0 = unflat_idx[0][cls.port_names.index(port)]
                    idx_ax1 = unflat_idx[1][cls.port_names.index(port)]
                    if not as_slice:  # if no slice shall be returned
                        idx = (idx_ax0, idx_ax1)
                    else:  # if slice shall be returned
                        idx = (idx_ax0, slice(idx_ax1, idx_ax1 + 1))
                else:  # flat indexing is requested
                    idx_start = cls._port_own_idx_2D[
                        cls.port_names.index(port)
                    ]
                    if not as_slice:  # if no slice shall be returned
                        idx = idx_start
                    else:  # if slice shall be returned
                        idx = slice(idx_start, idx_start + 1)
            else:  # if 1D temperature array
                idx_start = cls._port_own_idx[cls.port_names.index(port)]
                if not as_slice:  # if no slice shall be returned
                    idx = idx_start
                else:  # if slice shall be returned
                    idx = slice(idx_start, idx_start + 1)
        elif target == 'massflow':
            if cls._T_port.ndim == 1:
                if cls.dm_invariant:
                    # if the massflow is invariant (all cells in the part
                    # (or flow channel) have the same massflow)
                    if not cls.multiple_flows:  # if no separate flows channels
                        if not as_slice:
                            # if no slice shall be returned
                            idx = 0
                        else:
                            # if slice shall be returned
                            idx = slice(0, 1)
                    else:  # else if part has separate flow channels (hex f.i.)
                        for fc in cls._flow_channels:  # find port's channel
                            if fc in port:  # fc is current port's flow channel
                                fc_idx = cls._flow_channels.index(fc)  # fc idx
                        # massflow target array cell is equivalent with flow
                        # channel index:
                        if not as_slice:
                            # if no slice shall be returned
                            idx = fc_idx
                        else:
                            # if slice shall be returned
                            idx = slice(fc_idx, fc_idx + 1)
                elif cls._dm_io.size == cls.port_num:
                    # if the massflow arrays have been collapsed, their size
                    # is equal to the number of ports on that part.
                    # Thus the index to get is the number of the port:
                    if not cls.multiple_flows:  # if no separate flows channels
                        # location of port:
                        idx_start = cls.port_names.index(port)
                        if not as_slice:
                            # if no slice shall be returned
                            idx = idx_start
                        else:
                            # if slice shall be returned
                            idx = slice(idx_start, idx_start + 1)
                    else:
                        raise NotImplementedError
                        # loop over flow channels to find the port's channel
                        for fc in cls._flow_channels:  # find port's channel
                            if fc in port:  # fc is current port's flow channel
                                fc_idx = cls._flow_channels.index(fc)  # fc idx
                        # AB HIER IST ES NOCH FALSCH:
                        # wie verheirate ich flow channel index mit
                        # port spezifischem index?
                        # return either as tuple or as slice index:
                        if not as_slice:
                            # if no slice shall be returned
                            idx = fc_idx
                        else:
                            # if slice shall be returned
                            idx = slice(fc_idx, fc_idx + 1)
                else:  # else if different massflows in source part can exist
                    # get location to port cell in massflow grid. This is the
                    # same for parts with and/or without flow channels!
                    idx_start = cls._port_own_idx[cls.port_names.index(port)]
                    if not as_slice:
                        # if no slice shall be returned
                        idx = idx_start
                    else:
                        # if slice shall be returned
                        idx = slice(idx_start, idx_start + 1)
            else:  # ndim > 1, double ports or multiple flows for example
                if not flat_index and not cls.multiple_flows:
                    # 2D-non-flat indexing and NO multiple flow channels!
                    # if double ports have been added but no multiple flows
                    # exist, get the unflattened 2D indices:
                    unflat_idx = np.unravel_index(
                        cls._port_own_idx_2D, cls._T_port.shape
                    )
                    # if double ports have been added and thus all grids
                    # have more rows AND the massflow or port_specs (2D for
                    # some cases) are requested:
                    idx_ax0 = unflat_idx[0][cls.port_names.index(port)]
                    idx_ax1 = unflat_idx[1][cls.port_names.index(port)]
                    # return either as tuple or as slice index:
                    if not as_slice:
                        # if no slice shall be returned
                        idx = (idx_ax0, idx_ax1)
                    else:
                        # if slice shall be returned
                        idx = (idx_ax0, slice(idx_ax1, idx_ax1 + 1))
                elif flat_index and not cls.multiple_flows:
                    # else if flat indexing AND NO multiple flow channels
                    idx_start = cls._port_own_idx_2D[
                        cls.port_names.index(port)
                    ]
                    # return either as tuple or as slice index:
                    if not as_slice:
                        # if no slice shall be returned
                        idx = idx_start
                    else:
                        # if slice shall be returned
                        idx = slice(idx_start, idx_start + 1)
                else:  # else if multiple flow channels:
                    # loop over flow channels to find the port's channel
                    for fc in cls._flow_channels:  # find port's channel
                        if fc in port:  # fc is current port's flow channel
                            fc_idx = cls._flow_channels.index(fc)  # fc idx
                    # return either as tuple or as slice index:
                    if not as_slice:
                        # if no slice shall be returned
                        idx = fc_idx
                    else:
                        # if slice shall be returned
                        idx = slice(fc_idx, fc_idx + 1)
        elif target == 'port_specs':
            if cls._T_port.ndim == 1:
                # if different massflows in source part can exist and/or
                # temperature or port specs are requested
                if cls._collapsed:  # if port arrays are collapsed
                    # then the index is the number of the port
                    idx_start = cls.port_names.index(port)
                else:  # else if port arrays are not collapsed, get normal idx
                    idx_start = cls._port_own_idx[cls.port_names.index(port)]
                if not as_slice:
                    # if no slice shall be returned
                    idx = idx_start
                else:
                    # if slice shall be returned
                    idx = slice(idx_start, idx_start + 1)
            else:  # ndim > 2, not existing if collapsed
                if not flat_index:  # 2D-non-flat indexing
                    # if double ports have been added, get the unflattened 2D
                    # indices:
                    unflat_idx = np.unravel_index(
                        cls._port_own_idx_2D, cls._T_port.shape
                    )
                    # if double ports have been added and thus all grids have
                    # more rows AND the massflow or port_specs (2D for some
                    # cases) are requested:
                    idx_ax0 = unflat_idx[0][cls.port_names.index(port)]
                    idx_ax1 = unflat_idx[1][cls.port_names.index(port)]
                    # return either as tuple or as slice index:
                    if not as_slice:
                        # if no slice shall be returned
                        idx = (idx_ax0, idx_ax1)
                    else:
                        # if slice shall be returned
                        idx = (idx_ax0, slice(idx_ax1, idx_ax1 + 1))
                else:
                    # flat indexing:
                    idx_start = cls._port_own_idx_2D[
                        cls.port_names.index(port)
                    ]
                    # return either as tuple or as slice index:
                    if not as_slice:
                        # if no slice shall be returned
                        idx = idx_start
                    else:
                        # if slice shall be returned
                        idx = slice(idx_start, idx_start + 1)

        #        if cls._T_port.ndim == 1 and target != 'temperature':
        #            # if no double ports have been added to source part or the
        #            # temperature (1D for all cases) is requested
        #            if cls.dm_invariant and target == 'massflow':
        #                # if the massflow is invariant (all cells in the part have the
        #                # same massflow) and requested
        #                if not as_slice:
        #                    # if no slice shall be returned
        #                    idx = 0
        #                else:
        #                    # if slice shall be returned
        #                    idx = slice(0, 1)
        #            else:
        #                # if different massflows in source part can exist and/or
        #                # temperature or port specs are requested
        #                idx_start = (cls._port_own_idx[cls.port_names.index(port)])
        #                if not as_slice:
        #                    # if no slice shall be returned
        #                    idx = idx_start
        #                else:
        #                    # if slice shall be returned
        #                    idx = slice(idx_start, idx_start + 1)
        #        elif (cls._T_port.ndim == 2  # if double ports have been added
        #              and (target == 'massflow' or target == 'port_specs')):
        #            if not flat_index:  # 2D-non-flat indexing
        #                # if double ports have been added, get the unflattened 2D
        #                # indices:
        #                unflat_idx = np.unravel_index(cls._port_own_idx_2D,
        #                                              cls._T_port.shape)
        #                # if double ports have been added and thus all grids have more
        #                # rows AND the massflow or port_specs (2D for some cases) are
        #                # requested:
        #                idx_ax0 = (unflat_idx[0][cls.port_names.index(port)])
        #                idx_ax1 = (unflat_idx[1][cls.port_names.index(port)])
        #                # make tuple out of it to be able to return as a single value:
        #                if not as_slice:
        #                    # if no slice shall be returned
        #                    idx = (idx_ax0, idx_ax1)
        #                else:
        #                    # if slice shall be returned
        #                    idx = (idx_ax0, slice(idx_ax1, idx_ax1 + 1))
        #            else:
        #                # flat indexing:
        #                idx_start = cls._port_own_idx_2D[cls.port_names.index(port)]
        #                # make tuple out of it to be able to return as a single value:
        #                if not as_slice:
        #                    # if no slice shall be returned
        #                    idx = idx_start
        #                else:
        #                    # if slice shall be returned
        #                    idx = slice(idx_start, idx_start + 1)
        #        elif target == 'temperature':  # if index to temp. grid is requested
        #            if cls._T_port.ndim > 1:  # if 2D or 3D array
        #                if not flat_index:  # if unflattened index is requested
        #                    unflat_idx = np.unravel_index(cls._port_own_idx_2D,
        #                                                  cls._T_port.shape)
        #                    idx_ax0 = (unflat_idx[0][cls.port_names.index(port)])
        #                    idx_ax1 = (unflat_idx[1][cls.port_names.index(port)])
        #                    if not as_slice:  # if no slice shall be returned
        #                        idx = (idx_ax0, idx_ax1)
        #                    else:  # if slice shall be returned
        #                        idx = (idx_ax0, slice(idx_ax1, idx_ax1 + 1))
        #                else:  # flat indexing is requested
        #                    idx_start = cls._port_own_idx_2D[
        #                            cls.port_names.index(port)]
        #                    if not as_slice:  # if no slice shall be returned
        #                        idx = idx_start
        #                    else:  # if slice shall be returned
        #                        idx = slice(idx_start, idx_start + 1)
        #            else:  # if 1D temperature array
        #                idx_start = (cls._port_own_idx[cls.port_names.index(port)])
        #                if not as_slice:  # if no slice shall be returned
        #                    idx = idx_start
        #                else:  # if slice shall be returned
        #                    idx = slice(idx_start, idx_start + 1)

        return idx

    #    @staticmethod
    #    def _get_arr_idx(part, port, target=None,
    #                     as_slice=False, flat_index=False, cls=None):
    #        """
    #        Returns the index to the array cell of the given port at the given
    #        part. This only works for actual ports, but not for standard indexes
    #        to arrays.
    #        If `as_slice=True` is set, the index will be returned as a slice,
    #        so that it will always construct a memory view to the position of the
    #        port.
    #        The parameter `target` defines which data array shall be indexed.
    #        Currently \'temperature\', \'massflow\' and \'port_specs\' (port
    #        specifications) are supported.
    #
    #        Old:
    #        (`T_array=True` specifies that the temperature array shall be indexed
    #        for accessing the cells current temperature. If set to 'False', all
    #        other port values like massflow or cross section are will be indexed.)
    #
    #        Parameters:
    #        -----------
    #        cls : Class
    #            If the part to get the indices from has not yet been added to
    #            SimEnv Class (for example when getting sport specifications while
    #            calling the class' __init__ method), the class instance has to be
    #            passed to this method within the `cls`-parameter.
    #        """
    #
    #        # assert correct input types:
    #        # list of indices which are supported:
    #        sup_idx = ['temperature', 'massflow', 'port_specs']
    #        err_str = ('`target` must be a string which defines the array which '
    #                   'is to be indexed. The following arrays can currently be '
    #                   'indexed with the `_get_arr_idx()`-method:\n'
    #                   + str(sup_idx))
    #        assert target in sup_idx, err_str
    #        err_str = ('`as_slice` must be given as a bool value!')
    #        assert type(as_slice) == bool, err_str
    #
    #        if part in self.parts:
    #            cls = self.parts[part]
    #        else:
    #            assert cls is not None, ('Class must be given when part is not '
    #                                     'yet added to SimEnv!')
    #
    #        # if part and port are given as ids, convert them to strings:
    #        if type(part) == int and type(port) == int:
    #            part, port = cls._get_str_or_id(part_id=part, port_id=port)
    #        elif type(part) != str or type(port) != str:
    #            err_str = (
    #                'Part and port must be both be either given as integer ID '
    #                '(the global port ID out of all ports in the simulation '
    #                'environmnent, not the local index to the value array!) or '
    #                'string identifier!')
    #            raise TypeError(err_str)
    #
    #        # assert if port exists in part:
    #        err_str = ('The port ' + port + ' was not found at part ' + part +
    #                   '! This error might also be caused by trying to pass '
    #                   'a mix of an all-ports-identifier and port names to a '
    #                   'function which can take multiple ports as parameters, '
    #                   'for example when defining port specifications.')
    #        assert port in cls.port_names, err_str
    #
    #        if cls._T_port.ndim == 1 and target != 'temperature':
    #            # if no double ports have been added to source part or the
    #            # temperature (1D for all cases) is requested
    #            if cls.dm_invariant and target == 'massflow':
    #                # if the massflow is invariant (all cells in the part have the
    #                # same massflow) and requested
    #                if not as_slice:
    #                    # if no slice shall be returned
    #                    idx = 0
    #                else:
    #                    # if slice shall be returned
    #                    idx = slice(0, 1)
    #            else:
    #                # if different massflows in source part can exist and/or
    #                # temperature or port specs are requested
    #                idx_start = (cls._port_own_idx[cls.port_names.index(port)])
    #                if not as_slice:
    #                    # if no slice shall be returned
    #                    idx = idx_start
    #                else:
    #                    # if slice shall be returned
    #                    idx = slice(idx_start, idx_start + 1)
    #        elif (cls._T_port.ndim == 2  # if double ports have been added
    #              and (target == 'massflow' or target == 'port_specs')):
    #            if not flat_index:  # 2D-non-flat indexing
    #                # if double ports have been added, get the unflattened 2D
    #                # indices:
    #                unflat_idx = np.unravel_index(cls._port_own_idx_2D,
    #                                              cls._T_port.shape)
    #                # if double ports have been added and thus all grids have more
    #                # rows AND the massflow or port_specs (2D for some cases) are
    #                # requested:
    #                idx_ax0 = (unflat_idx[0][cls.port_names.index(port)])
    #                idx_ax1 = (unflat_idx[1][cls.port_names.index(port)])
    #                # make tuple out of it to be able to return as a single value:
    #                if not as_slice:
    #                    # if no slice shall be returned
    #                    idx = (idx_ax0, idx_ax1)
    #                else:
    #                    # if slice shall be returned
    #                    idx = (idx_ax0, slice(idx_ax1, idx_ax1 + 1))
    #            else:
    #                # flat indexing:
    #                idx_start = cls._port_own_idx_2D[cls.port_names.index(port)]
    #                # make tuple out of it to be able to return as a single value:
    #                if not as_slice:
    #                    # if no slice shall be returned
    #                    idx = idx_start
    #                else:
    #                    # if slice shall be returned
    #                    idx = slice(idx_start, idx_start + 1)
    #        elif target == 'temperature':  # if index to temp. grid is requested
    #            if cls._T_port.ndim > 1:  # if 2D or 3D array
    #                if not flat_index:  # if unflattened index is requested
    #                    unflat_idx = np.unravel_index(cls._port_own_idx_2D,
    #                                                  cls._T_port.shape)
    #                    idx_ax0 = (unflat_idx[0][cls.port_names.index(port)])
    #                    idx_ax1 = (unflat_idx[1][cls.port_names.index(port)])
    #                    if not as_slice:  # if no slice shall be returned
    #                        idx = (idx_ax0, idx_ax1)
    #                    else:  # if slice shall be returned
    #                        idx = (idx_ax0, slice(idx_ax1, idx_ax1 + 1))
    #                else:  # flat indexing is requested
    #                    idx_start = cls._port_own_idx_2D[
    #                            cls.port_names.index(port)]
    #                    if not as_slice:  # if no slice shall be returned
    #                        idx = idx_start
    #                    else:  # if slice shall be returned
    #                        idx = slice(idx_start, idx_start + 1)
    #            else:  # if 1D temperature array
    #                idx_start = (cls._port_own_idx[cls.port_names.index(port)])
    #                if not as_slice:  # if no slice shall be returned
    #                    idx = idx_start
    #                else:  # if slice shall be returned
    #                    idx = slice(idx_start, idx_start + 1)
    #
    #        return idx

    def _massflow_idx_from_temp_idx(self, part, index):
        """
        This function transforms an index of the temperature array to an index
        of a massflow array.
        """

        # check for existence of index in temperature array
        self._check_isinrange(
            part=part, index=index, target_array='temperature'
        )
        # get index to massflow array
        if self.parts[part].dm_invariant:  # if same massflow in flow channel
            return 0
        else:
            return index

    def _get_specs_n_props(
        self,
        material=None,
        pipe_specs=None,
        material_only=False,
        pipe_specs_only=False,
        **kwargs
    ):
        """
        Extract pipe specifications and material properties.

        Pipe specifications and material properties of parts are extracted from
        data tables. Errors in the specifications are checked for. Collected
        values are saved to calling part.
        """
        # assert that the bool values are really bool values:
        err_str = (
            '`material_only` or `pipe_specs_only` was not passed as a '
            'bool value! The following values were passed (False is '
            'the default value): `material_only='
            + str(material_only)
            + '`, `pipe_specs_only='
            + str(pipe_specs_only)
            + '`.'
        )
        assert isinstance(material_only, bool) and isinstance(
            pipe_specs_only, bool
        ), err_str

        # if part has no info topology yet, construct it:
        if not hasattr(self, 'info_topology'):
            self.info_topology = dict()

        # get material if not only pipe specs are requested:
        if not pipe_specs_only:
            # assert and get material specs:
            err_str = (
                '`material` must be given and has to be one of the '
                'following:\n'
                + str(list(self._models._mprop_sld.columns))
                + '\nNew materials can be added to `mat_props.xlsx` '
                'and then passed to the function `update_databases()` of '
                'the `utility`-package if required.'
            )
            # assert that material was given:
            if material is None:
                assert 'material' in kwargs, err_str
            # assert that correct material:
            assert material in self._models._mprop_sld.columns, err_str
            # get material properties:
            self._cp_wll = float(self._models._mprop_sld[material].c_p)
            self._rho_wll = float(self._models._mprop_sld[material].rho)
            # heat conductivity of the wall is constant in all parts, but
            # non-numeric parts save and pass on the lambda value of connected
            # parts to enable heat flux calculation, thus in non-numeric parts
            # lam wall is an array of length of number of ports which will be
            # pre-filled here and filled by get_port_connections() method:
            if self.solve_numeric:
                self._lam_wll = float(
                    self._models._mprop_sld[material]['lambda']
                )
            else:
                self._lam_wll = np.full_like(
                    self._T_port,
                    float(self._models._mprop_sld[material]['lambda']),
                )
            # save material information to info dict:
            if 'all_ports' not in self.info_topology:
                # create empty port sub-dict if not yet existing:
                self.info_topology['all_ports'] = dict()
            self.info_topology['all_ports'].update(
                {
                    'material': material,
                    'lambda material [W/(mK)]': self._lam_wll,
                    'rho material [kg/m^3]': self._rho_wll,
                    'c_p material [J/(kgK)]': self._cp_wll,
                }
            )

        # error string for asserting pipe specs:
        err_str = (
            self._base_err
            + self._arg_err.format('pipe_specs')
            + 'The nominal diameter and pipe type of connection ports has to be '
            'given for each part with \'pipe_specs=...\'.\n\n'
            'If ALL ports have the SAME specifications (also including ports '
            'added manually), `pipe_specs` must be a 2-level dict with the '
            'first-level-key \'all\' as declaration that all ports have the '
            'same specifications. The pipe type \'pipe_type=...\' and the '
            'nominal diameter \'DN=...\' have to be specified in the second '
            'level of the dict. The dict must then look like:\n'
            'pipe_specs={\'all\': {\'pipe_type\': \'EN10255-medium\', '
            '\'DN\': \'DN50\'}}\n\n'
            'If at least one port has a specific value, the specifications '
            'for all ports have to be passed separately.\n'
            'The passed \'pipe_specs\'-dict must be a 2-level dict and '
            'look like:\n'
            'pipe_specs={\'in\': {\'pipe_type\': \'EN10255-medium\', '
            '\'DN\': \'DN50\'},\n'
            '            \'out\': {\'pipe_type\': \'EN10255-heavy\', '
            '\'DN\': \'DN65\'}}\n'
            'Here `in` and `out` are the respective port identifiers. '
            'For other port names, these need to match the identifiers.'
        )
        # get pipe specs if not only material is requested:
        if not material_only:
            # assert pipe specs:
            # if pipe specs is not given, it must be passt by kwargs:
            if pipe_specs is None:
                assert 'pipe_specs' in kwargs, err_str
                pipe_specs = kwargs['pipe_specs']
            # assert correct type:
            assert isinstance(pipe_specs, dict), err_str
            # assert all port specs keys:
            for key in pipe_specs.keys():
                assert 'pipe_type' in pipe_specs[key].keys() and (
                    ('DN' in pipe_specs[key].keys())
                    or (
                        'A_i' in pipe_specs[key].keys()
                        and 'A_wall' in pipe_specs[key].keys()
                    )
                ), err_str

            # loop over given pipe_specs to extract them:
            for key, value in pipe_specs.items():
                # extract pipe type and DN:
                pipe_type = value['pipe_type']
                if 'DN' in value.keys():
                    DN = value['DN']
                    get_from_table = True
                else:  # else take it directly!
                    get_from_table = False
                    DN = 'self defined, Ai: {0:.3G}, Awall: {1:.3G}'.format(
                        value['A_i'], value['A_wall']
                    )
                # assert correct pipe specs:
                err_str = (
                    self._base_err
                    + self._arg_err.format('pipe_type')
                    + '`pipe_type` must be given and has to be one of '
                    'the following:\n'
                    + str(list(self._models._pipe_specs.columns.levels[0]))
                    + '.\nOther pipe types can be added to '
                    '`pipe_specs.xlsx` and then passed to the function '
                    '`update_databases()` of the `utility`-package if '
                    'required.'
                )
                assert pipe_type in self._models._pipe_specs.columns, err_str
                if get_from_table:
                    # assert correct DN:
                    err_str = (
                        self._base_err
                        + self._arg_err.format('pipe_type')
                        + 'The pipes nominal diameter has to be given like '
                        '`DN=\'DN50\'`. If the diameter was not found but is '
                        'required, the table with supported pipe specs in '
                        '`pipe_specs.xlsx` can be extended and then passed to the '
                        'function `update_databases()` of the `utility`-module.'
                    )
                    assert (
                        DN in self._models._pipe_specs[pipe_type].columns
                    ), err_str

                # check if only key is 'all' and if yes, get values
                if 'all' in pipe_specs and len(pipe_specs.keys()) == 1:
                    # save them to single flaot value since all ports have the
                    # same specs AND if they are not non-numeric parts:
                    if self.solve_numeric:
                        if get_from_table:
                            self._A_wll_own_p = float(
                                self._models._pipe_specs[pipe_type, DN].A_wall
                            )
                            self._A_fld_own_p = float(
                                self._models._pipe_specs[pipe_type, DN].A_i
                            )
                        else:
                            self._A_wll_own_p = float(value['A_wall'])
                            self._A_fld_own_p = float(value['A_i'])
                    else:
                        # else if part is solved non-numerically, the part has
                        # to pass on the values of the connected parts, thus
                        # these areas will be stored in an array which is
                        # filled with own values now and will be filled with
                        # connected part values, if available, in
                        # get_port_connections():
                        if get_from_table:
                            self._A_wll_own_p = np.full_like(
                                self._T_port,
                                float(
                                    self._models._pipe_specs[
                                        pipe_type, DN
                                    ].A_wall
                                ),
                            )
                            self._A_fld_own_p = np.full_like(
                                self._T_port,
                                float(
                                    self._models._pipe_specs[pipe_type, DN].A_i
                                ),
                            )
                        else:
                            self._A_wll_own_p = np.full_like(
                                self._T_port, float(value['A_wall'])
                            )
                            self._A_fld_own_p = np.full_like(
                                self._T_port, float(value['A_i'])
                            )
                    # save pipe spec information to info dict:
                    if 'all_ports' not in self.info_topology:
                        # create empty port sub-dict if not yet existing:
                        self.info_topology['all_ports'] = dict()
                    self.info_topology['all_ports'].update(
                        {
                            'pipe_type': pipe_type,
                            'DN': DN,
                            'pipe_specs': dict(
                                self._models._pipe_specs[pipe_type, DN]
                            )
                            if get_from_table
                            else value,
                        }
                    )
                else:
                    # else other port specs are given, save them to each port's
                    # array position
                    # get target port index:
                    trgt_idx = self._get_arr_idx(
                        self.name,
                        key,
                        target='port_specs',
                        as_slice=False,
                        cls=self,
                    )
                    # check if target arrays already exist and is an array of
                    # the same shape as the port temperature array
                    # (then only add specs to array) and if not yet existing
                    # construct new arrays:
                    if not (
                        hasattr(self, '_A_wll_own_p')
                        and isinstance(self._A_wll_own_p, np.ndarray)
                        and self._A_wll_own_p.shape == self._T_port.shape
                    ):
                        # construct a new array:
                        self._A_wll_own_p = np.zeros_like(self._T_port)
                        self._A_fld_own_p = np.zeros_like(self._T_port)
                    # save specs to arrays at index:
                    if get_from_table:
                        self._A_wll_own_p[
                            trgt_idx
                        ] = self._models._pipe_specs.loc[pipe_type, DN][
                            'A_wall'
                        ]
                        self._A_fld_own_p[
                            trgt_idx
                        ] = self._models._pipe_specs.loc[pipe_type, DN]['A_i']
                    else:
                        self._A_wll_own_p[trgt_idx] = value['A_wall']
                        self._A_fld_own_p[trgt_idx] = value['A_i']
                    # save pipe spec information to info dict:
                    if key not in self.info_topology:
                        # create empty port sub-dict if not yet existing:
                        self.info_topology[key] = dict()
                    self.info_topology[key].update(
                        {
                            'pipe_type': pipe_type,
                            'DN': DN,
                            'pipe_specs': (
                                self._models._pipe_specs.loc[pipe_type, DN]
                            )
                            if get_from_table
                            else value,
                        }
                    )
            # now check again to assert that all ports have received their
            # specs:
            if 'all' not in pipe_specs:
                err_str = (
                    'Not all ports in part ' + self.name + ' have been '
                    'provided with pipe specifications!'
                )
                assert len(pipe_specs) == self.port_num, err_str

    def _get_topo_cond(self, port, parent_port=None):
        """
        Returns the connection conditions for the topology analyzation for the
        current part at the current port.
        **The source part's identifiers and indices can only be used if the
        value is simply passed on from the parent part, otherwise it has to be
        constructed separately for each part** in the respective target part's
        `get_flow_routine()`-method.

        The returns include:
        - Index/Indices to create memory view to the target massflow array
          cell. Depending on the dimensions of the arrays including ports
          either a single index for 1D indexing is returned or a tuple of
          indices is returned for 2D indexing. For the 1D case it is also
          checked if the massflow is invariant within the target part and if
          so, this will be regarded when constructing the index. Using only a
          memory view to automatically copy source port's values **IS NOT
          CHECKED FOR** and thus must be implemented manually in the target
          part's `get_flow_routine()`-method.
        - Split up name strings of the source part and port, returning source
          part's name and source port's name as separate strings.
        - Index/Indices to create memory view to the source massflow array
          cell. Depending on the dimensions of the arrays including ports
          either a single index for 1D indexing is returned or a tuple of
          indices is returned for 2D indexing. For the 1D case it is also
          checked if the massflow is invariant within the source part and if
          so, this will be regarded when constructing the index. Using multiple
          source port's to calculate the target port's value is **NOT
          SUPPORTED** within this method. This must be implemented manually in
          the target part's `get_flow_routine()`-method.
        - Algebraic sign of the massflow depending on the predefined positive
          flow direction through a port. A positive algebraic sign will be
          returned when the massflow characteristic (the flow direction through
          a port for positive massflow) is contrary in target and source port,
          meaning that an 'in' port is connected to an 'out' port. When similar
          ports are connected, a negative algebraic sign will be returned.

        Parameters:
        -----------
        port : string
            Target port of which the topology analyzer is trying to find the
            solving method.
        parent_port : string
            Identifier of the port where the topology analyzer is coming from.
            This string is constructed of the parent part's part name and port
            name, separated by a semicolon: 'part;port'. For cases where the
            massflow value is passed on (either positive or negative), this
            also is the source part/port. For other cases **this does not
            apply** and the source part/port has to be constructed manually
            in the target part's `get_flow_routine()`-method. If no parent_port
            is passed, only trgt_idx will be returned as not None!

        Returns:
        --------
        trgt_idx : int, tuple of int
            The index to the target port's (or for invariant massflow: part's)
            massflow array cell. If no double ports have been added to the
            part, this is a single integer value. If double ports have been
            added to the part, this is a tuple of two integers, the first
            indexing axis 0, the second indexing axis 1.
        src_part : string
            Source part's string identifier **if the parent part is the source
            part** (when just passing on the massflow as pos. or neg. value).
        src_port : string
            Source part's port string identifier **if the parent part is the
            source part** (when just passing on the massflow as pos. or neg.
            value).
        src_idx : int, tuple of int
            The index to the source port's (or for invariant massflow: part's)
            massflow array cell. If no double ports have been added to the
            part, this is a single integer value. If double ports have been
            added to the part, this is a tuple of two integers, the first
            indexing axis 0, the second indexing axis 1.
        alg_sign : string
            String with the information if the passed on value can be passed on
            with the same algebraic sign ('positive' is returned) or if the
            negative of the value has to be passed on ('negative' is returned).

        """

        # get port index of target port as slice to create memory view
        # depending on number of dimensions of the massflow array grid:
        if self._T_port.ndim == 1:
            # if no double ports have been added to target part, get index to
            # 1D array:
            trgt_idx = self._get_arr_idx(
                self.name, port, 'massflow', as_slice=True
            )
        elif self._T_port.ndim == 2:
            # if double ports have been added and thus all grids have more than
            # one dimension, get flat slice index to 2D array:
            trgt_idx = self._get_arr_idx(
                self.name, port, 'massflow', as_slice=True, flat_index=True
            )

        # the following gets the parent/source port's identifiers and indices,
        # if parent_port is passed:
        if parent_port is not None:
            # get source_part: parent_port is a string consisting of
            # 'src_part;src_port'. get each value out of it:
            src_part, src_port = parent_port.split(';')

            # get source index of source port's massflow cell as slice:
            if self._models.parts[src_part]._T_port.ndim == 1:
                # if no double ports have been added to source part, get slice
                # index to 1D source array:
                src_idx = self._get_arr_idx(
                    src_part,
                    src_port,
                    'massflow',
                    as_slice=True,
                    flat_index=True,
                )
            elif self._models.parts[src_part]._T_port.ndim == 2:
                # if double ports have been added and thus all grids have more
                # than one dimension get flat slice index to 2D source array:
                src_idx = self._get_arr_idx(
                    src_part,
                    src_port,
                    'massflow',
                    as_slice=True,
                    flat_index=True,
                )

            # check port setup to get algebraic sign of flow:
            # find port number of target and source port in part (index into
            # tuples with port names and characteristics)and use this to get
            # the positive massflow direction:
            trgt_prt_chr = self.dm_char[self.port_names.index(port)]
            src_prt_chr = self._models.parts[src_part].dm_char[
                self._models.parts[src_part].port_names.index(src_port)
            ]

            # check for algebraic sign of connected ports:
            if trgt_prt_chr != src_prt_chr:
                # if both ports have opposing massflow characteristics (out to
                # in or vice versa), the passed massflow has the same algebraic
                # sign in both parts, thus is positive
                alg_sign = 'positive'
            else:
                # if both ports have the same massflow characteristics (in to
                # in or out to out), the passed massflow has a contrary
                # algebraic sign in both parts, thus is negative:
                alg_sign = 'negative'

            return trgt_idx, src_part, src_port, src_idx, alg_sign
        else:
            # if parent_port is not passed, only trgt_idx will be returned
            return trgt_idx

    @staticmethod
    def _angle_to_x_axis(vector):
        """
        Returns the angle between the 2D-vector `vector` and the x-axis with
        unit vector (1 0) of the cartesian coordinate system. The angle is
        returned as a **radian** angle.
        """

        #        unit_vec = np.array([1, 0])

        #        angle = (np.arccos(np.dot(vector, unit_vec)
        #                 / (np.sqrt((vector * vector).sum())
        #                    * np.sqrt((unit_vec * unit_vec).sum()))))
        # atan does the same with atan2(y, x):
        angle = _math.atan2(vector[1], vector[0])

        return angle

    @staticmethod
    def _rotate_vector(vector, angle):
        """
        Rotates a 2D vector.

        The `vector` is rotated around the **radian** `angle`. The resulting
        vector is returned.
        """
        # create rotation matrix (only valid for 2d vector rotation around
        # z axis with e-vector (0 0 1):
        R = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        # rotate vector:
        vector_rot = np.dot(R, vector)

        return vector_rot

    def _heun(self, solve_num_expl, solve_nonnum, timestep):
        """
        Calculate the result of a differential equation using Heun's method.

        Non adaptive Heun solver. Mixing with implicit steps is not allowed.

        Parameters
        ----------
        solve_num_expl : TYPE
            DESCRIPTION.
        solve_nonnum : TYPE
            DESCRIPTION.
        timestep : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # solve predictor step for all numeric parts:
        for part in solve_num_expl:
            # get results from last timestep and pass them to
            # current-timestep-temperature-array:
            self.parts[part].T[:] = self.parts[part].res[self.stepnum[0] - 1]
            # calculate differential at result:
            self.parts[part]._df0 = solve_num_expl[part]()
            # calculate and save predictor step:
            self.parts[part].T[:] = (
                self.parts[part].T + timestep * self.parts[part]._df0
            )
        # solve predictor step for all non-numeric parts:
        for part in solve_nonnum:
            self.solve_nonnum[part]()
        # update ports with new values:
        self._update_ports(source='intermediate', all_ports=True, nnum=False)
        self._update_ports(source='intermediate', all_ports=True, nnum=True)
        # solve corrector step for all numeric parts:
        for part in solve_num_expl:
            # calculate differential at predictor step result:
            self.parts[part]._df1 = solve_num_expl[part]()
            # solve heun method and save to result:
            self.parts[part].res[self.stepnum[0]] = self.parts[part].res[
                self.stepnum[0] - 1
            ] + (timestep / 2) * (
                self.parts[part]._df0 + self.parts[part]._df1
            )
            # also save to T arrays to be able to use memoryviews
            self.parts[part].T[:] = self.parts[part].res[self.stepnum[0]]
        # solve corrector step for all non-numeric parts:
        for part in solve_nonnum:
            self.solve_nonnum[part]()
        # update numeric parts' ports with results:
        self._update_ports(source='results', all_ports=True, nnum=False)
        # update non-numeric ports (they only have intermediate values as they
        # have no differential method!)
        self._update_ports(source='intermediate', all_ports=True, nnum=True)

    def _heun_adaptive(
        self, solve_num_expl, solve_num_impl, solve_nonnum, old_step
    ):
        """
        Solve a differential equation using adaptive stepping Heun's method.

        Adaptive steps and mixing with implicit solver allower.

        Parameters
        ----------
        solve_num_expl : TYPE
            DESCRIPTION.
        solve_num_impl : TYPE
            DESCRIPTION.
        solve_nonnum : TYPE
            DESCRIPTION.
        old_step : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # get new step size from last adaptive step:
        _h = self._new_step

        # get often accessed vars as local vars to increase the speed:
        stepnum = self.stepnum[0]
        rtol = self.__rtol
        atol = self.__atol
        # save port values to backup array (COPY to avoid memoryview!) to
        # be able to reset them when the step is not accepted:
        self.__parr_bkp[:] = self.ports_all
        # set step to not accepted to begin loop:
        self._step_accepted = False
        # increase counter for how many consecutive steps the solution was
        # stable:
        self._steps_since_fail += 1
        # set instable steps counter for this step to 0:
        cnt_instable = 0
        # reset maximum von Neumann stable step:
        self._vN_max_step[0] = np.inf
        # set step counter to 0:
        i = 0
        # start loop
        while not self._step_accepted:
            # reset system truncation error accumulator to zero for this
            # timestep:
            err = 0.0
            # set step to stable for each new iteration:
            self._step_stable[0] = True
            # SOLVE STEPS:
            # solve predictor step for all numeric parts:
            for part in solve_num_expl:
                # if first try, add last step's part truncation error to part:
                if i == 0:
                    self.parts[part]._trnc_err += self.parts[
                        part
                    ].__new_trnc_err
                # get results from last timestep and pass them to
                # current timestep temperature array:
                self.parts[part].T[:] = self.parts[part].res[stepnum - 1]
                # calculate differential at result:
                self.parts[part]._df0[:] = solve_num_expl[part](_h)
                # calculate and save predictor step:
                self.parts[part].T[:] = (
                    self.parts[part].T + _h * self.parts[part]._df0
                )
                # store breaking part if not stable step
                if not self._step_stable[0]:
                    self._breaking_part = part
                    break

            # =============================================================================
            #             for part in solve_njit:
            #                 step_result(diff_fun, args, h, y_prev)
            # =============================================================================

            # do the following only if there was no von Neumann stability
            # violation in the predictor step. step_stable is set from within
            # the part-internal stability checkers.
            if self._step_stable[0]:
                # solve half step for implicit parts, if any:
                for part in solve_num_impl:
                    # get results from last timestep as flat array:
                    # CAUTION!! DO NOT USE A VIEW HERE! Otherwise the starting
                    # value of the step will be altered in each iteration!
                    T_prev = self.parts[part].res[stepnum - 1].ravel().copy()
                    # solve using scipy.optimize.root
                    self.parts[part].T[:] = _root(
                        solve_num_impl[part],
                        T_prev,
                        args=(T_prev, _h / 2, self.parts[part]._input_args),
                        method='hybr',
                    ).x.reshape(self.parts[part].T.shape)

                # solve predictor step for all non-numeric parts:
                for part in solve_nonnum:
                    self.solve_nonnum[part](_h)
                # update all ports:
                _pf._upddate_ports_interm(
                    ports_all=self.ports_all,
                    trgt_indices=self._ports_trgt_idx,
                    ports_src=self._ports_src_interm,
                    source=0,
                )
                # solve corrector step for all numeric parts:
                for part in solve_num_expl:
                    # calculate differential at predictor step result:
                    self.parts[part]._df1[:] = solve_num_expl[part](_h)
                    # solve heun method and save to result:
                    (
                        err,
                        self.parts[part].__new_trnc_err,
                    ) = _pf._heun_corrector_adapt(
                        res=self.parts[part].res,
                        T=self.parts[part].T,
                        df0=self.parts[part]._df0,
                        df1=self.parts[part]._df1,
                        trnc_err_cell_weight=self.parts[
                            part
                        ]._trnc_err_cell_weight,
                        _h=_h,
                        stepnum=stepnum,
                        rtol=rtol,
                        atol=atol,
                        err=err,
                        new_trnc_err=self.parts[part].__new_trnc_err,
                    )

                # solve next half step for implicit parts, if any:
                for part in solve_num_impl:
                    # get results from last half timestep as flat array:
                    # CAUTION!! DO NOT USE A VIEW HERE! Otherwise the
                    # starting value of the step will be altered in each
                    # iteration!
                    T_prev = self.parts[part].T.ravel().copy()
                    # solve using scipy.optimize.root
                    self.parts[part].res[stepnum] = _root(
                        solve_num_impl[part],
                        T_prev,
                        args=(T_prev, _h / 2, self.parts[part]._input_args),
                        method='hybr',
                    ).x.reshape(self.parts[part].T.shape)

                # this got moved to numba:
                #                # GET TRUCATION ERROR FOR HEUN COMPARED WITH LOWER ORDER EULER
                #                # TO CALC. NEW STEPSIZE:
                #                # truncation error approximation is the difference of the total
                #                # heun result and the euler (predictor) result saved in T. The
                #                # trunc. error is calculated by taking the root mean square
                #                # norm of the differences for each part. This applies a root
                #                # mean square error weighting over the cells.
                #                # To get the systems truncation error, the norms have to be
                #                # added up by taking the root of the sum of the squares.
                # #                # get sum of squares of part's local errors:
                # #                self.parts[part]._trnc_err = (
                # #                        ((self.parts[part].res[stepnum]
                # #                          - self.parts[part].T[:])**2).sum()
                # #                                                )
                #                # get each part's local relative error as euclidean matrix norm
                #                # (sqrt not yet taken to enable summing up the part's errors)
                #                # weighted by the relative and absolute tolerance:
                #                trnc_err = (
                #                        (((self.parts[part].res[stepnum]
                #                           - self.parts[part].T)
                #                          * self.parts[part]._trnc_err_cell_weight
                #                          / (np.maximum(self.parts[part].res[stepnum - 1],
                #                                        self.parts[part].res[stepnum])
                #                             * rtol + atol))**2).sum()
                #                            )
                #                # sum these local relative errors up for all parts:
                #                err += trnc_err
                #                # now get root mean square error for part by dividing part's
                #                # trnc_err by its amount of cells and taking the root:
                #                self.parts[part].__new_trnc_err = (
                #                        (trnc_err / self.parts[part].T.size)**0.5)
                # #                self.parts[part]._trnc_err = (
                # #                        np.sqrt(((self.parts[part].res[self.stepnum]
                # #                                  - self.parts[part].T[:])**2).sum()
                # #                                / self.parts[part].T.size)
                # #                                                )
                # #
                # #                # get system truncation error:
                # #                self._sys_trnc_err = np.sqrt(self._sys_trnc_err**2
                # #                                               + self.parts[part]._trnc_err)
                #                # now also save to T arrays to be able to easily use
                #                # memoryviews in diff functions:
                #                self.parts[part].T[:] = self.parts[part].res[self.stepnum]
                # solve corrector step for all non-numeric parts:
                for part in solve_nonnum:
                    self.solve_nonnum[part](_h)

                # update all ports:
                _pf._upddate_ports_result(
                    ports_all=self.ports_all,
                    trgt_indices=self._ports_trgt_idx,
                    ports_src=self._ports_src_result,
                    stepnum=stepnum,
                    src_list=self._ports_src_res_idx,
                )

                # numba implementation of adaptive step control. currently
                # not used due to bugs...
                # =============================================================================
                # (_h, timestep,
                #  self._step_accepted, err_rms) = self.__adaptive_step_control(
                #      _h, err, self._num_cells_tot_nmrc_expl,
                #      self._solver_state, stepnum,
                #      self._max_factor, self.__min_factor,
                #      self._steps_since_fail, self._fctr_rec_stps,
                #      self._failed_steps,
                #      self.__safety, self.__order,
                #      self.__max_stepsize, self.__min_stepsize,
                #      self.ports_all, self.__parr_bkp,
                # )
                # =============================================================================

                # ADAPTIVE TIMESTEP CALCULATION:
                # if no von Neumann stability violations (checked before)
                # if self._step_stable:
                # get all part's RMS error by dividing err by the amount of all
                # cells of explicit parts in the system and taking the root:
                err_rms = (err / self._num_cells_tot_nmrc_expl) ** 0.5
                # check for good timesteps:
                # err_rms already has the relative and absolute tolerance
                # included, thus only checking against its value:
                if 0.0 < err_rms < 1.0:  # accept and increase stepsize
                    # error is lower than tolerance, thus step is accepted.
                    self._step_accepted = True
                    # save successful timestep to simulation environment:
                    self.timestep = _h
                    # get new stepsize (err_rms is inverted thus neg. power),
                    # with min-factor set to 1 to avoid going to lower
                    # stepsizes, since this is err_rms < 1 anyways:
                    _h *= max(  # Article_Balac2013
                        1,
                        min(
                            self._max_factor[0],
                            (
                                self.__safety
                                * err_rms ** (-1 / (self.__order + 1))
                            ),
                        )
                        * (self._steps_since_fail / self._fctr_rec_stps) ** 2,
                    )
                    # check if step is not above max step:
                    if _h > self.__max_stepsize:
                        _h = self.__max_stepsize  # reduce to max stepsize
                        # save to state that max stepsize was reached:
                        self._solver_state[stepnum] = 5
                    else:
                        # else save to state that error was ok in i steps:
                        self._solver_state[stepnum] = 4
                elif err_rms == 0.0:
                    # if no RMS (most probably the step was too small so
                    # rounding error below machine precision led to cut off of
                    # digits) step will also be accepted:
                    self._step_accepted = True
                    # save successful timestep to simulation environment:
                    self.timestep = _h
                    # get maximum step increase for next step:
                    _h *= self._max_factor[0]
                    # save to state that machine epsilon was reached:
                    self._solver_state[stepnum] = 7
                    # check if step is not above max step:
                    if _h > self.__max_stepsize:
                        _h = self.__max_stepsize  # reduce to max stepsize
                else:
                    # else error was too big.
                    # check if stepsize already is at minimum stepsize. this
                    # can only be true, if stepsize has already been reduced to
                    # min. stepsize, thus to avoid infinite loop set
                    # step_accepted=True and skip the rest of the loop:
                    if _h == self.__min_stepsize:
                        self._step_accepted = True
                        # save not successful but still accepted timestep to
                        # simulation environment:
                        self.timestep = _h
                        # save this special event to solver state:
                        self._solver_state[stepnum] = 6
                    else:
                        # else if stepsize not yet at min stepsize, reduce
                        # stepsize further by error estimate if this is not
                        # less than the minimum factor and redo the step.
                        # equation like in Article_Balac2013, just without the
                        # max_factor, since it is only calculated for err > 1
                        # anyways:
                        _h *= max(
                            self.__min_factor,
                            (
                                self.__safety
                                * err_rms ** (-1 / (self.__order + 1))
                            ),
                        )
                        # check if step is not below min step:
                        if _h < self.__min_stepsize:
                            _h = self.__min_stepsize  # increase to min step
                        # reset ports array for retrying step:
                        self.ports_all[:] = self.__parr_bkp
                        # count failed steps at this step number:
                        self._failed_steps[stepnum] += 1
            else:  # catch von Neumann stability violations
                # if von Neumann stability violated, do not accept step.
                # This can happen even though the RMS-error is ok, since single
                # non-stable parts can have a too small impact on the RMS. In
                # this case _step_accepted will be overwritten.
                self._step_accepted = False  # redo the step
                # inrease counter for failed loops
                cnt_instable += 1
                # set new step to maximum von Neumann step (calc. in parts)
                # with a security factor of .9 to avoid going right back into
                # instability after this step.
                _h = self._vN_max_step[0] * 0.9
                # count failed steps at this step number:
                self._failed_steps[stepnum] += 1
                # remember last failed step to only slowly increase the step
                # size mult. factor over the next steps. (incremental counter
                # for steps since last failed step, resets here)
                self._steps_since_fail = 0
                # reset ports array for retrying step:
                self.ports_all[:] = self.__parr_bkp
                # break loop if no solution was found after 50 tries:
                if cnt_instable == 100:
                    raise ValueError(
                        'No stable solution found, stopping solver.\n'
                        'Required stepsize for stable solution: {0:4G}s.\n'
                        'Breaking part: {1}.'.format(_h, self._breaking_part)
                    )
                    # set timeframe to 0 to break the outer simulation loop
                    self.timeframe = 1e-9
                    # save error to solver state:
                    self._solver_state[stepnum] = 99
                    break
            # increase step counter:
            i += 1
        # save new timestep for the next timestep iteration:
        self._new_step = _h
        # save error to array to enable stepwise system error lookup:
        self._sys_trnc_err[stepnum] = err_rms

    @staticmethod
    # @nb.njit
    def __adaptive_step_control(
        _h,
        err,
        num_cells_tot_nmrc_expl,
        _solver_state,
        stepnum,
        _max_factor,
        __min_factor,
        _steps_since_fail,
        _fctr_rec_stps,
        _failed_steps,
        __safety,
        __order,
        __max_stepsize,
        __min_stepsize,
        ports_all,
        __parr_bkp,
    ):
        """
        Control stepsize.

        Adaptive timestep control by error estimates. Von Neumann stability
        condition violations have been checked before and are thus not included
        in this method.
        """
        # get all part's RMS error by dividing err by the amount of all
        # cells of explicit parts in the system and taking the root:
        err_rms = (err / num_cells_tot_nmrc_expl) ** 0.5
        # check for good timesteps:
        # err_rms already has the relative and absolute tolerance
        # included, thus only checking against its value:
        #                if .9 < err_rms < 1.1:
        #                    # error is at tolerance. To avoid recomputing too many
        #                    # steps, the step will be accepted in this range without
        #                    # changing the step size.
        #                    self._step_accepted = True
        #                    # save successful timestep to simulation environment:
        #                    self.timestep = _h
        if 0.0 < err_rms < 1.0:  # accept and increase stepsize
            # error is lower than tolerance, thus step is accepted.
            _step_accepted = True
            # save successful timestep to simulation environment:
            timestep = _h
            # get new stepsize (err_rms is inverted thus neg. power),
            # with min-factor set to 1 to avoid going to lower
            # stepsizes, since this is err_rms < 1 anyways:
            _h *= max(  # Article_Balac2013
                1,
                min(
                    _max_factor[0],
                    (__safety * err_rms ** (-1 / (__order + 1))),
                )
                * (_steps_since_fail / _fctr_rec_stps) ** 2,
            )
            # check if step is not above max step:
            if _h > __max_stepsize:
                _h = __max_stepsize  # reduce to max stepsize
                # save to state that max stepsize was reached:
                _solver_state[stepnum] = 5
            else:
                # else save to state that error was ok in i steps:
                _solver_state[stepnum] = 4
        elif err_rms == 0.0:
            # if no RMS (most probably the step was too small so
            # rounding error below machine precision led to cut off of
            # digits) step will also be accepted:
            _step_accepted = True
            # save successful timestep to simulation environment:
            timestep = _h
            # get maximum step increase for next step:
            _h *= _max_factor[0]
            # save to state that machine epsilon was reached:
            _solver_state[stepnum] = 7
            # check if step is not above max step:
            if _h > __max_stepsize:
                _h = __max_stepsize  # reduce to max stepsize
        else:
            # else error was too big.
            # check if stepsize already is at minimum stepsize. this
            # can only be true, if stepsize has already been reduced to
            # min. stepsize, thus to avoid infinite loop set
            # step_accepted=True and skip the rest of the loop:
            if _h == __min_stepsize:
                _step_accepted = True
                # save not successful but still accepted timestep to
                # simulation environment:
                timestep = _h
                # save this special event to solver state:
                _solver_state[stepnum] = 6
            else:
                # else if stepsize not yet at min stepsize, reduce
                # stepsize further by error estimate if this is not
                # less than the minimum factor and redo the step.
                # equation like in Article_Balac2013, just without the
                # max_factor, since it is only calculated for err > 1
                # anyways:
                _h *= max(
                    __min_factor, (__safety * err_rms ** (-1 / (__order + 1)))
                )
                # check if step is not below min step:
                if _h < __min_stepsize:
                    _h = __min_stepsize  # increase to min step
                # reset ports array for retrying step:
                ports_all[:] = __parr_bkp
                # count failed steps at this step number:
                _failed_steps[stepnum] += 1
                # do not accept step:
                _step_accepted = False
        return _h, timestep, _step_accepted, err_rms

    def _call_postponed(self):
        """
        Call postponed evaluations of add_part etc. methods.

        This method calls the postponed functions stored in the `__postponed`
        dict. The function arguments were passed to each function in the
        function construction process, for example in `add_part()`.
        """
        try:  # add and init parts, if any
            for add_part in self.__postpone['add_part'].values():
                add_part()
            #            for init_part in self.__postpone['init_part'].values():
            #                init_part()  # old method using lambda
            for init_part, kwargs in self.__postpone['init_part'].items():
                try:
                    self.parts[init_part].init_part(**kwargs)
                except TypeError as e:  # if any args not given
                    raise TypeError(
                        'While adding part `{0}` to the simulation '
                        'environment, the following error '
                        'occurred: '.format(init_part) + e.args[0]
                    )
        except KeyError as ke:
            if str(ke) == 'add_part':  # this avoids catchin key errors from
                raise AssertionError(
                    (  # routines deeper in the program
                        'No parts have been added to the simulation environment! '
                        'Add at least one part with `{0}.add_part()` before '
                        'initializing the simulation.'
                    ).format(self._simenv_name)
                )
            else:  # else raise the error from the deep routine
                raise

        if 'add_open_port' in self.__postpone:  # add open ports, if any
            for add_open_port in self.__postpone['add_open_port'].values():
                add_open_port()

        if 'dyn_BC' in self.__postpone:  # add dynamic boundary cond., if any
            for dyn_BC in self.__postpone['dyn_BC'].values():
                dyn_BC()
            self._initialize_dynamic_BC()  # initialize dyn BC

        if 'connect_ports' in self.__postpone:  # connect ports, if any
            for connect_ports in self.__postpone['connect_ports'].values():
                connect_ports()

        # check for unconnected ports
        self._check_unconnected_ports()

        if 'add_ctrl' in self.__postpone:  # add and init controls, if any
            for add_ctrl in self.__postpone['add_ctrl'].values():
                add_ctrl()
            #            for init_ctrl in self.__postpone['init_ctrl'].values():
            #                init_ctrl()  # old method using lambda
            for init_ctrl, kwargs in self.__postpone['init_ctrl'].items():
                self.ctrls[init_ctrl].init_controller(**kwargs)

    def _create_port_updater(self):
        # preallocate lists
        interm_result_1_list = [None] * self.num_ports  # intermediate result
        result_list = [None] * self.num_ports  # final step result
        # get indices
        result_idx_list = [-9999] * self.num_ports  # src index to results
        trgt_idx_list = [-9999] * self.num_ports  # target index to ports_all
        for part in self.parts.values():
            for i in range(part.port_ids.size):
                trgt_idx = part.port_ids[i]  # get target index to ports_all
                src_idx = part._port_own_idx[i]  # get flat src index
                # now make this flat source index into a tuple of n elements
                # for n-dimensional T arrays, where the first element is a
                # slice. This allows to get a view to a cell of T, regardless
                # if the array is contigouos or not.
                # unravel flat index into n-dim index:
                src_idx_inter = np.unravel_index(src_idx, part.T.shape)
                # slice first dim and unpack the rest to len n tuple index:
                src_slc_inter = (
                    slice(src_idx_inter[0], src_idx_inter[0] + 1),
                    *src_idx_inter[1:],
                )
                # get a view of T regardless of shape and c-contiguity:
                interm_result_1_list[trgt_idx] = part.T[src_slc_inter]
                # save target index into ports all array:
                trgt_idx_list[trgt_idx] = trgt_idx
                # results array is always c-conti, thus just reshape and save
                # view:
                result_list[trgt_idx] = part.res.reshape(-1)
                # get size of res array except for the first dimension (equal
                # size of T) and each port's index:
                result_idx_list[trgt_idx] = (part.T.size, src_idx)
        # cut out None values:
        interm_result_1_list = [
            x for x in interm_result_1_list if x is not None
        ]
        result_list = [x for x in result_list if x is not None]
        result_idx_list = [x for x in result_idx_list if x != -9999]
        trgt_idx_list = [x for x in trgt_idx_list if x != -9999]
        # make them to tuples where arrays are stored, else just store the list
        self._ports_src_interm = (tuple(interm_result_1_list),)
        self._ports_src_result = tuple(result_list)
        self._ports_trgt_idx = tuple(trgt_idx_list)
        self._ports_src_res_idx = tuple(result_idx_list)

    def _collapse_port_arrays(self):
        """
        Not needed anymore, since all relevant ports are now collapsed from the
        start!
        """
        for part in self.parts.values():
            if part.collapse_arrays and not part._collapsed:
                part._T_port = part._T_port.flat[part._port_own_idx_2D]
                #                part._UA_port_wll = part._UA_port_wll.flat[
                #                        part._port_own_idx_2D]
                part._UA_port_fld = part._UA_port_fld.flat[
                    part._port_own_idx_2D
                ]
                part._UA_port = part._UA_port.flat[part._port_own_idx_2D]
                part._port_gsp = part._port_gsp.flat[part._port_own_idx_2D]
                #                part._A_p_fld_mean = part._A_p_fld_mean.flat[
                #                        part._port_own_idx_2D]
                part._lam_port_fld = part._lam_port_fld.flat[
                    part._port_own_idx_2D
                ]
                part._lam_fld_own_p = part._lam_fld_own_p.flat[
                    part._port_own_idx_2D
                ]
                #                part._port_subs_gsp = part._port_subs_gsp.flat[
                #                        part._port_own_idx_2D]
                part._cp_port = part._cp_port.flat[part._port_own_idx_2D]

                # NEWNEWNEW:
                part._A_wll_conn_p = part._A_wll_conn_p.flat[
                    part._port_own_idx_2D
                ]
                part._A_fld_conn_p = part._A_fld_conn_p.flat[
                    part._port_own_idx_2D
                ]
                part._lam_wll_conn_p = part._lam_wll_conn_p.flat[
                    part._port_own_idx_2D
                ]

                # EVEN NEWER:
                #                if part.constr_type == 'Pipe':
                part._dm_port = part._dm_port.flat[part._port_own_idx_2D]

                part._collapsed = True  # set bool checker to True if collapsed
                print(part.name, 'collapsed!')

    def _create_dataclasses(self):
        """
        This method creates data classes in which all data needed for the
        calculations of the differentials is stored. This enables passing only
        a single data class as argument to an outsourced and jitted function
        instead of a multitude (for example 58 for a pipe) of arguments,
        increasing the performance of calling jitted functions.

        """

        for part in self.parts.values():
            if hasattr(part, '_calc_att_list'):
                # pass the instance attribute dict to the data class creation
                # function. This function only selects specific data types as
                # inputs:
                # data named tuple
                part.__dtdict = {}
                i = 0
                for lmnt in part.new_list:
                    part.__dtdict[lmnt] = getattr(part, part._calc_att_list[i])
                    i += 1
                part.__dtnt = _namedtuple('data_nt', part.new_list)
                part._data_nt = part.__dtnt(**part.__dtdict)

    def _build_simenv(self):
        """
        This method builds the simenv by adding all parts, controls, boundary
        conditions and connecting them as well as building the topology of
        the grid.
        Afterwards checks are run to make sure that everything is correctly
        initialized.
        This function must not be run twice during setting up a simulation
        environment, otherwise errors will be raised.
        """

        # construct the model environment by calling all postponed part, port
        # or control adding and initialization functions and connecting ports
        # in the correct order:
        self._call_postponed()

        # check if all parts have been initialized:
        for name, part in self.parts.items():
            if not part.initialized:
                err_str = (
                    'Part `' + name + '` not initialized! Initialize the part '
                    'by calling its `{0}.parts[\'' + name + '\'].init_part()` '
                    'method.'
                ).format(self._simenv_name)
                raise UserWarning(err_str)
        # check if all parts which can be actuators have their controls
        # specified or are set to static:
        for part in self.parts:
            if self.parts[part].control_req:
                if not self.parts[part].ctrl_defined:
                    raise UserWarning(
                        self.parts[part].constr_type
                        + ' `'
                        + part
                        + '` needs to be controlled or defined as '
                        '`ctrl_required=False`.'
                    )
        # check if all controls are initialized:
        for controller in self.ctrl_l:
            if not controller.initialized:
                raise UserWarning(
                    (
                        'Controller `{0}` is not initialized! '
                        'Initialize the controller by calling its `{1}.ctrls[\''
                        '{0}\'].init_controller()` method.'
                    ).format(controller.name, self._simenv_name)
                )

        # check if solver was set:
        err_slvr = (
            'The solver method was not set! Set the solver method '
            'with `{0}.set_solver()`, before initializing the '
            'simulation.'
        ).format(self._simenv_name)
        assert self._solver_set, err_slvr

        # check if timeframe was set:
        err_tf = (
            'The timeframe for the simulation environment has to be set '
            'before initializing the simulation environment by calling the '
            'method `{0}.set_timeframe()`.'
        ).format(self._simenv_name)
        assert self._timeframe_set, err_tf

        # check if save to disk was set:
        err_ds = (
            'Disk saving of results has to be set before initializing the '
            'simulation environment by calling the method '
            '`{0}.set_disksaving()`.'
        ).format(self._simenv_name)
        assert self._disksaving_set, err_ds

        # calculate flow net to get routine for updating massflows:
        self._get_topology()

    def _initialize_arrays(self, *, max_steps_b4_save_to_disk=1e4):
        """Initialize all arrays."""
        # update flow net for the first time:
        self._update_FlowNet()

        # construct port indexing arrays and get constant values of connected
        # ports:
        self._get_port_connections()
        # calculate U*A values for wall-wall-port connections:
        self._calc_UAwll_port()

        # ARRAY PREALLOCATION:
        # get maximum steps before saving to disk:
        self._msb4save = int(max_steps_b4_save_to_disk)
        # get total RAM to be able to estimate maximum number of allowed array
        # cells:
        self._ram = _psutil.virtual_memory().total
        # maximum cell allocation of half the ram for float64 arrays (thus /8):
        self._num_max_cells = self._ram * 0.5 / 8
        # get number of maximum timesteps when all arrays should be saved in
        # RAM:
        self._max_steps = self._num_max_cells / self._num_cells_tot_nmrc
        # preallocate result arrays with 0K temperature for the full amount
        # of timesteps (for dynamic timestepping an estimate has to be made at
        # perhaps around 5*num_steps) for numeric parts.
        if self.adaptive_steps:
            self.num_steps = int(100 * self.num_steps)
            # also save original amount of steps to use that again when
            # enlarging arrays:
            self.__base_num_steps = self.num_steps + 1
            # give all parts a new truncation error variable if adaptive steps:
            for part in self.parts:
                if self.parts[part].solve_numeric:  # only for numeric parts!
                    self.parts[part].__new_trnc_err = 0.0
        # if number of steps is above msb4sav or greater than maximum number of
        # steps, limit to these numbers. saving to disk is already enabled
        # during sim init...
        if self.num_steps > self._msb4save or self.num_steps > self._max_steps:
            # limit consecutive number of steps:
            self.num_steps = (
                self._msb4save
                if self._max_steps > self._msb4save
                else self._max_steps
            )

        # save views/references of all simulation environment wide arrays and
        # lists which are needed in calculations to parts:
        for part in self.parts.values():
            part.ports_all = self.ports_all
            part._vN_max_step = self._vN_max_step
            part._max_factor = self._max_factor
            part._step_stable = self._step_stable
            part.stepnum = self.stepnum

        # initial conditions are set for the first row of the res array:
        # print('init sim: num_steps', self.num_steps)
        for part in self.parts:
            # if a part which has to be calculated numerically:
            if self.parts[part].solve_numeric:
                # extend value result array and set starting value
                self.parts[part].res = np.zeros(
                    (self.num_steps + 1,) + self.parts[part].T.shape
                )
                self.parts[part].res[0] = self.parts[part]._T_init
                # extend massflow result array and set starting value
                self.parts[part].res_dm = np.zeros(
                    (self.num_steps + 1, self.parts[part].dm.shape[0])
                )
                self.parts[part].res_dm[0] = self.parts[part].dm
                # update ports with the init values (_update_ports method
                # using dicts needed here, since result arrays not fully
                # initialized here and thus fast method not yet initialized):
                self.parts[part]._update_ports(source='results')
                # if special initialization methods exist:
                if hasattr(self.parts[part], 'res_dQ'):
                    self.parts[part].res_dQ = np.zeros(
                        (self.num_steps + 1,)
                        + self.parts[part]._dQ_heating.shape
                    )
                if hasattr(self.parts[part], '_special_array_init'):
                    self.parts[part]._special_array_init(self.num_steps)

        # if part is a part which has NOT to be calculated numerically:
        for part in self.solve_nonnum:
            # preallocate all arrays and call one init. solver step
            self.parts[part].res = np.zeros(
                (self.num_steps + 1,) + self.parts[part].T.shape
            )
            self.parts[part].res[0] = self.parts[part]._T_init
            self.parts[part].res_dm = np.zeros(
                (self.num_steps + 1, self.parts[part].dm.shape[0])
            )
            self.parts[part].solve(self.timestep)
            self.parts[part]._update_ports(source='intermediate', nnum=True)

        # initialize time vector
        # print(self.num_steps)
        self.time_vec = np.zeros(self.num_steps + 1)
        self.time_vec[0] = 0.0
        # initialize system truncation error:
        self._sys_trnc_err = np.zeros_like(self.time_vec)
        self._failed_steps = np.zeros_like(self.time_vec)
        # initialize solver state vector:
        self._solver_state = np.zeros_like(self.time_vec)
        # and their total representatives to enable saving to disk (empty since
        # these are filled while saving to disk):
        self._total_sys_trnc_err = np.zeros((0,))
        self._total_failed_steps = np.zeros((0,))
        self._total_solver_state = np.zeros((0,))
        # initialize backup for ports array to enable redoing steps:
        self.__parr_bkp = np.zeros_like(self.ports_all)
        # reset some other values to allow restarting the sim:
        self.stepnum[0] = 0
        self._vN_max_step[0] = np.inf
        self._SimEnv__last_step_t_idx = 0

        # create inputs for numba njit port updater:
        self._create_port_updater()

    def _create_diff_inputs(self):
        """
        Create differential inputs.

        This method creates the lists containing each part's differential
        (and non numeric solve) method inputs. Using these lists as inputs
        avoids passing a multitude of separate arguments to the methods and
        thus speeds up function calls as well as enabling jit-compilation
        of the full methods.
        If all part's support a fully jitted differential or non numeric solve
        method, this will also result in the solver loops being jitted and
        parallelized, achieving major performance boosts.
        """
        # loop over all parts
        for name, part in self.parts.items():
            part._input_args = []  # create empty input args list
            # loop over list of input argument names. THIS IS STRICTLY SORTED
            # IN THE ORDER OF THE INPUT ARGS OF EACH PART'S METHOD!!!
            # create argument containers:
            for arg in part._input_arg_names_sorted:
                part._input_args.append(part.__dict__[arg])
            # make to tuple, since numba can only expand tuple args with *:
            part._input_args = tuple(part._input_args)
            # preallocate differential grids:
            part._df0 = np.zeros_like(part.T)
            part._df1 = np.zeros_like(part.T)

        # loop over fully njit parts to store their inputs in a tuple in the
        # correct order
        self._solve_njit_args = []
        for part in self.solve_num_njit.keys():
            self._solve_njit_args.append(self.parts[part]._input_args)
        self._solve_njit_args = tuple(self._solve_njit_args)

    def _prep_inputs_for_njits(self):
        self._njit_interm_res = tuple(
            [self.parts[k].T.reshape(-1) for k in self.solve_num_njit.keys()]
        )
        self._njit_final_res = tuple(
            [self.parts[k].res.reshape(-1) for k in self.solve_num_njit.keys()]
        )

    def _reset_simenv(self):
        # reset timeframe:
        self.set_timeframe(
            timeframe=self.timeframe,
            adaptive_steps=self.adaptive_steps,
            **self.__bkp_kwargs_timeframe
        )

        # reset all temperature and massflow arrays inside the parts:
        for part in self.parts.values():
            part._reset_to_init_cond()

        # reset all controllers:
        for ctrl in self.ctrls.values():
            ctrl._reset_to_init_cond()

        # reset dynamic boundary conditions:
        self._initialize_dynamic_BC()

        self._new_step = 1.0  # this is the adaptive step starting step
        self.timestep = 1.0  # this is the adaptive step final value

    def initialize_sim(self, *, build_simenv=True, reset_simenv=False):

        if build_simenv:
            self._build_simenv()

        if reset_simenv:
            self._reset_simenv()

        # Initialiaze simulation arrays
        self._initialize_arrays(
            max_steps_b4_save_to_disk=self._save_every_n_steps
        )

        # Construct inputs to differential functions
        self._create_diff_inputs()

        # prepare inputs for numba jitted functions. NOT NEEDED CURRENTLY
        self._prep_inputs_for_njits()

        # prepare numba jitclass. DEPRECATED!
        # self._construct_jitclass()

        # Set simulation environment to initialized:
        self._initialized = True

    def _check_memory_address(self):
        """
        Assert that memory addresses of important arrays did not change.

        Due to extensive use of memory-views, checking for consistency of
        memory addresses of arrays makes sure that the memory views were not
        broken in the process of the simulation.

        """

        # loop over parts
        for part in self.parts:
            # check for changes in memory address of T-arrays:
            if (
                self.parts[part].T.__array_interface__['data'][0]
                != self.parts[part]._memadd_T
            ):
                print(part, 'memory address of T-array changed!')
            # if available, check for changes in memory address of massflow
            # arrays:
            if hasattr(self.parts[part], '_memadd_dm'):
                if (
                    self.parts[part].dm.__array_interface__['data'][0]
                    != self.parts[part]._memadd_dm
                ):
                    print(
                        part, 'memory address of massflow-array (dm) changed!'
                    )

    def _finalize_sim(self):
        # Check for consistency of important arrays:
        self._check_memory_address()

        # save remaining results to disk
        if self.__save_to_disk:  # if disk saving was used so far...
            self._free_memory(array_length=self.stepnum[0])

        # add total stepnum up to current stepnum:
        self.stepnum[0] += self._total_stepnum

        # open store without context manager for remaining storage operations:
        self._disk_store['store_tmp'] = pd.HDFStore(
            self._disk_store['path_tmp'], mode='r+'
        )

        # save to disk, if activated (default)
        if self.__save_to_disk:
            # res and res_dm backcalculation
            for part in self.parts:
                self.parts[part].res_dm = np.append(
                    self._disk_store['store_tmp'][part + '/dm'].values,
                    self.parts[part].res_dm,
                    axis=0,
                )
                # if 3D res array, reshaping has to be done before
                if self.parts[part].res.ndim == 2:
                    self.parts[part].res = np.append(
                        self._disk_store['store_tmp'][part + '/res'].values,
                        self.parts[part].res,
                        axis=0,
                    )
                elif self.parts[part].res.ndim == 3:  # reshape back to 3D if
                    self.parts[part].res = np.append(  # it was 3D before
                        self._disk_store['store_tmp'][
                            part + '/res'
                        ].values.reshape(-1, *self.parts[part].res.shape[1:]),
                        self.parts[part].res,
                        axis=0,
                    )
                # if the part has a special method:
                if hasattr(self.parts[part], '_final_datastore_backcalc'):
                    self.parts[part]._final_datastore_backcalc(
                        disk_store=self._disk_store, part=part
                    )

            # compress store if set:
            self._compress_store()
            # time vectors etc full data
            self.time_vec = np.append(
                self._total_time_vec, self.time_vec, axis=0
            )
            self._sys_trnc_err = np.append(
                self._total_sys_trnc_err, self._sys_trnc_err, axis=0
            )
            self._failed_steps = np.append(
                self._total_failed_steps, self._failed_steps, axis=0
            )
            self._solver_state = np.append(
                self._total_solver_state, self._solver_state, axis=0
            )

        # cut all arrays and values to the actually needed array lengths:
        self._crop_results()

        # check for valid results if a part has a checking method:
        for p in self.parts.values():
            if hasattr(p, '_check_results'):
                p._check_results()

        # calculate timestep vector for plotting and use last value twice:
        self.time_step_vec = np.append(
            np.diff(self.time_vec),
            self.time_vec[-1:] - self.time_vec[-2:-1],
            axis=0,
        )

        # and again save values to store, this time the utility vectors:
        if self.__save_to_disk:
            self._disk_store['store'].put(
                'sim_vecs',
                pd.DataFrame(
                    index=self._disk_store_timevec,
                    data={
                        'time': self.time_vec,
                        'timestep': self.time_step_vec,
                        'sys_trnc_err': self._sys_trnc_err,
                        'failed_steps': self._failed_steps,
                        'solver_state': self._solver_state,
                    },
                ),
                format='table',
                complevel=self._disk_store['complevel'],
                complib=self._disk_store['complib'],
            )
            if len(self._disk_store_utility) > 0:
                for k, v in self._disk_store_utility.items():
                    v(
                        store=self._disk_store['store'],
                        store_kdws=dict(
                            format='table',
                            complevel=self._disk_store['complevel'],
                            complib=self._disk_store['complib'],
                        ),
                    )

            # finally close store:
            self._disk_store['store'].close()

        # count solver states in the first column:
        states = dict(zip(*np.unique(self._solver_state, return_counts=True)))
        # append 0 for states that did not occur:
        states[4] = 0 if 4 not in states else states[4]
        states[5] = 0 if 5 not in states else states[5]
        states[6] = 0 if 6 not in states else states[6]
        states[7] = 0 if 7 not in states else states[7]
        # print solver state message:
        states_list = [
            '4: Normal steps',
            '5: Step limited to maximum stepsize',
            '6: Step limited to minimum stepsize',
            '7: Step truncation error 0 (machine epsilon reached)',
        ]
        if not self.suppress_printing:
            print(
                '\n\n'
                'The simulation was solved in '
                + str(self.stepnum[0])
                + ' steps.'
                '\n'
                'The following solver states occured during solving:\n'
                '    Normal steps (including repeated): '
                + str(states[4])
                + '\n'
                '    Steps limited to maximum stepsize: '
                + str(states[5])
                + '\n'
                '    Steps limited to minimum stepsize: '
                + str(states[6])
                + '\n'
                '    Roundoff error governed step solution (truncation error is '
                '0, respectively at machine epsilon): ' + str(states[7]) + '\n'
                '    Number of steps which had to be repeated: '
                + str(self._failed_steps.sum())
                + '\n'
                'Detailed stepwise solver states can be found in '
                '`self._failed_steps` and `self._solver_state`, with the '
                'following states:\n' + str(states_list)
            )

        # measure time taken for postprocessing
        self._sim_postproc_time = _time_.time()

        self._write_report()

    def start_sim(self):
        """Run the simulation."""

        if not self._initialized:
            raise UserWarning(
                (
                    'Simulation environment is not correctly initialized! '
                    'Initialize the simulation environment by calling the method '
                    '`{0}.initialize_sim()` before starting the simulation.'
                ).format(self._simenv_name)
            )

        # assert that start date is set correctly if disksaving:
        if self._disksaving_set and self._disk_store['start_date'] is None:
            raise TypeError(
                'Diskaving is enabled with `start_date=\'infer\'`, but no '
                'dynamic boundary condition with a DatetimeIndex to infer '
                'the start date from was given. Either set the start date '
                'explicitly or pass a dynamic boundary condition with a '
                'start date.'
            )

        # get system time:
        self._sim_start_time = _time_.time()

        # define current simulation time:
        self.time_sim = 0.0

        # set initial stepnum
        self.stepnum[0] = 1

        while self.time_sim < self.timeframe:
            # update all massflows:
            self._update_FlowNet()

            self.solver(
                self.solve_num_expl,
                self.solve_num_impl,
                self.solve_nonnum,
                self.timestep,
            )
            # run control algorithms:
            self._update_control(self.timestep)

            # update all dynamic BC:
            self._update_dyn_BC(self.time_sim, self.timestep)

            self.time_sim += self.timestep
            # add elapsed time in seconds to time vector:
            self.time_vec[self.stepnum[0]] = self.time_sim

            # check if stepnum is greater than num_steps. And if yes call
            # function to save current results to DataFrame, save this to disk
            # and clear current result arrays:
            if self.stepnum[0] == self.num_steps:
                # check if not already total sim timeframe reached:
                if self.time_sim < self.timeframe:
                    self._free_memory()

            self.stepnum[0] += 1

            # show progress in 5% steps (thus counter /5):
            Counter = self.time_sim / self.timeframe * 100 / 5
            if (Counter - int(Counter)) < 0.01:
                sys.stdout.write('\r')
                sys.stdout.write(
                    'Simulation progress: [%-20s] %d%%'
                    % ('=' * (int(Counter / 5 * 5)), (int(Counter * 5)))
                )
                sys.stdout.flush()

            # check for nan values every 5k steps:
            if self._total_stepnum % 5000 == 0:
                for name, prt in self.parts.items():
                    # loop over all parts and assert that none of the arrays
                    # contains any nan
                    isanynan = np.any(np.isnan(prt.T)) and np.any(
                        np.isnan(prt.dm)
                    )
                    if isanynan:
                        print(
                            '\n\n'
                            'WARNING: Invalid value found between step {0} '
                            'at time {1}s and step {2} at time {3}s!\n'
                            'Stopping simulation.\n\n'.format(
                                self._total_stepnum - 5000,
                                self.time_vec[self._total_stepnum - 5000],
                                self._total_stepnum,
                                self.time_sim,
                            )
                        )
                        # stop simulation by setting timeframe to almost zero
                        self.timeframe = 1e-9
                        # set solver state to failed:
                        self._solver_state[self._total_stepnum] = 98
                        break  # break loop for nan checking

        # measure time taken purely for simulation
        self._sim_end_time = _time_.time()

        # finalize sim (checks, postprocessing, storing data)
        self._finalize_sim()

    def __deprecated_free_memory(self, array_length=-1):
        """
        Free the arrays in memory by saving results to disk.

        If disksaving is set, arrays are saved to disk and cleared. Otherwise
        array size will be increased at the cost of a higher memory
        consumption.

        """
        if not self.__save_to_disk:
            # timevector etc. enlarging:
            arr_to_append = np.zeros(self.__base_num_steps)
            self.time_vec = np.append(self.time_vec, arr_to_append, axis=0)
            self._sys_trnc_err = np.append(
                self._sys_trnc_err, arr_to_append, axis=0
            )
            self._failed_steps = np.append(
                self._failed_steps, arr_to_append, axis=0
            )
            self._solver_state = np.append(
                self._solver_state, arr_to_append, axis=0
            )
            # loop over parts:
            for part in self.parts:
                # for all parts:
                self.parts[part].res_dm = np.append(
                    self.parts[part].res_dm,
                    np.zeros(
                        (
                            self.__base_num_steps,
                            self.parts[part].res_dm.shape[1],
                        )
                    ),
                    axis=0,
                )
                self.parts[part].res = np.append(
                    self.parts[part].res,
                    np.zeros(
                        (self.__base_num_steps, self.parts[part].res.shape[1])
                    ),
                    axis=0,
                )

            # set numer of needed steps to new amount:
            self.num_steps = self.time_vec.size - 1
        else:  # if saving to disk as HDFstore
            # index for hdf store
            hdf_idx = self._disk_store[  # create time index for hdf store
                'curr_step_start_date'
            ] + pd.to_timedelta(self.time_vec[:array_length], unit='s')
            # # next start date ist the current end date: NOPE NOPE! timevew includes this!
            # # self._disk_store['curr_step_start_date'] = hdf_idx[-1]
            # print('new start', self._disk_store['curr_step_start_date'])
            # store current timevector etc. to total timevector (which is
            # appended to the total size) and reset current arrays:
            self._total_time_vec = np.append(
                self._total_time_vec, self.time_vec[:array_length], axis=0
            )
            self._total_sys_trnc_err = np.append(
                self._total_sys_trnc_err,
                self._sys_trnc_err[:array_length],
                axis=0,
            )
            self._total_failed_steps = np.append(
                self._total_failed_steps,
                self._failed_steps[:array_length],
                axis=0,
            )
            self._total_solver_state = np.append(
                self._total_solver_state,
                self._solver_state[:array_length],
                axis=0,
            )
            # save last state and reset the rest
            self.time_vec[0] = self.time_vec[self.stepnum[0]]
            self._sys_trnc_err[0] = self._sys_trnc_err[self.stepnum[0]]
            self._failed_steps[0] = self._failed_steps[self.stepnum[0]]
            self._solver_state[0] = self._solver_state[self.stepnum[0]]
            self.time_vec[1:] = 0
            self._sys_trnc_err[1:] = 0
            self._failed_steps[1:] = 0
            self._solver_state[1:] = 0
            for part in self.parts:  # loop over parts to save results
                # save results to disk:
                self._disk_store['store_tmp'].append(
                    part + '/dm',
                    pd.DataFrame(
                        data=self.parts[part].res_dm[:array_length, ...],
                        index=hdf_idx,
                    ),
                )
                # reshape data to 2D array if it was 3D, else keep 2D
                if self.parts[part].res[:array_length, ...].ndim == 2:
                    self._disk_store['store_tmp'].append(
                        part + '/res',
                        pd.DataFrame(
                            data=self.parts[part].res[:array_length, ...],
                            index=hdf_idx,
                        ),
                    )
                elif self.parts[part].res[:array_length, ...].ndim == 3:
                    # reshape by number of values per first dimension to
                    # reshape to 2D array and if available, extract column
                    # names from part
                    coln = getattr(self.parts[part], '_res_val_names', None)
                    self._disk_store['store_tmp'].append(
                        part + '/res',
                        pd.DataFrame(
                            data=self.parts[part]
                            .res[:array_length, ...]
                            .reshape(
                                -1,
                                np.prod(  # get no. of vals per dim 1
                                    self.parts[part].res.shape[1:]
                                ),
                            ),
                            index=hdf_idx,
                            columns=coln,
                        ),
                    )
                # set current result to row 0 of array and clear the rest:
                self.parts[part].res_dm[0, ...] = self.parts[part].res_dm[
                    self.stepnum[0], ...
                ]
                self.parts[part].res_dm[1:, ...] = 0.0
                self.parts[part].res[0, ...] = self.parts[part].res[
                    self.stepnum[0], ...
                ]
                self.parts[part].res[1:, ...] = 0.0
                # for parts with special arrays, use the special method:
                if hasattr(self.parts[part], '_special_free_memory'):
                    self.parts[part]._special_free_memory(
                        disk_store=self._disk_store,
                        part=part,
                        array_length=array_length,
                        hdf_idx=hdf_idx,
                        stepnum=self.stepnum,
                    )
            # reset stepnum back to 0 (+1 will be set right after
            # free_memory method) and bank total stepnum:
            self._total_stepnum += self.stepnum[0]
            self.stepnum[0] = 0

    def _free_memory(self, array_length=-1):
        """
        Free the arrays in memory by saving results to disk.

        If disksaving is set, arrays are saved to disk and cleared. Otherwise
        array size will be increased at the cost of a higher memory
        consumption.

        """
        if not self.__save_to_disk:
            # timevector etc. enlarging:
            arr_to_append = np.zeros(self.__base_num_steps)
            self.time_vec = np.append(self.time_vec, arr_to_append, axis=0)
            self._sys_trnc_err = np.append(
                self._sys_trnc_err, arr_to_append, axis=0
            )
            self._failed_steps = np.append(
                self._failed_steps, arr_to_append, axis=0
            )
            self._solver_state = np.append(
                self._solver_state, arr_to_append, axis=0
            )
            # loop over parts:
            for part in self.parts:
                # for all parts:
                self.parts[part].res_dm = np.append(
                    self.parts[part].res_dm,
                    np.zeros(
                        (
                            self.__base_num_steps,
                            self.parts[part].res_dm.shape[1],
                        )
                    ),
                    axis=0,
                )
                self.parts[part].res = np.append(
                    self.parts[part].res,
                    np.zeros(
                        (self.__base_num_steps, self.parts[part].res.shape[1])
                    ),
                    axis=0,
                )

            # set numer of needed steps to new amount:
            self.num_steps = self.time_vec.size - 1
        else:  # if saving to disk as HDFstore
            with pd.HDFStore(self._disk_store['path_tmp'], mode='r+') as store:
                # index for hdf store
                hdf_idx = self._disk_store[  # create time index for hdf store
                    'curr_step_start_date'
                ] + pd.to_timedelta(self.time_vec[:array_length], unit='s')
                # # next start date ist the current end date: NOPE NOPE! timevew includes this!
                # # self._disk_store['curr_step_start_date'] = hdf_idx[-1]
                # print('new start', self._disk_store['curr_step_start_date'])
                # store current timevector etc. to total timevector (which is
                # appended to the total size) and reset current arrays:
                self._total_time_vec = np.append(
                    self._total_time_vec, self.time_vec[:array_length], axis=0
                )
                self._total_sys_trnc_err = np.append(
                    self._total_sys_trnc_err,
                    self._sys_trnc_err[:array_length],
                    axis=0,
                )
                self._total_failed_steps = np.append(
                    self._total_failed_steps,
                    self._failed_steps[:array_length],
                    axis=0,
                )
                self._total_solver_state = np.append(
                    self._total_solver_state,
                    self._solver_state[:array_length],
                    axis=0,
                )
                # save last state and reset the rest
                self.time_vec[0] = self.time_vec[self.stepnum[0]]
                self._sys_trnc_err[0] = self._sys_trnc_err[self.stepnum[0]]
                self._failed_steps[0] = self._failed_steps[self.stepnum[0]]
                self._solver_state[0] = self._solver_state[self.stepnum[0]]
                self.time_vec[1:] = 0
                self._sys_trnc_err[1:] = 0
                self._failed_steps[1:] = 0
                self._solver_state[1:] = 0
                for part in self.parts:  # loop over parts to save results
                    # save results to disk:
                    store.append(
                        part + '/dm',
                        pd.DataFrame(
                            data=self.parts[part].res_dm[:array_length, ...],
                            index=hdf_idx,
                        ),
                    )
                    # reshape data to 2D array if it was 3D, else keep 2D
                    if self.parts[part].res[:array_length, ...].ndim == 2:
                        store.append(
                            part + '/res',
                            pd.DataFrame(
                                data=self.parts[part].res[:array_length, ...],
                                index=hdf_idx,
                            ),
                        )
                    elif self.parts[part].res[:array_length, ...].ndim == 3:
                        # reshape by number of values per first dimension to
                        # reshape to 2D array and if available, extract column
                        # names from part
                        coln = getattr(
                            self.parts[part], '_res_val_names', None
                        )
                        store.append(
                            part + '/res',
                            pd.DataFrame(
                                data=self.parts[part]
                                .res[:array_length, ...]
                                .reshape(
                                    -1,
                                    np.prod(  # get no. of vals per dim 1
                                        self.parts[part].res.shape[1:]
                                    ),
                                ),
                                index=hdf_idx,
                                columns=coln,
                            ),
                        )
                    # set current result to row 0 of array and clear the rest:
                    self.parts[part].res_dm[0, ...] = self.parts[part].res_dm[
                        self.stepnum[0], ...
                    ]
                    self.parts[part].res_dm[1:, ...] = 0.0
                    self.parts[part].res[0, ...] = self.parts[part].res[
                        self.stepnum[0], ...
                    ]
                    self.parts[part].res[1:, ...] = 0.0
                    # for parts with special arrays, use the special method:
                    if hasattr(self.parts[part], '_special_free_memory'):
                        self.parts[part]._special_free_memory(
                            disk_store=store,
                            part=part,
                            array_length=array_length,
                            hdf_idx=hdf_idx,
                            stepnum=self.stepnum,
                        )
                # reset stepnum back to 0 (+1 will be set right after
                # free_memory method) and bank total stepnum:
                self._total_stepnum += self.stepnum[0]
                self.stepnum[0] = 0

    def _compress_store(self):
        """
        Compress disk store.

        Disk store is compressed by dropping values which are not set to be
        saved and applying compression algorithms.

        """
        # backup store time vector for later use:
        self._disk_store_timevec = self._disk_store['store_tmp'][
            list(self.parts.keys())[0] + '/res'
        ].index.copy()
        # compress store if set
        if self._disk_store['compress']:
            # create new store with compression set
            self._disk_store['store'] = pd.HDFStore(
                self._disk_store['path'],
                mode='w',
                complevel=self._disk_store['complevel'],
                complib=self._disk_store['complib'],
            )
            # loop over all keys in temp store and append compressed
            for key in self._disk_store['store_tmp'].keys():
                # get table:
                dst_tab = self._disk_store['store_tmp'][key]
                # get part name of key:
                pn = key.split('/')[1]
                # get disk store value of the part:
                dsp = self._disk_store_parts[pn]
                # if bool and False, skip saving this part. if bool and True,
                # convert to a full slice for saving all cells:
                if isinstance(dsp, bool) and not dsp:
                    continue  # skip part, no saving
                elif isinstance(dsp, bool) and dsp:
                    dsp = np.s_[:, :]  # index all cells
                elif dst_tab.shape[1] == 1:
                    # if the current table has only one col (for example
                    # massflow cols), save this col
                    dsp = np.s_[:, :]
                elif isinstance(dsp, (int, list, tuple)):
                    # slice time dimension and select cols
                    dsp = (
                        np.s_[:, list((dsp,))]
                        if isinstance(dsp, int)
                        else np.s_[:, list(dsp)]
                    )
                elif isinstance(dsp, slice):
                    dsp = np.s_[:, dsp]
                # resample or not?
                if not self._disk_store['resample']:  # no resampling
                    self._disk_store['store'].put(
                        key,
                        dst_tab.iloc[dsp],
                        format='table',
                        complevel=self._disk_store['complevel'],
                        complib=self._disk_store['complib'],
                    )
                else:  # resample
                    self._disk_store['store'].put(
                        key,
                        dst_tab.iloc[dsp]
                        .resample(self._disk_store['resample_freq'])
                        .mean()
                        .interpolate(),
                        format='table',
                        complevel=self._disk_store['complevel'],
                        complib=self._disk_store['complib'],
                    )
            # close and delete uncompressed store:
            self._disk_store['store_tmp'].close()
            os.remove(self._disk_store['path_tmp'])
        else:  # if no compression, just copy ref to temp and rename
            # if no resampling, just copy ref etc
            if not self._disk_store['resample']:
                # close store to enable renaming:
                self._disk_store['store_tmp'].close()
                # rename HDFStore
                os.rename(
                    self._disk_store['path_tmp'], self._disk_store['path']
                )
                # update references:
                self._disk_store['store_tmp'] = pd.HDFStore(
                    self._disk_store['path'], mode='a'
                )
                # copy reference:
                self._disk_store['store'] = self._disk_store['store_tmp']
                print('fehler hier?')
            else:
                self._disk_store['store'] = pd.HDFStore(
                    self._disk_store['path'],
                    mode='w',
                    complevel=self._disk_store['complevel'],
                    complib=self._disk_store['complib'],
                )
                # loop over all keys in temp store and append resampled
                for key in self._disk_store['store_tmp'].keys():
                    self._disk_store['store'].put(
                        key,
                        self._disk_store['store_tmp'][key]
                        .resample(self._disk_store['resample_freq'])
                        .mean()
                        .interpolate(),
                        format='table',
                    )
                # close and delete unresampled store:
                self._disk_store['store_tmp'].close()
                os.remove(self._disk_store['path_tmp'])

    def _write_report(self):
        states = dict(zip(*np.unique(self._solver_state, return_counts=True)))
        # append 0 for states that did not occur:
        states[4] = 0 if 4 not in states else states[4]
        states[5] = 0 if 5 not in states else states[5]
        states[6] = 0 if 6 not in states else states[6]
        states[7] = 0 if 7 not in states else states[7]
        # FAILED SIM: Invalid values found
        states[98] = 0 if 98 not in states else states[98]
        # FAILED SIM: Instable solution
        states[99] = 0 if 99 not in states else states[99]

        # generate info for report
        failure = True if ((states[98] != 0) or (states[99] != 0)) else False
        fail_info = {
            'Sim. failed at step'
            if failure
            else 'Sim. solved in n steps': (self.stepnum[0]),
        }
        if failure:
            fail_info['failure reason'] = (
                'Invalid value encountered'
                if states[98] != 0
                else 'No stable solution with given min. stepsize'
            )

        qrtl_25, med, qrtl_75 = np.percentile(
            self.time_step_vec, q=[25, 50, 75]
        )

        info = {
            'MultiSim version': __version__,
            'Simulation success': True if states[99] == 0 else False,
        }
        info.update(fail_info)
        info.update(
            {
                '\n': '\n',
                'Solver information': '',
                'Normal steps (including repeated)': states[4],
                'Steps limited to maximum stepsize': states[5],
                'Steps limited to minimum stepsize': states[6],
                (
                    'Roundoff error governed step solution (truncation error is '
                    '0 or epsilon)'
                ): states[7],
                'Number of repeated steps': self._failed_steps.sum(),
                '\n': '\n',
                '': '',
                'Stepsize summary': '',
                'Total timeframe in s': self.time_vec.max(),
                'Total timeframe in d': self.time_vec.max() / 3600 / 24,
                'Mean stepsize in s': self.time_step_vec.mean(),
                'Median stepsize in s': med,
                '25% quartile stepsize in s': qrtl_25,
                '75% quartile stepsize in s': qrtl_75,
                ' ': ' ',
                'Misc. information': ' ',
                'Executed file': sys.argv[0],
                'Simulation started at': pd.to_datetime(
                    self._sim_start_time, unit='s', utc=True
                ).tz_convert('Europe/Berlin'),
                'Simulation finished at': pd.to_datetime(
                    self._sim_end_time, unit='s', utc=True
                ).tz_convert('Europe/Berlin'),
                'Simulation duration in s': (
                    self._sim_end_time - self._sim_start_time
                ),
                'Simulation duration in h': (
                    self._sim_end_time - self._sim_start_time
                )
                / 3.6e3,
                'Postprocessing duration in s': (
                    self._sim_postproc_time - self._sim_end_time
                ),
                'Steps per second': self.stepnum[0]
                / (self._sim_end_time - self._sim_start_time),
            }
        )
        rprt = pd.DataFrame.from_dict(info, orient='index')
        if self.__save_to_disk:
            rprt.to_csv(
                self._disk_store['path'].replace('.h5', '_REPORT.csv'),
                header=False,
                sep=';',
            )

    def return_stored_data(self):
        """
        Return stored result data.

        Raises
        ------
        ValueError
            If disksaving was not used for the simulation.

        Returns
        -------
        dfs : dict
            dict consisting of MultiIndex DataFrames with the simulation
            result data.

        """
        if self.__save_to_disk:
            dfs = {}
            # make sure it is open
            self._disk_store['store'].open(mode='r')
            for part in self.parts:
                try:  # try to catch parts where no res are stored
                    dfs[part] = {
                        'res': self._disk_store['store'][part + '/res'].copy(),
                        'dm': self._disk_store['store'][part + '/dm'].copy(),
                    }
                    # if part has a special method:
                    if hasattr(self.parts[part], '_special_return_store'):
                        self.parts[part]._special_return_store(
                            disk_store=self._disk_store, dfs=dfs, part=part
                        )
                except KeyError:
                    continue
            dfs['sim_vecs'] = self._disk_store['store']['sim_vecs'].copy()
            # if meters were added, also extract these:
            ds_keys = self._disk_store['store'].keys()
            mtrs = {}  # temporary dict for meters
            for k in ds_keys:  # loop over keys to find meters
                if 'meters' not in k:
                    continue
                mtrs[k.split('/')[2]] = self._disk_store['store'][k].copy()
            if len(mtrs) > 0:  # add them to dfs is meters were found
                dfs['meters'] = mtrs
            # close store
            self._disk_store['store'].close()
            return dfs
        else:
            raise ValueError('Disksaving not activated!')

    def _crop_results(self):
        """
        Crop all result arrays, time and error vectors etc. to the final size.
        """
        # cut arrays into correct length:
        self.time_vec = self.time_vec[0 : self.stepnum[0]]
        self._sys_trnc_err = self._sys_trnc_err[0 : self.stepnum[0]]
        self._failed_steps = self._failed_steps[0 : self.stepnum[0]]
        # for solver states only keep those with relevant messages:
        self._solver_state = self._solver_state[0 : self.stepnum[0]]
        # loop over parts:
        for part in self.parts:
            # for all parts:
            self.parts[part].res_dm = self.parts[part].res_dm[
                0 : self.stepnum[0]
            ]
            self.parts[part].res = self.parts[part].res[0 : self.stepnum[0]]
            # if a part has a special method:
            if hasattr(self.parts[part], '_special_crop_results'):
                self.parts[part]._special_crop_results(stepnum=self.stepnum[0])

    def _check_isadded(self, *, part, kind):
        """
        This method checks if `part` was already defined to be added to the
        simulation environment when it will be built. Thus the dictionary
        containing the postponed adding methods will be searched for the part.
        Parts of the kinds 'part', 'control' and 'open_port' will be checked.
        This method will raise an error **if the part has already been added**.
        """

        err_str = '`kind={0}` not supported.'
        assert kind in ['part', 'control', 'open_port'], err_str.format(kind)

        err_str = (
            'The {0} `name=\'{1}\'` has already been added to the '
            'simulation environment.'
        )

        if kind == 'part':
            assert part not in self.__postpone['add_part'], err_str.format(
                kind, part
            )
        if kind == 'control':
            assert part not in self.__postpone['add_ctrl'], err_str.format(
                kind, part
            )
        elif kind == 'open_port':
            assert (
                part not in self.__postpone['add_open_port']
            ), err_str.format(kind, part)

    # @classmethod
    def _check_ispart(self, *, part):
        """Check if `part` exists in self.parts."""
        # check for close part names:
        setpart = set(part)
        diffs = []
        for p in self.parts.keys():
            if len(setpart.symmetric_difference(set(p))) < 2:
                diffs.append(p)
        if len(diffs) > 0:
            err_str = (
                'Part `{0}` does not exist in simulation environment '
                '`{1}`.\nThe following existing part names have a low '
                'difference to `{0}`:\n{2}\nTypo?'
            ).format(str(part), self._simenv_name, repr(diffs))
        else:
            err_str = (
                'Part `{0}` does not exist in the simulation environment '
                '`{1}`.'
            ).format(str(part), self._simenv_name)
        assert part in self.parts, err_str

    # @classmethod
    def _check_isport(self, *, part, port):
        """Check if `port` exists at `part`."""
        self._check_ispart(part=part)  # check for part existence
        # check for close port names:
        setport = set(port)
        diffs = []
        for p in self.parts[part].port_names:
            if len(setport.symmetric_difference(set(p))) < 2:
                diffs.append(p)
        if len(diffs) > 0:
            err_str = (
                'Port `{0}` does not exist at part `{1}` in simulation '
                'environment `{2}`.\nThe following existing port names have '
                'a low difference to `{0}`:\n{3}\nTypo?'
            ).format(str(port), str(part), self._simenv_name, repr(diffs))
        else:
            err_str = (
                'Port `{0}` does not exist at part `{1}` in simulation '
                'environment `{2}`.'
            ).format(str(port), str(part), self._simenv_name)
        assert port in self.parts[part].port_names, err_str

    # @classmethod
    def _check_isinrange(self, *, part, index, target_array=None):
        """
        Check if `index` is in the range of a **flattened** array.

        Array shape of the array `target_array` of `part` is checked for index.
        """
        self._check_ispart(part=part)  # check for part existence

        targets = ['temperature', 'massflow', 'ports']
        err_str = (
            'For checking if index is in range of the FLATTENED value array '
            'of part `{0}` the target array has to be given with `target=X`, '
            'where X can be one of the following:\n' + str(targets)
        ).format(part)
        assert target_array in targets, err_str

        err_str = (
            'Index {0} for the FLATTENED {1} value array of part `{2}` in '
            'the simulation environment `{3}` must be given as an integer '
            'value. If value array ndim>2, use `array[0].reshape(-1)` to see '
            'which index to select.'
        ).format(str(index), target_array, part, self._simenv_name)
        assert isinstance(index, (int, np.integer)), err_str

        err_str = (
            'Index {0} is not in range of the FLATTENED {1} value array '
            'of part `{2}` with size {3} and shape {4} in the simulation '
            'environment `{5}`.'
        ).format(
            str(index),
            target_array,
            str(part),
            str(self.parts[part].T.size),
            str(self.parts[part].T.shape),
            self._simenv_name,
        )
        if target_array == 'temperature':
            assert index < self.parts[part].T.size, err_str
        elif target_array == 'massflow':
            assert index < self.parts[part]._dm_io.size, err_str
        elif target_array == 'ports':
            assert index < self.parts[part]._T_port.size, err_str
