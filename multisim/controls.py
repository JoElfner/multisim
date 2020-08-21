# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:20:47 2017

@author: elfner
"""

import numpy as np

from .precomp_funs import ctrl_deadtime


class Controls:
    """
    This class contains class methods for the following control modes:

    pv_const:
        Hold the process variable at a constant predefined setpoint.
    pv_diff:
        Hold the sprocess variable at a constant difference to a defined
        reference value.
    chp_eex:
        Control the electric power output of a CHP-unit according to underlying
        EEX prices.

    The control modes can be combined with the following reaction modes which
    define how a change in the reference/input value will be reacted upon on
    the time axis. This includes reactions like an immediate jump ''
    SLOPE, JUMP, EXPONENTIAL etc...

    reference port vs. reference value! make it kwargs!
    steady state: clear values of last timestep! (perhaps make something with
    a yield statement for the transient state?)
    cv: control variable
    pv: process variable
    sp: setpoint

    Parameters:
    -----------
    name : string
        Name of the controller. Can be chosen freely, only ';' is not allowed.
    kind : string
        Kind of the controller. Needs to be one of the following: `P`, `I`,
        `PI`, `PD`, `PID`.
    actuator : string
        Name of the part, which is the actuator for the current controller.
    controlled_part : string
        Name of the part, whichs **process variable** (german: Regelgröße) will
        be controlled by the controller/actuator.
    controlled_port : string, int
        Name of the port where the process variable to be controlled can be
        found in the controlled part. Can be given as a string if it is one of
        the ports or as integer index corresponding a cell in the temperature
        array `T` of the part. If given as integer index, each cell can be
        chosen, not only the ports.
    ctrl_algorithm : string
        Name of the control algorithm which can be chosen for specific parts to
        make the controller more stable with default `K`-value-settings. To get
        a list of the supported control algorithms, just leave this value empty
        and take a look at the error message.
        The chosen control algorithm MUST BE specific for the chosen
        'controlled_part' (and port)!
        The control algorithms are explained in detail at the section
        ``Control Algorithms`` below.
    reference_part : string
        Name of the part, from which a reference value for the **setpoint
        variable** (german: Führungsgröße, Sollwert) will be taken. If given,
        'reference_port' also has to be given and 'difference_to_ref' has to be
        passed to `init_controller()`. If not given, 'constant_value' has to be
        passed to `init_controller()`.
    reference_port : string, int
        Name of the port where the reference value for the setpoint variable
        can be found in the reference part. Can be given as a string if it is
        one of the ports or as integer index corresponding a cell in the
        temperature array `T` of the part. If given as integer index, each
        cell can be chosen, not only the ports.
    steady_state : bool
        Default: False. CAUTION: Setting this to true will make the controller
        assume, that the controlled part was in a steady before the current
        timestep.

    Control Algorithms:
    -------------------
        direct_ctrl:
            fddsf
        pump_HeatGen_T_out:
            sd
        pump_HeatGen_T_in:
            asd
        3w_valve_direct:
            asd
        3w_valve_:
            gldfg
        link_parts:
            Links two actuators. Both acuators must be of **the same kind**.
            The actuator having a controller has to be created first and **must
            not use** `link_parts` as control algorithm. The actuator with the
            control algorithm `link_parts` must be created after the first
            actuator. The first actuator (the one, which is not using
            `linked_parts`-algorithm) can be created in any supported way.
            The second actuator, needs to have the first actuator passed as
            **`reference_part`**, `reference_port` can be left empty.
            `controlled_part` and port can be set to any existing part, it does
            not affect the linked controller.
            The ports or controlled variable of both actuators **are the
            same**. That means for a pump which is controlling ``dm`` (the
            massflow), the linked pump also controls the massflow. For
            3-way-valves the `out1/in1` and `out2/in2` ports are connected.
            This can't be changed!

    """

    def __init__(self, name, master_cls, **kwargs):

        base_err = (  # define leading base error message
            'While adding controller `' + str(name) + '` to the simulation '
            'environment, the following error occurred:\n'
        )
        arg_err = (  # define leading error for missing/incorrect argument
            'Missing argument or incorrect type: {0}\n\n'
        )
        self._base_err = base_err  # save to self to access it in controllers
        self._arg_err = arg_err  # save to self to access it in controllers

        assert type(name) == str, (
            base_err
            + arg_err.format('name')
            + '`name` must be given as a string.'
        )
        self.name = name  # save name
        self._models = master_cls

        # ---> GET ACTUATOR
        # actuator which controls the plant using the control variable:
        assert (
            'actuator' in kwargs and kwargs['actuator'] in self._models.parts
        ), (
            base_err
            + arg_err.format('actuator')
            + 'No actuator was given with `actuator=X` or specified '
            'actuator was not found. The following actuators have been '
            'defined:\n' + str(self._models._actuators)
        )
        self.actuator = kwargs['actuator']
        # check if part can be an actuator:
        assert self._models.parts[self.actuator].is_actuator, (
            base_err
            + arg_err.format('actuator')
            + 'The selected actuator '
            + self.actuator
            + ' is not an actuator.'
        )
        # check if actuator already has a controller or is defined to not need
        # a controller:
        err_str = (
            base_err
            + arg_err.format('actuator')
            + 'Actuator `'
            + self.actuator
            + '` is either already '
            'controlled by another controller or set to not needing a '
            'controller.'
        )
        assert not self._models.parts[self.actuator].ctrl_defined, err_str

        # ---> DEFINE CONTROL ALGORITHMS
        #        # defined control algorithms: DO NOT CHANGE THE ORDER/INDEX! THIS IS
        #        # CRUCIAL FOR USING THE CORRECT ALGORITHM!!!!!!!!!!!!
        #        # DIRECT CONTROL ALGORITHMS MUST BE AT POSITION:
        #        # 0, 1*100, 2*100, ..., n*100, SO THAT position%100 == 0! This is
        #        # needed to get 'controlled_part_T_in' in method init_controller!
        #        self.ctrl_algs = np.empty([1000], dtype='<U30')
        #        self.ctrl_algs[0] = 'direct_ctrl'
        #        self.ctrl_algs[101] = 'pump_HeatGen_T_out'
        #        self.ctrl_algs[102] = 'pump_HeatGen_T_in'
        #        self.ctrl_algs[200] = '3w_valve_direct'
        #        self.ctrl_algs[201] = '3w_valve_mix'
        #        self.ctrl_algs[211] = '3w_valve_split_HeatGen'
        #        self.ctrl_algs[999] = 'link_parts'
        #        # translate control algorithm in an integer value for fast if/else
        #        # choice in calculation:
        err_str = (
            base_err
            + arg_err.format('process_CV_mode')
            + 'The processing method of the control variable CV '
            '(the controller output) has to be given with '
            '`process_CV_mode=X`, where X is one of the following:\n'
            '    - \'direct\': The CV is passed directly, without further '
            'manipulation, from the controller to the actuator.\n'
            '    - \'part_specific\': The CV is passed to the actuator '
            'part\'s `_proc_ctrl()`-method, where it will be manipulated '
            'and assigned to the correct location.\n'
            '    - \'custom\': The CV will be manipulated by a custom '
            'function, which needs to be given with `process_cv_func=Y` '
            'in the next step.\n'
            '    - \'linked\': The controller will be linked to another '
            'controller and do exactly the same as the parent controller. '
            'The parent controller must not be a linked controller '
            'itself.'
        )
        # list of supported CV processing methods:
        pcv = ['direct', 'part_specific', 'custom', 'linked']
        assert (
            'process_CV_mode' in kwargs and kwargs['process_CV_mode'] in pcv
        ), err_str
        self._process_cv_mode = kwargs['process_CV_mode']  # save choice
        if self._process_cv_mode == 'direct':
            self._process_cv_fun = self._process_cv_direct
        elif self._process_cv_mode == 'part_specific':
            self._process_cv_fun = self._models.parts[
                self.actuator
            ]._process_cv
        elif self._process_cv_mode == 'custom':  # custom function for proc.
            err_str = (
                base_err
                + arg_err.format('process_CV_mode')
                + 'If the processing method of the control variable '
                'CV was set to `process_CV_mode=\'custom\'`, the custom '
                'processing function has to be given with '
                '`process_cv_func=X`, where X is a callable taking the CV, '
                'a view to the target array cell and the timestep as '
                'arguments.'
            )
            assert 'process_cv_func' in kwargs and callable(
                kwargs['process_cv_func']
            ), err_str
            self._process_cv_fun = kwargs['process_cv_func']
        else:  # linked controller
            pass  # do nothing

        #        assert (type(ctrl_algorithm) == str and
        #                ctrl_algorithm in self.ctrl_algs), err_str
        #        # get number of control algorithm
        #        self.ctrl_alg = int(np.where(self.ctrl_algs == ctrl_algorithm)[0])

        #        # ---> GET PART DEFINITIONS: ACTUATOR, CONTROLLED PART, REFERENCE PART
        #        # actuator which controls the plant using the control variable:
        #        assert 'actuator' in kwargs and kwargs['actuator'] in self._models.parts, (
        #                base_err +
        #                'No actuator was given with `actuator=X` or specified '
        #                'actuator was not found! The following actuators have been '
        #                'defined:\n' +
        #                str(self._models._actuators))
        # #        assert actuator in self._models.parts, (
        # #                'The specified actuator `' + actuator + '` for controller `'
        # #                + name + '` was not found! The following actuators have been '
        # #                'defined:\n' + str(self._models._actuators))
        #        self.actuator = kwargs['actuator']
        #        # check if part can be an actuator:
        #        assert self._models.parts[self.actuator].is_actuator, (
        #                base_err + 'The selected actuator ' + self.actuator + ' is '
        #                'not an actuator!')
        #        # check if actuator already has a controller or is defined to not need
        #        # a controller:
        #        err_str = ('Actuator `' + self.actuator + '` is either already '
        #                   'controlled by another controller or set to not needing a '
        #                   'controller!')
        #        assert not self._models.parts[self.actuator].ctrl_defined, base_err + err_str

        # check and get lower and upper limits of actuator if not linked:
        #        err_str = ('No lower and upper limits have been specified for '
        #                   'actuator ' + self.actuator + '!')
        #        if self.ctrl_alg != 999:
        #            assert 'lims' in self._models.parts[self.actuator].__dict__, (
        #                    base_err + err_str)
        #            self._llim, self._ulim = self._models.parts[self.actuator].lims

        err_str = (
            self._base_err
            + self._arg_err
            + 'A controller requires the following additional arguments:\n'
            '    - `CV_saturation`: List or tuple with the lower and upper '
            'saturation limit of the control variable (CV, the controller '
            'output). CV will be clipped to `lower_lim <= CV <= upper_lim` '
            'before passing it to the CV processing method (where part '
            'specific limits can also be applied). For a '
            'Bang-Bang-controller, the upper limit is the on-value.\n'
            'Negative and infinite values are allowed with '
            '`lower_lim < upper lim`. For example '
            '`CV_saturation=(lower_lim, upper_lim)=(-1, np.inf)`.\n'
        )
        # CV saturation:
        assert (
            'CV_saturation' in kwargs
            and isinstance(kwargs['CV_saturation'], (list, tuple))
            and isinstance(kwargs['CV_saturation'][0], (int, float))
            and isinstance(kwargs['CV_saturation'][1], (int, float))
        ), err_str.format('CV_saturation')
        self._cv_sat_min = float(kwargs['CV_saturation'][0])
        self._cv_sat_max = float(kwargs['CV_saturation'][1])
        assert self._cv_sat_min < self._cv_sat_max, err_str.format(
            'CV_saturation'
        )

        # also get lower and upper limits from actuators (these correspond to
        # the hardware-limits, while CV sat. is more like a software limit):
        err_str = (
            base_err
            + arg_err.format('lower_limit, upper_limit')
            + 'No lower and upper limit have been specified for actuator `'
            + str(self.actuator)
            + '`. These need to be specified when adding '
            'the actuator to the simulation environment.'
        )
        assert hasattr(self._models.parts[self.actuator], '_llim') and hasattr(
            self._models.parts[self.actuator], '_ulim'
        ), err_str
        self._llim = self._models.parts[self.actuator]._llim
        self._ulim = self._models.parts[self.actuator]._ulim

        # part whichs behaviour has to be controlled and at which port to
        # control (where to get the process variable from)
        # check if all required inputs given, if parts/ports exist and save
        # part name as string and port as index
        assert 'controlled_part' in kwargs, (
            base_err + 'The part at which the process variable to be '
            'controlled is located has to be given with '
            '`controlled_part=X`, where X is the name of the part.'
        )
        assert 'controlled_port' in kwargs, (
            base_err + 'The location at which the process variable to be '
            'controlled is located at the controlled part has to be given '
            'with `controlled_port=X`, where X is either an index to the '
            'array cell or the name of the port.'
        )
        self.ctrl_part, self.ctrl_port_id = self.__check_args(
            'controlled_part',
            'controlled_port',
            'add_control',
            controlled_part=kwargs['controlled_part'],
            controlled_port=kwargs['controlled_port'],
        )
        # also save port name as string for easy look-up
        self.ctrl_port = kwargs['controlled_port']
        # also save ID of part for fast access:
        self.ctrl_part_id = self._models.parts[self.ctrl_part].part_id
        # get memoryview to controlled port value as process variable using
        # the port/cell ID ([id:id+1:1] to slice and also enable getting last
        # value in array):
        self.pv = self._models.parts[self.ctrl_part].T[
            self.ctrl_port_id : self.ctrl_port_id + 1 : 1
        ]

        # if reference part and port are set, get IDs and memoryview:
        err_str = (
            base_err + '`reference_part=X` has to be given. The reference '
            'part, if specified, will be used to either get a reference set '
            'point (SP) value from that part OR to define the '
            'parent/reference actuator of a linked controller when '
            '`process_CV_mode=\'linked\'` is set. X can be one of '
            'the following:\n'
            '    - \'none\': No reference part can be used. The SP value is '
            'constant or a predefined time series.\n'
            '    - part name: The name of the reference part. The SP value '
            'can be taken from this part or this part can be used as a '
            'parent/reference actuator if the controller is set to '
            '\'linked\'.'
        )
        assert 'reference_part' in kwargs and (
            kwargs['reference_part'] == 'none'
            or kwargs['reference_part'] in self._models.parts
        ), err_str
        if not kwargs['reference_part'] == 'none':
            err_str = (  # check for ref port
                base_err + 'A reference part is set. The reference port must '
                'be given with `reference_port=X`, where X is either a string '
                'matching the name of a port at the reference part or an '
                'integer index to a cell of the value array of the reference '
                'part.'
            )
            assert 'reference_port' in kwargs and isinstance(
                kwargs['reference_port'], (str, int)
            ), err_str
            # check if all required inputs given, if parts/ports exist and
            # save part name as string and port as index
            self.ref_part, self.ref_port_id = self.__check_args(
                'reference_part',
                'reference_port',
                reference_part=kwargs['reference_part'],
                reference_port=kwargs['reference_port'],
            )
            # also save port name as string for easy look-up
            self.ref_port = kwargs['reference_port']  # (if given as string)
            # also save ID of part for fast access:
            self.ref_part_id = self._models.parts[self.ref_part].part_id
            # get memoryview to reference port value as reference variable
            # using the port/cell ID. This will be summed up with the
            # differece value, if given, and passed to to controller as
            # setpoint variable
            self._ref = self._models.parts[self.ref_part].T[
                self.ref_port_id : self.ref_port_id + 1 : 1
            ]
            # save bool that ref port is used:
            self.use_ref = True
        else:
            self.ref_part = 'none'
            self.ref_port = 'none'
            # set if reference port is used
            self.use_ref = False

        # get setpoint (SP):
        if self.ref_part == 'none' and self._process_cv_mode != 'linked':
            # if no reference part: get constant setpoint (SP) value
            err_str = (
                base_err + 'Since `reference_part=\'none\'` is set, the '
                'setpoint (SP) value must be given with `setpoint=X`, '
                'where X is an integer or float value.\n'
                'To set a value of another part as SP value, this part '
                'must be defined as `reference_part`. To set a dynamic '
                'time series as SP value, first set a constant value and '
                'use the method `simenv.assign_boundary_cond()`, where '
                'simenv refers to the chosen simulation environment name, '
                'to assign a time series to this SP value.'
            )
            assert 'setpoint' in kwargs and isinstance(
                kwargs['setpoint'], (int, float)
            ), err_str
            self.sp = np.array([kwargs['setpoint']], dtype=np.float64)
        elif self._process_cv_mode != 'linked':
            # get SP as reference to ref part
            err_str = (
                base_err + '`reference_part=' + self.ref_part + '` is set for '
                'an unlinked controller, thus the setpoint (SP) value will '
                'be taken from the reference part. A constant difference to '
                'this SP will be applied with `SP = ref_value + ref_SP_diff`. '
                'This difference must be given with `ref_SP_diff=X`, where X '
                'is an integer or float value. 0, negative and positive '
                'values are allowed.'
            )
            assert 'ref_SP_diff' in kwargs and isinstance(
                kwargs['ref_SP_diff'], (int, float)
            ), err_str
            self._ref_diff = float(kwargs['difference_to_ref'])

        #        # ---> CHECK CTRL ALGORITHMS
        #        # check if actuator definetely needs a special control (all actuators
        #        # with 2 outputs) AND if it is not just linked to another (controlled)
        #        # actuator:
        #        if (self._models.parts[self.actuator].actuator_special and
        #                self.ctrl_alg != 999):
        #            if self._models.parts[self.actuator].constr_type == '3w_valve':
        #                #  check if 3w_valve is in the range for the 3w_valve algs:
        #                err_str = ('A 3-way-valve needs a special control algorithm. '
        #                           'Chose one of the following: ' +
        #                           str(self.
        #                               ctrl_algs[200:300][self.
        #                                                  ctrl_algs[200:300] != '']))
        #                assert 200 <= self.ctrl_alg < 300, err_str
        #                # for a 3w_valve the port for the actuator has to be given:
        #                err_str = ('If a 3-way-valve has been chosen as actuator, the '
        #                           'A or B port which primarily has to be controlled '
        #                           '(positive feedback on controller action, that '
        #                           'means that the control variable, for example the '
        #                           'valve opening of the port, is proportional to the '
        #                           'error) needs to be chosen by passing it to '
        #                           '`add_control()` as `actuator_port=port_name`.')
        #                assert 'actuator_port' in kwargs, err_str
        #                self.act_port = kwargs['actuator_port']
        #                # get actuator port id:
        #                """
        #                TO DO:
        #                    Check if kwargs['actuator_port'] is A or B and not AB!
        #                    But need to change valve port naming scheme to A/B/AB
        #                    before!
        #                """
        #                self.act_port = self.__check_args(
        #                        'actuator', 'actuator_port', caller='add_control',
        #                        actuator=self.actuator,
        #                        actuator_port=(self.act_port))[1]
        #                # set the in-or-outlet, which is not chosen as actuator port,
        #                # as the second port to be controlled by the algorithm:
        #                self.act_port_scnd = 1 - self.act_port
        #
        #        # LINKING ALGORITHM:
        #        # if ctrl_alg == 'link_parts' == 999, the current actuator will be
        #        # linked to another actuator, so that the current actuator will always
        #        # have the same values as the other actuator. only works, if the
        #        # actuators are of the same type:
        #        if self.ctrl_alg == 999:
        #            # check if reference part is given:
        #            self.ref_part = self.__check_args('reference_part', 'no_port',
        #                                              'add_control',
        #                                              reference_part=reference_part)
        #            assert (self._models.parts[self.actuator].constr_type ==
        #                    self._models.parts[self.ref_part].
        #                    constr_type), ('If one actuator\'s controls shall be '
        #                                   'linked to another actuator\'s control, so '
        #                                   'that the control variables will be '
        #                                   'copied, both actuators have to be of the '
        #                                   'same type.')
        #            # find other actuator in controls:
        #            for i, controls in enumerate(self._models.ctrl_l):
        #                # if found, check other actuators controls:
        #                if controls.actuator == self.ref_part:
        #                    # if other actuator is direct control algorithm, get its
        #                    # defined actuator Cv:
        #                    if controls.ctrl_alg == 0:
        #                        self._act_cv_name = controls._act_cv_name
        #                        # create memoryview from own actuator cv to linked
        #                        # actuator cv:
        #                        self._models.parts[self.actuator].__dict__[
        #                                self._act_cv_name][:] = (
        #                                self._models.parts[self.ref_part].__dict__[
        #                                        self._act_cv_name][:])
        #                    # if actuators are 3w_valves:
        #                    elif self._models.parts[self.actuator].constr_type == '3w_valve':
        #                        # link port factor array of 3w_valve with memoryview to
        #                        # other linked 3w_valves port factor array:
        #                        self._models.parts[self.actuator].pf_arr = (
        #                                self._models.parts[controls.actuator].pf_arr[:])
        #                    else:
        #                        raise TypeError('The currently chosen actuator does '
        #                                        'not (yet) support linking! To '
        #                                        'implement linking for this actuator '
        #                                        'search for \'LINKING ALGORITHM:\' in '
        #                                        'the Controls class.')
        #
        #        # DIRECT CONTROL ALGORITHM INITIALIZATION:
        #        # check if variable to control is given for direct control algorithm:
        #        if self.ctrl_alg == 0:
        #            # assert if actuator CV is given:
        #            assert ('self.actuator_CV' in
        #                    kwargs), ('If direct control algorithm is chosen, the '
        #                              'control variable name has to be given! It must '
        #                              'be passed to `add_control()` as '
        #                              '`self.actuator_CV=X`, with X being a '
        #                              'string which exactly matches the name of the '
        #                              'variable to control, for example \'dm\' to '
        #                              'control the massflow of a pump.')
        #            # assert if given actuator CV exists:
        # #            assert (kwargs['self.actuator_CV'] in
        # #                    self._models.parts[self.actuator].
        # #                    __dict__), ('The given \'self.actuator_CV\' ' +
        # #                                kwargs['self.actuator_CV'] +
        # #                                ' for controller ' + self.name +
        # #                                ' and actuator ' + self.actuator +
        # #                                ' does not exist!')
        #            err_str = ('The given \'self.actuator_CV\' ' +
        #                                kwargs['self.actuator_CV'] +
        #                                ' for controller ' + self.name +
        #                                ' and actuator ' + self.actuator +
        #                                ' does not exist!')
        #            assert hasattr(self._models.parts[self.actuator],
        #                           kwargs['self.actuator_CV']), err_str
        #            # set actuator CV as memoryview to selected actuator CV:
        #            self._act_cv_name = kwargs['self.actuator_CV']
        #            self._act_cv = (
        #                    self._models.parts[self.actuator].__dict__[self._act_cv_name][:])

        # ---> CHECK CTRL ALGORITHMS
        # check if actuator definetely needs a special control (all actuators
        # with 2 outputs) AND if it is not just linked to another (controlled)
        # actuator:
        if (
            self._models.parts[self.actuator].actuator_special
            and self._process_cv_mode != 'linked'
        ):
            # for a 3w_valve the port for the actuator has to be given:
            err_str = (
                base_err
                + 'The selected actuator `{0}` requires a special CV processing '
                'method, thus `process_CV_mode=\'direct\'` is not supported.\n'
                '\n\n'
                'Deprecated - Not used anymore:\n'
                '(Furthermore the port or cell position which has to be '
                'controlled (positive feedback on controller action, that '
                'means that the control variable, for example the valve '
                'opening of the port, is proportional to the error) must be '
                'given with `actuator_port=X`, where X is either the '
                'port name as a string or the cell index as integer value.)'
            ).format(self.actuator)
            assert 'actuator_port' in kwargs, err_str
            self.act_port = kwargs['actuator_port']
            # check actuator port and get index if not already int:
            self.act_port = self.__check_args(
                'actuator',
                'actuator_port',
                actuator=self.actuator,
                actuator_port=(self.act_port),
            )[1]
            # set the in-or-outlet, which is not chosen as actuator port,
            # as the second port to be controlled by the algorithm:
            self.act_port_scnd = 1 - self.act_port
        # LINKING ALGORITHM:
        # if linked controller, the current actuator will be
        # linked to another actuator, so that the current actuator will always
        # have the same values as the other actuator. only works, if the
        # actuators are of the same type:
        elif self._process_cv_mode == 'linked':
            err_str = (
                base_err + 'The controller is set to be linked. A linked '
                'controller requires a reference actuator, to which its '
                'actuator will be linked. This reference actuator must be '
                'given with `reference_part=X`, where X is the name of the '
                'reference actuator.'
            )
            assert self.ref_part != 'none', err_str
            #            # check if reference part is given:
            #            self.ref_part = self.__check_args(
            #                    'reference_part', 'no_port', reference_part=reference_part)
            assert (
                self._models.parts[self.actuator].constr_type
                == self._models.parts[self.ref_part].constr_type
            ), (
                base_err + 'If `process_CV_mode=\'linked\'` is set, '
                'the current actuator `' + self.actuator + '` has to '
                'be of the same part type as the reference/parent '
                'actuator `' + self.ref_part + '`'
            )
            # find other actuator in controls:
            found = False  # init bool checker
            for i, controls in enumerate(self._models.ctrl_l):
                # if found, check other actuators controls:
                if controls.actuator == self.ref_part:
                    found = True
                    # if other actuator is direct control algorithm, get its
                    # defined actuator CV:
                    if controls._process_cv_mode == 'direct':
                        # copy name of actuator control variable:
                        self._act_cv_name = controls._act_cv_name
                        # create memoryview from own actuator cv to linked
                        # actuator cv:
                        raise ValueError(
                            'this does not create a memoryview!'
                            'But is it ok to reallocate any array '
                            'here to create memoryviews? For '
                            'example does flow not still work '
                            'after reallocating?'
                        )
                        self._models.parts[self.actuator].__dict__[
                            self._act_cv_name
                        ][:] = self._models.parts[self.ref_part].__dict__[
                            self._act_cv_name
                        ][
                            :
                        ]
                    # if actuators are 3w_valves:
                    #                    elif self._models.parts[self.actuator].constr_type == '3w_valve':
                    #                        # link port factor array of 3w_valve with memoryview to
                    #                        # other linked 3w_valves port factor array:
                    #                        self._models.parts[self.actuator].pf_arr = (
                    #                                self._models.parts[controls.actuator].pf_arr[:])
                    # if actuators have special control method:
                    elif (
                        controls._process_cv_mode == 'part_specific'
                        or controls._process_cv_mode == 'custom'
                    ):
                        # copy name of actuator control variable:
                        self._act_cv_name = controls._act_cv_name
                        # check if actuators have a linking method:
                        try:
                            raise ValueError(
                                'this does not create a memoryview!'
                                'But is it ok to reallocate any array '
                                'here to create memoryviews? For '
                                'example does flow not still work '
                                'after reallocating?'
                            )
                            self._models.parts[self.actuator].__dict__[
                                self._act_cv_name
                            ][:] = self._models.parts[
                                self.ref_part
                            ]._link_actuator()
                        except AttributeError:
                            raise TypeError(
                                'The currently chosen actuator does '
                                'not (yet) support linking! To '
                                'implement linking for this actuator '
                                'search for \'LINKING ALGORITHM:\' in '
                                'the Controls class.'
                            )
                    break
            if not found:
                err_str = (
                    base_err
                    + arg_err.format(str(self.ref_part))
                    + 'No parent/reference controller with actuator `'
                    + self.ref_part
                    + '` was found!'
                )

        # replaced with getting act cv from actuators itself
        #        # DIRECT CONTROL ALGORITHM INITIALIZATION:
        #        # check if variable to control is given for direct control algorithm:
        #        if self._process_cv_mode == 'direct':
        #            # assert if actuator CV is given:
        #            assert ('actuator_CV' in kwargs and
        #                    type(kwargs['actuator_CV']) == str), (
        #                base_err + arg_err.format('actuator_CV') +
        #                'If direct control algorithm is chosen, the '
        #                'control variable name has to be given with '
        #                '`actuator_CV=X`, where X is a string which '
        #                'matches the name of the variable to control '
        #                'exactly, for example \'dm\' to  control the '
        #                'massflow of a pump.')
        #            # assert if given actuator CV exists:
        # #            assert (kwargs['self.actuator_CV'] in
        # #                    self._models.parts[self.actuator].
        # #                    __dict__), ('The given \'self.actuator_CV\' ' +
        # #                                kwargs['self.actuator_CV'] +
        # #                                ' for controller ' + self.name +
        # #                                ' and actuator ' + self.actuator +
        # #                                ' does not exist!')
        #            err_str = (
        #                base_err + arg_err.format('actuator_CV') +
        #                'The given `actuator_CV=' + kwargs['actuator_CV'] +
        #                '` for actuator ' + self.actuator + ' does not exist.')
        #            assert hasattr(self._models.parts[self.actuator],
        #                           kwargs['actuator_CV']), err_str
        #            # set actuator CV as memoryview to selected actuator CV:
        #            self._act_cv_name = kwargs['actuator_CV']
        #            self._act_cv = (
        #                    self._models.parts[self.actuator].__dict__[self._act_cv_name][:])

        # get actuator CV name and memory view to the actuator CV:
        self._act_cv_name = self._models.parts[self.actuator]._actuator_CV_name
        self._act_cv = self._models.parts[self.actuator]._actuator_CV[:]

        # ---> get controller dependencies (of other controllers):
        # check if dependencies exist:
        err_str = (
            base_err
            + arg_err.format('sub_controller')
            + 'The argument `sub_controller=True/False`, depicting '
            'if the controller\'s action is depending on another '
            '(already added) controller or part, is missing. Please pass '
            'the missing argument as a bool value.'
        )
        assert (
            'sub_controller' in kwargs
            and type(kwargs['sub_controller']) == bool
        ), err_str
        self.sub_ctrl = kwargs['sub_controller']
        # if True, get further information:
        if self.sub_ctrl:
            # get information if master is a part or a controller:
            err_str = (
                base_err
                + arg_err.format('master_type')
                + '`sub_control=True` is set. Please pass the '
                'parameter `master_type=X`, depicting if the controller '
                'action is depending on another controller or on another '
                'part. Thus `X` must either be \'part\' or \'controller\'.'
            )
            assert (
                'master_type' in kwargs
                and type(kwargs['master_type']) == str
                and kwargs['master_type'] in ['part', 'controller']
            ), err_str
            self._master_type = kwargs['master_type']
            # get additional information depending on type:
            if self._master_type == 'controller':  # if master is a controller
                # get master controller:
                err_str = (
                    base_err
                    + arg_err.format('master_controller')
                    + '`sub_control=True` and '
                    '`master_type=\'controller\'` are set. The controller, '
                    'on which the action of `' + self.name + '` is depending, '
                    'has to be given with `master_controller=X`, where `X` is '
                    'the name of the (already added) master controller. The '
                    'following controllers have already been added:\n'
                    + str(list(self._models.ctrls))
                )
                assert (
                    'master_controller' in kwargs
                    and kwargs['master_controller'] in self._models.ctrls
                ), err_str
                self.master_ctrl = kwargs['master_controller']  # save name
                # get memory view to master controller's actuator cv value:
                self._master_act_cv = self._models.ctrls[
                    self.master_ctrl
                ]._act_cv[:]
                #                # get lower limit of master controller to check if active:
                #                self._master_act_llim = self._models.ctrls[self.master_ctrl]._llim
                # get off-state of master controller to check if active:
                self._master_act_offst = self._models.ctrls[
                    self.master_ctrl
                ]._off_state
            else:  # else if master is a part
                # get master part:
                err_str = (
                    base_err
                    + arg_err.format('master_part')
                    + '`sub_control=True` and `master_type=\'part\'` '
                    'are set. The part, on which the action of `'
                    + self.name
                    + '` is depending, has to be given with `master_part=X`,'
                    ' where `X` is the name of the master '
                    'part. The following parts have been added:\n'
                    + str(list(self._models.parts))
                )
                assert (
                    'master_part' in kwargs
                    and kwargs['master_part'] in self._models.parts
                ), err_str
                self.master_part = kwargs['master_part']
                # get dependency variable:
                err_str = (
                    base_err
                    + arg_err
                    + '`sub_control=True`, `master_type=\'part\'` '
                    'and `master_part=' + self.master_part + '` are set. '
                    'The variable of the part, on which the action of `'
                    + self.name
                    + '` is depending, has to be given with '
                    '`master_variable=X`, where `X` is the name of variable '
                    'in the master part as a string. The variable in the '
                    'master part has to be stored in an array.\n'
                    'Additionally the array index of the storage location '
                    'of the master part variable has to be given with '
                    '`master_variable_index=X`, where `X` is the integer '
                    'index to the variable array >=0. Currently only '
                    '1D indices are supported.'
                )
                # assert variable:
                assert 'master_variable' in kwargs, err_str.format(
                    'master_variable'
                )
                assert hasattr(
                    self._models.parts[self.master_part],
                    kwargs['master_variable'],
                ), (
                    base_err
                    + arg_err
                    + '`master_variable=\'{0}\'`'.format(
                        kwargs['master_variable']
                    )
                    + ' was not found '
                    'in master part `{0}`.'.format(self.master_part)
                ).format(
                    'master_variable'
                )
                # TODO: master_variable='dm'  -->
                # only allowed for pumps (currently), since otherwise
                # views to master variable will be broken in get_topology for
                # most parts in the part specific _get_flow_routine method.
                # solution: construct a sim env wide dict where informations
                # for all memory views are stored to restore them after
                # topology construction. will this work without breaking other
                # stuff? probably EXCLUDE all topology consruction memory views
                # from this!
                # for now: break if non-pump master variables are set to dm
                if kwargs['master_variable'] == 'dm':
                    assert self._models.parts[
                        self.master_part
                    ].constr_type in (
                        'PipeWithPump',
                        'PPump',
                    ), 'err see above, tODO'
                self._master_variable = kwargs['master_variable']  # save name
                assert 'master_variable_index' in kwargs and isinstance(
                    kwargs['master_variable_index'], int
                ), err_str.format('master_variable_index')
                # save index:
                self._master_variable_index = kwargs['master_variable_index']
                # check for correct index:
                err_str = (
                    base_err + 'Index {0} is out of range for the chosen '
                    'master variable with array shape {1}'.format(
                        self._master_variable_index,
                        getattr(
                            self._models.parts[self.master_part],
                            self._master_variable,
                        ).shape,
                    )
                )
                assert (
                    0
                    <= self._master_variable_index
                    < getattr(
                        self._models.parts[self.master_part],
                        self._master_variable,
                    ).shape[0]
                ) and isinstance(
                    self._master_variable_index, int
                ), err_str.format(
                    'master_variable_index'
                )
                # get memory view to master variable index location:
                self._master_act_cv = getattr(
                    self._models.parts[self.master_part],
                    self._master_variable,
                    None,
                )[
                    slice(
                        self._master_variable_index,
                        self._master_variable_index + 1,
                    )
                ]
                #                # set lower limit of master "actuator" (in fact it is NOT an
                #                # actuator) to 0 to check if active:
                #                self._master_act_llim = 0.
                # set off_state of master "actuator" (in fact it is NOT an
                # actuator) to 0 to check if active:
                self._master_act_offst = 0.0
            # get kind of dependency:
            err_str = (
                base_err
                + arg_err.format('dependency_kind')
                + '`sub_control=True` is set. The kind of '
                'dependency has to be given with `dependency_kind=X`, where '
                '`X` is one of the following strings:\n'
                '    \'concurrent\': The actuator controlled by `'
                + self.name
                + '` is only in action, when the actuator controlled by the '
                'master controller is in action or the master part variable '
                'is NOT zero.\n'
                '    \'sequential\': The actuator controlled by `'
                + self.name
                + '` is only in action, when the actuator controlled by the '
                'master controller is NOT in action or the master part '
                'variable is zero.'
            )
            dep_list = ['concurrent', 'sequential']  # list of dependencies
            assert (
                'dependency_kind' in kwargs
                and type(kwargs['dependency_kind']) == str
                and kwargs['dependency_kind'] in dep_list
            ), err_str
            self.dependency_kind = kwargs['dependency_kind']  # kind of dep.
            # save to bool variable for fast checking:
            if self.dependency_kind == 'concurrent':
                self._concurrent = True
            else:
                self._concurrent = False

        # get off state of controller (mostly for the use in subcontrollers
        err_str = (  # but also helpful for general checks)
            base_err
            + arg_err.format('off_state')
            + 'The control variable (CV, the controller output), which is set '
            'when the actuator is switched off, has to be  given with '
            '`off_state=X`, where X is an integer or float value in the units '
            'of the controlled actuator variable (for example [kg/s] for the '
            'massflow of a pump).\n'
            'This value will not be set by the controller, but will be used '
            'to check if a controller has switched off an actuator.'
        )
        assert 'off_state' in kwargs and isinstance(
            kwargs['off_state'], (int, float)
        ), err_str
        self._off_state = float(kwargs['off_state'])

        # ---> Time domain initialization
        # check if controller was defined as continuous or discrete time domain
        # controller:
        err_str = (
            base_err
            + arg_err.format('time_domain')
            + 'The time domain in which the controller is defined '
            'must be given with `time_domain=\'continuous\'` or '
            '`time_domain=\'discrete\'`.\n'
            'As a rule of thumb the simulation timestep needs to be at '
            'least 10 to 20 times smaller than the smallest plant time '
            'constant (approximately the time until a value change '
            'can be observed after a step of the controller) in the '
            'part of the plant which has to be controlled by a '
            'continuous time controller. When using adaptive '
            'timesteps, the average simulation timestep needs to be at '
            'least 20 times smaller than smallest plant time constant. '
            'Otherwise the use of a discrete controller is encouraged.'
        )
        assert 'time_domain' in kwargs and kwargs['time_domain'] in [
            'discrete',
            'continuous',
        ], err_str
        # save to bool checker:
        if kwargs['time_domain'] == 'discrete':
            self.discrete_time = True
            # set default values for values only needed in discrete controller:
            self.prev_cv = 0.0
        elif kwargs['time_domain'] == 'continuous':
            self.discrete_time = False
            # set default values for values only needed in conti controller:
            self.error_sum = 0.0
            self.i_term = 0.0
            self.prev_i_term = 0.0
            self.d_term = 0.0

        # check for deadtime and get deadtime:
        err_str = (
            base_err
            + arg_err.format('deadtime')
            + 'The deadtime of the controller has to be given with '
            '`deadtime=X`, where X is the deadtime in seconds as an integer '
            'or float value >=0. The deadtime is the time delay of the '
            'controller. To omit the use of deadtime for this controller, '
            '`deadtime=0` can be passed.'
        )
        assert (
            'deadtime' in kwargs
            and isinstance(kwargs['deadtime'], (int, float))
            and kwargs['deadtime'] >= 0
        ), err_str
        self.deadtime = kwargs['deadtime']
        # construct deadtime array with 100 times the length of
        # deadtime (divided by a 1s timestep):
        self.dt_arr = np.zeros(np.ceil(self.deadtime * 100).astype(int))
        # PV array same shape as dt_arr and with starting value 25:
        self.pv_arr = np.ones_like(self.dt_arr) * 25
        # save length of arrays:
        self.len_dt_arr = self.dt_arr.shape[0]

        # get and set cv slope
        err_str = (
            base_err
            + arg_err.format('slope')
            + 'The maximum negative and positive slope of the control variable '
            '(CV, the controller output) have to be given with `slope=X`, '
            'where X is a tuple of integer or float values in [CV_unit/s] '
            'defining the maximum change of the CV per second, like '
            '`slope=(negative_slope, positive_slope)`. The slope of CV will '
            'be clipped to `negative_slope <= CV_slope <= positive_slope`. To '
            'disable the use of a slope, set `slope=(0, 0)`.\n'
            'The following restrictions apply:\n'
            'negative_slope < 0 < positive_slope\n'
            'Warning: The use of a slope limits the impact of '
            '`sub_controller=True`, since the change of the CV will be '
            'limited to the slope independently of the slope of the master '
            'controller.'
        )
        assert (
            'slope' in kwargs
            and type(kwargs['slope']) == tuple
            and len(kwargs['slope']) == 2
            and isinstance(kwargs['slope'][0], (int, float))
            and isinstance(kwargs['slope'][1], (int, float))
        ), err_str
        self._slope_n = float(kwargs['slope'][0])  # save negative slope
        self._slope_p = float(kwargs['slope'][1])  # save positive slope
        if (
            self._slope_n == 0.0 and self._slope_p == 0.0
        ):  # if both are set to 0
            self._slope = False  # don't use slope
        else:  # if both are NOT set to 0, use slope
            self._slope = True  # use slope
            assert self._slope_n < 0.0 < self._slope_p, err_str  # asser restr.

        # invert controller action (f.i. increase pump flow with negative
        # err, that is f.i. sp=85, pv=95 -> increase flow to cool pv):
        err_invert = (
            base_err
            + arg_err.format('invert')
            + 'Invert controller action? That is f.i. increase positive pump '
            'flow for a negative error, f.i. sp=85, pv=95 -> err=-10 -> '
            'increase pump flow. Useful f.i. for cooling operations to not '
            'have to invert the pump operation direction. Type: bool'
        )
        assert 'invert' in kwargs and isinstance(
            kwargs['invert'], bool
        ), err_invert
        # set to 1 if no inverting, else -1 if inverting
        self._invert = 1 if not kwargs['invert'] else -1

        # check if steady state PID controller of transient state:
        if 'steady_state' in kwargs:
            self._steady_state = kwargs['steady_state']
        else:
            self._steady_state = False

        # ---> set default values
        # set default values for cva_max, cva_min (max/min change per step to
        # the process variable), anti_windup (maximum i-term):
        #        self.cv_a_max = np.inf
        #        self.cv_a_min = - np.inf
        #        self.anti_windup = 50.0
        # set default values for values of last step (also starting values for
        # first step) for conti and discrete controller:
        self.cv = 0.0
        self.error = 0.0
        self.prev_error = 0.0
        # defaults for low pass filtering derivate error
        self._prev_deriv = 0.0
        self._deriv_err = 0.0
        self._deriv = 0.0
        # set default values for deadtime calculation:
        self.dt_idx = 0
        self.last_dt_idx = 0
        self.delayed_pv = 25

        # set actuator is controlled status to True:
        self._models.parts[self.actuator].ctrl_defined = True
        self._models.parts[self.actuator]._ctrld_by = name

        # set initialized to False:
        self.initialized = False

    def init_controller(self, **kwargs):
        # check if kwargs exists if not linked controller:
        if not kwargs and self.ctrl_alg != 999:
            err_str = (
                'No arguments have been passed to `init_controller()` '
                'for controller ' + self.name + '! '
                'Since each controller type or controller algorithm '
                '(except for a linked controller) requires a specific '
                'set of arguments, just pass a random argument like '
                '`abc=55` to `init_controller()` and all missing '
                'required arguments will be asked for in error '
                'messages!'
            )
            raise ValueError(err_str)

        #        # else get kwargs if not linked controller:
        #        elif self.ctrl_alg != 999:

        # ALGORITHM INITIALIZATIONS:
        #            # HEATGENERATOR/EXCHANGER ALGORITHM INITIALIZATION:
        #            # check if heatgen/heatexchanger algorithm is chosen (211-214 is
        #            # heatgen/heatexchanger algorithm of splitting valve!):
        #            if 100 < self.ctrl_alg < 200 or 211 <= self.ctrl_alg <= 213:
        #                # ensure that T_in port is given:
        #                err_str = ('For the current control algorithm for heat '
        #                           'generators or heat exchangers for controller '
        #                           + self.name + ' with actuator ' + self.actuator
        #                           + ', a port or cell has to be given, which defines '
        #                           'the position of the inlet temperature of the '
        #                           'controlled part (generator/exchanger): '
        #                           + self.ctrl_part + '\n'
        #                           'This is needed to calculate the actuator value. '
        #                           'The port name or the cell index of the inlet '
        #                           'temperature has to be passed to '
        #                           '`init_controller()` as: '
        #                           '\'controlled_part_T_in = port name or '
        #                           'cell index\'.')
        #                assert ('controlled_part_T_in' in kwargs), err_str
        #                # get index and check for existence:
        #                self.ctrl_part_T_in_id = (
        #                        self.__check_args('controlled_part',
        #                                          'controlled_part_T_in',
        #                                          controlled_part=self.ctrl_part,
        #                                          **kwargs)[1])
        #                # make memoryview to the location of T_in:
        #                self.ctrl_part_T_in = (
        #                        self._models.parts[self.ctrl_part].T[
        #                                self.ctrl_part_T_in_id:
        #                                self.ctrl_part_T_in_id + 1:1])
        #                # check if selected control algorithm contains 'HX' and thus
        #                # is meant to control a HeateXchanger:
        #                if 'HX' in self.ctrl_algs[self.ctrl_alg]:
        #                    # check if additionally needed information is given:
        #                    err_str = ('The flow over a heat exchanger (HX) has to '
        #                               'be controlled and a special control algorithm '
        #                               'was chosen for controller ' + self.name +
        #                               ' and actuator ' + self.actuator + '.\n'
        #                               'Thus the following arguments have to be '
        #                               'passed to the controllers `init_controller()` '
        #                               ' method: the heat exchangers part name as '
        #                               '`HX_part_name=...` and the side of the heat '
        #                               'exchanger at which the actuator is situated '
        #                               'as `HX_actuator_side=\'sup/dmd\'` which can '
        #                               'be either \'sup\' for the supply side or '
        #                               '\'dmd\' for the demand side.')
        #                    assert ('HX_part_name' in kwargs
        #                            and 'HX_actuator_side' in kwargs), err_str
        #                    # save HX actuator side:
        #                    self.HX_act_side = kwargs['HX_actuator_side']
        #                    # check if sup/dmd given correctly:
        #                    err_str = ('The side of the heat exchanger, passed to '
        #                               '`init_controller()` of controller ' + self.name
        #                               + ', where the actuator is located has to be '
        #                               'either \'sup\' or \'dmd\'!')
        #                    assert (self.HX_act_side == 'sup'
        #                            or self.HX_act_side == 'dmd'), err_str
        #                    # add HX part and the ids of the in/out ports and check:
        #                    # if supply side, get part name, port ids, check for
        #                    # their existence and construct memoryviews:
        #                    if self.HX_act_side == 'sup':
        #                        # get part and T_in_id (id only needed for convenience
        #                        # of easy lookup by user)
        #                        (self.HX,
        #                         self.HX_act_side_T_in_id
        #                         ) = (self.__check_args('HX_part_name',
        #                                                'HX_act_side_T_in',
        #                                                caller='init_controller',
        #                                                HX_act_side_T_in='T_sup_in',
        #                                                **kwargs
        #                                                ))
        #                        # get T_out_id (id only needed for convenience
        #                        # of easy lookup by user)
        #                        (self.HX_act_side_T_out_id
        #                         ) = (self.__check_args('HX_part_name',
        #                                                'HX_act_side_T_out',
        #                                                caller='init_controller',
        #                                                HX_act_side_T_out='T_sup_out',
        #                                                **kwargs
        #                                                )[1])
        #                        # construct memoryviews:
        #                        # (sup side in HX-T-array is the first column with
        #                        # index 0, T_in is the first cell (index 0) in this
        #                        # column, T_out the last (index 1))
        #                        self.HX_act_side_T_in = self._models.parts[self.HX].T[0:1, 0]
        #                        self.HX_act_side_T_out = (self._models.
        #                                                  parts[self.HX].T[1:2, 0])
        #                    else:
        #                        # now the same if the dmd side was chosen:
        #                        # get part and T_in_id (id only needed for convenience
        #                        # of easy lookup by user)
        #                        (self.HX,
        #                         self.HX_act_side_T_in_id
        #                         ) = (self.__check_args('HX_part_name',
        #                                                'HX_act_side_T_in',
        #                                                caller='init_controller',
        #                                                HX_act_side_T_in='T_dmd_in',
        #                                                **kwargs
        #                                                ))
        #                        # get T_out_id (id only needed for convenience
        #                        # of easy lookup by user)
        #                        (self.HX_act_side_T_out_id
        #                         ) = (self.__check_args('HX_part_name',
        #                                                'HX_act_side_T_out',
        #                                                caller='init_controller',
        #                                                HX_act_side_T_out='T_dmd_out',
        #                                                **kwargs
        #                                                )[1])
        #                        # construct memoryviews:
        #                        # (dmd side in HX-T-array is the second column with
        #                        # index 1, T_in is the LAST cell (index 1) in this
        #                        # column, T_out the FIRST (index 0). vice versa of sup)
        #                        self.HX_act_side_T_in = self._models.parts[self.HX].T[1:2, 1]
        #                        self.HX_act_side_T_out = (self._models.
        #                                                  parts[self.HX].T[0:1, 1])

        #            # 3-WAY-VALVE MIXING TYPE ALGORITHM INITIALIZATION:
        #            if 200 < self.ctrl_alg < 210:
        #                err_str = ('The control algorithm for 3-way-mixing valves '
        #                           'requires the information where to find the values '
        #                           'of the two inlet temperatures for the inlet '
        #                           'ports to be passed to `init_controller()`. These '
        #                           'have to be passed as:\n'
        #                           '\'port_T_in_A = port name or cell index\' '
        #                           'for port in1/A and '
        #                           '\'port_T_in_B = port name or cell index\', '
        #                           'for port in2/B, with the port name as a string or '
        #                           'the cell index as an integer index.\n'
        #                           'If both are set to \'none\', the values at the '
        #                           'connected ports are automatically chosen. IF THE '
        #                           'CONNECTED PART IS A NON-NUMERIC PART LIKE A '
        #                           '3W-VALVE OR A PUMP, THIS WILL NOT WORK!\n'
        #                           'If not \'none\', the connected parts also have '
        #                           'to be passed with:\n'
        #                           '\'part_T_in_A = part name\' and'
        #                           '\'part_T_in_B = part name\'.')
        #                # check if given:
        #                assert ('port_T_in_A' in kwargs and
        #                        'port_T_in_B' in kwargs), err_str
        #                # check if none
        #                if kwargs['port_T_in_A'] == 'none':
        #                    # if none, get connected port ID and make memory view to
        #                    # connected port value
        #                    # get connected part and port:
        #                    (conn_part,
        #                     conn_port) = ((self._models.
        #                                    port_links[self.actuator + ';' +
        #                                               'in1']).split(';'))
        #                    # check if part is numeric, and raise error if not:
        #                    if not self._models.parts[conn_part].solve_numeric:
        #                        raise TypeError('A non-numeric part can\'t be used '
        #                                        'as port_T_in_A in `init_controller()`'
        #                                        ' for actuator ' + self.actuator + '!')
        #                    # get index of connected port:
        #                    conn_port_idx = (
        #                            self._models.parts[conn_part]._port_own_idx[
        #                                    self._models.parts[conn_part].port_names.index(
        #                                            conn_port)])
        #                    # get memoryview to that part and port:
        #                    self.T_in_A = (
        #                            self._models.parts[conn_part].T[
        #                                    conn_port_idx:conn_port_idx + 1])
        #                else:
        #                    # check if all is given and calculate indexes:
        #                    self._part_T_in_A, self._port_T_in_A = self.__check_args(
        #                            'part_T_in_A', 'port_T_in_A', 'init_controller',
        #                            **kwargs)
        #                    # create memoryview to the value at port_T_in_A
        #                    self.T_in_A = (
        #                            self._models.parts[self._part_T_in_A].T[
        #                                    self._port_T_in_A:self._port_T_in_A + 1])
        #                if kwargs['port_T_in_B'] == 'none':
        #                    # if none, get connected port ID and make memory view to
        #                    # connected port value
        #                    # get connected part and port:
        #                    conn_part, conn_port = (
        #                            (self._models.port_links[
        #                                    self.actuator + ';' + 'in2']).split(';'))
        #                    # check if part is numeric, and raise error if not:
        #                    if not self._models.parts[conn_part].solve_numeric:
        #                        raise TypeError('A non-numeric part can\'t be used '
        #                                        'as port_T_in_B in `init_controller()`'
        #                                        ' for actuator ' + self.actuator + '!')
        #                    # get index of connected port:
        #                    conn_port_idx = (
        #                            self._models.parts[conn_part]._port_own_idx[
        #                                    self._models.parts[conn_part].port_names.index(
        #                                            conn_port)])
        #                    # get memoryview to that part and port:
        #                    self.T_in_B = (self._models.
        #                                   parts[conn_part].
        #                                   T[conn_port_idx:conn_port_idx + 1])
        #                else:
        #                    # check if all is given and calculate indexes:
        #                    self._part_T_in_B, self._port_T_in_B = self.__check_args(
        #                            'part_T_in_B', 'port_T_in_B', 'init_controller',
        #                            **kwargs)
        #                    # create memoryview to the value at port_T_in_A
        #                    self.T_in_B = (
        #                            self._models.parts[self._part_T_in_B].T[
        #                                    self._port_T_in_B:self._port_T_in_B + 1])

        #            # 3-WAY-VALVE SPLITTING TYPE ALGORITHM INITIALIZATION:
        #            if 210 < self.ctrl_alg < 220:
        #                # since a splitting valve does not change anything besides
        #                # the massflow in the A/B branches, does it need a control
        #                # algorithm?
        #                err_str = ('No control algorithms for splitting valves have '
        #                           'been specified yet! Search for '
        #                           '`3-WAY-VALVE SPLITTING TYPE ALGORITHM '
        #                           'INITIALIZATION` in Controls class and add needed '
        #                           'parts/ports and add the algorithm in the '
        #                           '´ctrl_algorithm()´ method.')
        #                raise TypeError(err_str)
        #                """
        #                TO DO
        #                HEATGENERATOR/EXCHANGER ALGORITHM INITIALIZATION already takes
        #                the needed additional input for temperatures/memviews for
        #                ctrl algs of spl valves for heatgen/hex
        #                """
        #                addstuffhere
        #                Models

        # set initialized
        self.initialized = True

    def clear(self):
        """
        Resets all initial values of the controller
        """

        # general values:
        self.cv = 0.0
        self.prev_cv = 0.0
        self.error = 0.0
        self.prev_error = 0.0
        # defaults for low pass filtering derivate error
        self._prev_deriv = 0.0
        self._deriv_err = 0.0
        self._deriv = 0.0

        # values for discrete controllers:
        if self.discrete_time:
            self.prev2_error = 0.0
        # values for conti time controllers:
        else:
            self.i_term = 0.0
            self.prev_i_term = 0.0
            self.error_sum = 0.0
            self.d_term = 0.0

    def ctrl(self, timestep):
        # only run ctrl if not linked/copied actuator:
        if self._process_cv_mode != 'linked':
            # get previous step's CV for slope calculation:
            self._prev_cv = self.cv
            # check if not a steady state controller:
            if not self._steady_state:
                # check if discrete time domain controller
                if self.discrete_time:
                    # if discrete time domain controller
                    # save last error (prev_error) and error from two timesteps
                    # before (prev2_error) for discrete transient state PID
                    # controller. This means saving errors of current step and
                    # the two steps before by shifting them:
                    self.prev2_error = self.prev_error
                    self.prev_error = self.error
                else:
                    # if coninuous time domain controller
                    # save last error and last I-term for transient state PID
                    # controller:
                    self.prev_error = self.error
                    self.prev_i_term = self.i_term
            # if ref port is specified get SP value from ref port with diff:
            if self.use_ref:
                self.sp = self._ref + self._ref_diff
            # check if deadtime != 0 (also works for np.timedelta64 values)
            if self.deadtime:
                # before extraction of pv add up timestep (time expired since
                # last call of controller) to deadtime array:
                (
                    self.error,
                    self.delayed_pv,
                    self.dt_idx,
                    self.last_dt_idx,
                ) = ctrl_deadtime(
                    self.deadtime,
                    timestep,
                    self.dt_arr,
                    self.pv_arr,
                    self.len_dt_arr,
                    self.dt_idx,
                    self.last_dt_idx,
                    self.delayed_pv,
                    self.sp,
                    self.pv,
                )
                print('deadtime used! check i-term/err_sum use!')
            # if no deadtime was chosen:
            else:
                # get current timestep error
                # (error = setpoint - process_variable):
                self.error = self.sp[0] - self.pv[0]
            # call controller function:
            self.run_controller(timestep)

            # limit change of CV to slope:
            if self._slope:
                # get current slope
                current_slope = (self.cv - self._prev_cv) / timestep
                self.cv = (  # limit change of cv to max. slope
                    self.cv
                    if self._slope_n <= current_slope <= self._slope_p
                    else self._prev_cv + self._slope_p * timestep
                    if (current_slope > self._slope_p)
                    else self._prev_cv + self._slope_n * timestep
                )

            self._process_cv_fun(self)

    def _process_cv_direct(self, dummy_cls_instance):
        # limit cv to hardware (part-specific limits) and set to actuator
        self._act_cv[:] = (
            self.cv
            if self._llim <= self.cv <= self._ulim
            else self._ulim
            if self.cv > self._ulim
            else self._llim
        )

    def _clip_cv_to_saturation(self):
        self.cv = (
            self._cv_sat_min
            if self.cv < self._cv_sat_min
            else self._cv_sat_max
            if self.cv > self._cv_sat_max
            else self.cv
        )

    def _cv_invert(self):
        # allow direction inversion, if specified
        self.cv *= self._invert

    def _subcontroller_check(self, pid=False):
        # check if controller is depending on another controller:
        if self.sub_ctrl:  # if yes check for kind of dependency:
            if self._concurrent:  # if concurrent with master controller:
                if self._master_act_cv[0] == self._master_act_offst:
                    # if master actuator cv at off_state -> deactivated
                    self.cv = self._off_state  # also set own cv to off state
                    if pid:  # for PID controllers
                        # also erase i-term since no CV means no error:
                        # (derivs and prev err don't matter, since they only
                        # accumulate for up to two steps)
                        self.i_term = 0.0
            else:  # if sequential with master controller:
                if self._master_act_cv[0] != self._master_act_offst:
                    # if master actuator cv NOT at off_state -> activated
                    self.cv = self._off_state  # also set own cv to off state
                    if pid:  # for PID controllers
                        # also erase i-term since no CV means no error:
                        self.i_term = 0.0

    def __check_args(self, part, port='no_port', caller='not_given', **kwargs):
        """
        Check for the existence of the given part and/or port.

        Parameters:
        -----------
        part : string
            Checks if `part`, given as the string name, is passed to **kwargs
            and if the port passed to **kwargs is existing in the simulation
            environmnent.
        port : string
            Checks if `port`, given as the string name of the port to add to
            the controller, is passed to **kwargs and if the port passed to
            **kwargs is existing at `part` in the simulation environment.
            For the default 'no_port', this function will only check for `part`
            and only return the part.
        caller : string
            Determines the caller function, from which this function was called
            to let the user of the simulation environment know, at which
            function an error occurred. Default is 'not_given', but this option
            should only be used for testing!
        **kwargs
            **kwargs can either be passed on from the calling function if the
            arguments have to calling function have to be checked or can be
            specified manually. If specified manually, the kwargs for the
            part and port to be checked have to have identical names with the
            part and port, for example: `partname = part-to-check`.
            The port can be given as string or integer. If port is given as
            string, only named ports can be chosen. If port is given as
            integer, all cells of `part` can be chosen.


        Returns:
        --------
        string
            Returns the string name of `part` if no exception occurred.
        integer
            Returns the integer cell index of `port` at `part` if no exception
            occurred.

        """
        base_err = (  # define leading base error message
            'While adding controller `{0}` to the simulation '
            'environment, the following error occurred:\n'
        ).format(self.name)

        # get calling function name and save it in a printable version:
        #        if caller == 'init_controller':
        #            caller = '`init_controller()`'
        #        elif caller == 'add_control':
        #            caller = '`add_control()`'
        #        elif caller == 'not_given':
        #            caller = '!caller not given!'
        #        else:
        #            raise NameError('The given caller function does not exist!')
        caller = '`add_control()`'  # init done while adding!

        # CHECKING PART:
        # assert that part is given:
        #        err_str = (
        #           'The part ' + part + ' has to be passed to ' + caller +
        #           ' for actuator ' + self.actuator + ' and control algorithm '
        #           + self.ctrl_algs[self.ctrl_alg] + '!')
        err_str = (
            'The part `{0}` has to be given to {1} for actuator {2}.'
        ).format(part, caller, self.actuator)
        assert part in kwargs, base_err + err_str
        # save given part:
        given_part = kwargs[part]
        # assert that given part exists:
        err_str = (
            'The part `{0}` passed to {1} as `{2}` for actuator `{3}` '
            'does not exist. The following parts have been '
            'defined:\n{4}'
        ).format(
            given_part,
            caller,
            part,
            self.actuator,
            repr(list(self._models.parts)),
        )
        assert given_part in self._models.parts, base_err + err_str

        # for the case that the given port is 'no_port' (for example when
        # linking actuators), this function ends here:
        if port == 'no_port':
            return given_part

        # CHECKING PORT
        # save given port
        given_port = kwargs[port]
        # check if port given as string or int:
        if type(given_port) == str:
            # catch error for not existing ports:
            try:
                given_port = self._models.parts[  # get int port index
                    given_part
                ]._port_own_idx[
                    self._models.parts[given_part].port_names.index(given_port)
                ]
            except ValueError:
                err_str = (
                    base_err
                    + 'The port `'
                    + given_port
                    + '` at part `'
                    + given_part
                    + '` passed to '
                    + caller
                    + ' as '
                    + port
                    + ' for actuator `'
                    + self.actuator
                    + '` of '
                    'controller `' + self.name + '` does not exist!'
                    'Either pass a cell index in the range of '
                    '0 <= index < '
                    + str(self._models.parts[given_part].T.shape[0])
                    + ' or one of the following ports:\n'
                    + str(self._models.parts[given_part].port_names)
                )
                raise ValueError(err_str)
        # if port given as id, catch error for not existing ports and if no
        # error return at the end of the function:
        elif type(given_port) == int:
            err_str = (
                base_err + 'The port index ' + str(given_port) + ' at '
                'part `'
                + given_part
                + '` passed to '
                + caller
                + ' as '
                + port
                + ' for actuator `'
                + self.actuator
                + '` is out of '
                'bounds for axis 0 with size '
                + str(self._models.parts[given_part].T.shape[0])
                + '.'
            )
            assert (  # assert that index is in range (pos. and negative!)
                0 <= given_port < self._models.parts[given_part].T.shape[0]
                or -self._models.parts[given_part].T.shape[0] <= given_port < 0
            ), err_str
            given_port = (  # calculate pos. index if negative, else take pos.
                given_port
                if given_port > -1
                else self._models.parts[given_part].T.shape[0] + given_port
            )
        elif type(given_port) != int:
            err_str = (
                base_err
                + 'The port '
                + given_port
                + ' at part '
                + given_part
                + ' passed to '
                + caller
                + ' as '
                + port
                + ' for actuator '
                + self.actuator
                + ' has to be passed as string or integer '
                'value!'
            )
            raise TypeError(base_err + err_str)

        # return given_part and given_port
        return given_part, given_port
