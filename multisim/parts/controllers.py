# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Jun 2018
"""

import numpy as np
import pandas as pd
import scipy.optimize as _sopt

from ..controls import Controls
from .. import precomp_funs as _pf


class PID(Controls):
    def __init__(self, name, master_cls, **kwargs):
        # initialize general controller functions:
        super().__init__(name, master_cls, **kwargs)

        # assert correct controller type
        terms = ['P', 'I', 'PI', 'PD', 'PID']  # , 'two-position-controler'
        err_trms = (
            self._base_err
            + self._arg_err.format('terms')
            + 'For a PID-controller the terms, out of which it '
            'is composed, have to be given with `terms=X`, where X is one of '
            'the following:\n' + str(terms)
        )
        assert 'terms' in kwargs and kwargs['terms'] in terms, err_trms
        self.terms = kwargs['terms']

        err_str_awndp = (
            self._base_err
            + self._arg_err
            + 'A PID controller requires the following additional arguments:\n'
            '    - `anti_windup`: Float, int or str value to prevent a windup '
            'of the ingegrative- (I-) term of the controller. For a '
            'continuous time controller the I-term is limited to '
            '`-anti_windup <= I-term <= anti_windup`, for a discrete time '
            'controller the CV value of the last step is limited to '
            '`-anti_windup <= last_CV <= anti_windup`.\n'
            'If set to \'auto_X\' with X being a float or integer, anti '
            'windup will be automatically scaled by Ki so that the term '
            '`-X <= i_term * ki <= X` will be satisfied. If X is omitted, '
            'the default of `X=1` is set.'
        )

        # GET COEFFICIENTS FOR PID (and loop tuning methods):
        # check if controller is P/I/D
        if self.terms in ['P', 'PI', 'PD', 'PID']:
            # assert and get loop tuning (coeff. setting) method:
            methods = ('manual', 'tune', 'ziegler-nichols')
            err_str = (
                self._base_err + self._arg_err.format('loop_tuning') + '\n'
                'If a controller of the kinds P/I and/or D '
                'is chosen, the loop tuning method for how to set the '
                'K-coefficients has to be given with `loop_tuning=X`.\n'
                '    `loop_tuning=\'manual\'`: Coefficients have to be given '
                'with `Kp=P, Ki=I, Kd=D`.\n'
                '    `loop_tuning=\'tune\'`: Tuning mode for Ziegler-Nichols '
                'to find a set of `Kp_crit` and `t_crit` values. In this '
                'mode, the controller will be treated as a P-controller, '
                'regardless of the set terms. Slopes should be set to values '
                'representing the part\'s behaviour. Slopes additionally '
                'restricted by the controller should be set AFTER tuning.\n'
                'This mode is only to be used when setting up the controller '
                'parameters for a simulation. Switch to \'manual\' or '
                '\'ziegler-nichoks\' after setup.'
                '    `loop_tuning=\'ziegler-nichols\'`: Adjust K-coefficients '
                'automatically, depending on the critical Kp value (above '
                'which oscillations of the CV/PV occur) and the oscillation '
                'period. More information will be provided when chosing '
                'Ziegler-Nichols loop tuning.\n\n'
                'If the chosen loop tuning method turns out to have '
                'small instabilities/oscillations, a multiplicative '
                'factor can be given for all K-coefficients. This is '
                'especially helpful to reduce the derivative term, '
                'since it tends to oscillate in time discrete systems '
                'with small timesteps.\n'
                'The multiplication factors must be given with '
                '`K_mult=[Kp_mult, Ki_mult, Kd_mult]`. To for example '
                'leave Kp and Ki unchanged while halving Kd, pass '
                '`K_mult=[1, 1, 0.5]`.'
            )
            assert 'loop_tuning' in kwargs, err_str
            assert kwargs['loop_tuning'] in methods, err_str
            # save method:
            self.loop_tuning = kwargs['loop_tuning']

            # rules list:
            rules = ['classic', 'pessen-int', 'some-os', 'no-os']

            # set all coefficients to 0:
            self.kp = 0.0
            self.ki = 0.0
            self.kd = 0.0
            # set all multiplicators to 1:
            self.kp_mult = 1.0
            self.ki_mult = 1.0
            self.kd_mult = 1.0
            # do the loop tuning:
            if self.loop_tuning == 'manual':
                # check for kind and respective K-coefficient
                err_str = (  # base error for all coefficients to format in err
                    self._base_err + self._arg_err.format('K{0}') + '\n'
                    '`K{0}` has to be given as integer or '
                    'float value for a controller of type `'
                    + self.terms
                    + '`.'
                )
                if 'P' in self.terms:
                    assert 'Kp' in kwargs and isinstance(
                        kwargs['Kp'], (int, float)
                    ), err_str.format('p')
                    self.kp = float(kwargs['Kp'])
                if 'I' in self.terms:
                    assert 'Ki' in kwargs and isinstance(
                        kwargs['Ki'], (int, float)
                    ), err_str.format('i')
                    self.ki = float(kwargs['Ki'])
                if 'D' in self.terms:
                    assert 'Kd' in kwargs and isinstance(
                        kwargs['Kd'], (int, float)
                    ), err_str.format('d')
                    self.kd = float(kwargs['Kd'])
            elif self.loop_tuning == 'tune':
                # check for kind and respective K-coefficient
                errtune = (  # base error for all coefficients to format in err
                    self._base_err + self._arg_err.format('Kp') + '\n'
                    '`Kp` has to be given as integer or '
                    'float value for a controller of type `'
                    + self.terms
                    + '`. during loop tuning mode.'
                )
                assert 'Kp' in kwargs and isinstance(
                    kwargs['Kp'], (int, float)
                ), errtune
                self.kp = float(kwargs['Kp'])
                self.ki = 0.0  # only use P controller
                self.kd = 0.0
                kwargs['anti_windup'] = np.inf  # overwrite
                print(
                    '\n\nLoop tuning mode selected for controller `{0}`.\n'
                    'Ki, Kd and anti-windup are deactivated. Make sure that '
                    'the actuator limits (and if applicable: max. scaling) '
                    'are set correctly. Don\'t forget to deactivate loop '
                    'tuning mode when done.'.format(self.name)
                )
            elif self.loop_tuning == 'ziegler-nichols':
                # assert that all parameters are given and of the correct type:
                err_str_crit = (
                    self._base_err
                    + self._arg_err.format('Kp_crit, T_crit')
                    + 'The loop tuning '
                    'method \'ziegler-nichols\' has been selected. This '
                    'method requires the critical proportional gain '
                    '`Kp_crit=X`, which is the lowest proportional gain '
                    'at which the output of the control loop has stable '
                    'and consistent oscillations after initiating a jump of'
                    'the process variable, as well as the '
                    'oscillation period `T_crit=Y` in [s].\n'
                    'To get these values, make this controller a '
                    'P-controller (set `Ki` and/or `Kd` to zero), set '
                    '`loop_tuning=\'manual\'`, initiate a relevant change '
                    'of the process variable and slowly increase the `Kp` '
                    'value until stable and consistent oscillations of '
                    'the process variable occur. Now set this `Kp` value '
                    'to `Kp_crit` and the oscillation period to `T_crit` '
                    'with `loop_tuning=\'ziegler-nichols\'`.\n`Ki` and '
                    '`Kd` must not be passed.'
                )
                err_str_rules = (
                    self._base_err
                    + self._arg_err.format('rule')
                    + 'For a PID controller, which is loop tuned with '
                    'Ziegler-Nichols, an additional coefficient '
                    'calculation rule has to be set with `rule=X`.\n'
                    'The following rules are supported:\n'
                    '    - \'classic\': Classic aggressive Ziegler-Nichols '
                    'tuning with high transient overshoot.\n'
                    '    - \'pessen-int\': Pessen integral rule. Slightly '
                    'higher proportional and derivate coefficients, but '
                    'lower integral coefficient.\n'
                    '    - \'some-os\': Low overshoot rule. Proportional '
                    'coefficient nearly halved, derivate coefficient 2.64 '
                    'times higher, integral coefficient unchanged.\n'
                    '    - \'no-os\': No overshoot rule. Proportional '
                    'coefficient set to a third, derivate coefficient '
                    '2.64 times higher, integral coefficient unchanged.'
                )

                if (
                    self.terms == 'PID'
                    and 'rule' in kwargs  # just raise err_str_rules if it
                    and kwargs['rule']  # is a PID and something not in rules
                    not in rules
                ):  # was passed
                    assert kwargs['rule'] in rules, err_str_rules

                # crit. values always need to be passed, so assert correctness:
                assert 'Kp_crit' in kwargs and isinstance(
                    kwargs['Kp_crit'], (int, float)
                ), err_str_crit
                assert 'T_crit' in kwargs and isinstance(
                    kwargs['T_crit'], (int, float)
                ), err_str_crit
                if 'I' not in self.terms:
                    assert 'Ki' not in kwargs, err_str_crit
                if 'D' not in self.terms:
                    assert 'Kd' not in kwargs, err_str_crit

                # save parameters:
                self.kp_crit = float(kwargs['Kp_crit'])
                self.t_crit = float(kwargs['T_crit'])

                # calculate coefficients depending on the controller type:
                if self.terms == 'P':
                    # if P controller
                    self.kp = self.kp_crit * 0.5
                elif self.terms == 'PI':
                    # if PI controller
                    self.kp = self.kp_crit * 0.45
                    self.t_n = self.t_crit * 0.85  # integration time constant
                    self.ki = self.kp / self.t_n
                elif self.terms == 'PD':
                    # if PD controller
                    self.kp = self.kp_crit * 0.55
                    self.t_v = self.t_crit * 0.15  # derivate time constant
                    self.kd = self.kp * self.t_v
                elif self.terms == 'PID':
                    # if PID controller
                    # assert that additional rule is given:
                    assert (
                        'rule' in kwargs and kwargs['rule'] in rules
                    ), err_str_rules
                    # save rule
                    self.loop_tuning_rule = kwargs['rule']
                    # set coefficients depending on rule:
                    if self.loop_tuning_rule == 'classic':
                        # classic aggressive ZN method:
                        self.kp = self.kp_crit * 0.6
                        self.t_n = self.t_crit * 0.5  # integration time const.
                        self.t_v = self.t_crit * 0.125  # derivate time const.
                        self.ki = self.kp / self.t_n
                        self.kd = self.kp * self.t_v
                    elif self.loop_tuning_rule == 'pessen-int':
                        # pessen integral method:
                        self.kp = self.kp_crit * 0.7
                        self.t_n = self.t_crit * 0.4  # integration time const.
                        self.t_v = self.t_crit * 0.15  # derivate time const.
                        self.ki = self.kp / self.t_n
                        self.kd = self.kp * self.t_v
                    elif self.loop_tuning_rule == 'some-os':
                        # some overshoot method:
                        self.kp = self.kp_crit * 0.33
                        self.t_n = self.t_crit * 0.5  # integration time const.
                        self.t_v = self.t_crit * 0.33  # derivate time const.
                        self.ki = self.kp / self.t_n
                        self.kd = self.kp * self.t_v
                    elif self.loop_tuning_rule == 'no-os':
                        # no overshoot method:
                        self.kp = self.kp_crit * 0.2
                        self.t_n = self.t_crit * 0.5  # integration time const.
                        self.t_v = self.t_crit * 0.33  # derivate time const.
                        self.ki = self.kp / self.t_n
                        self.kd = self.kp * self.t_v
        # get coefficient multipliers:
        err_str = (
            self._base_err
            + self._arg_err.format('K_mult')
            + 'If coefficient multipliers are passed to the '
            'controller with `K_mult=X`, X has to be a list with three '
            'elements of type integer or float.'
        )
        if 'K_mult' in kwargs:
            # assert:
            assert (
                type(kwargs['K_mult']) == list and len(kwargs['K_mult']) == 3
            ), err_str
            assert (
                isinstance(kwargs['K_mult'][0], (int, float))
                and isinstance(kwargs['K_mult'][1], (int, float))
                and isinstance(kwargs['K_mult'][2], (int, float))
            ), err_str
            # save multiplicators:
            self.kp_mult = float(kwargs['K_mult'][0])
            self.ki_mult = float(kwargs['K_mult'][1])
            self.kd_mult = float(kwargs['K_mult'][2])
            # and multiplicate values:
            self.kp *= self.kp_mult
            self.ki *= self.ki_mult
            self.kd *= self.kd_mult

        # ADAPTIVE COEFFICIENTS:
        # assert that adapt_coefficients is given and a bool value:
        err_str = (
            self._base_err
            + self._arg_err
            + '`adapt_coefficients=True/False` has to be given.\n'
            'If set to True, the PID controller coefficients (Kp, Ki, Kd) '
            'will be automatically adapted to the current simulation '
            'timestep to avoid controller instabilities. When using adaptive '
            'timesteps in the simulation, this is mandatory. To use this '
            'functionality, the norm timestep `norm_timestep=X`, at which the '
            'coefficients were designed, must be given.\n'
            'If set to False, the controller will always use the same '
            'coefficients regardless of the timestep.'
        )
        assert 'adapt_coefficients' in kwargs, err_str.format(
            'adapt_coefficients'
        )
        assert type(kwargs['adapt_coefficients']) == bool, err_str.format(
            'adapt_coefficients'
        )
        # save state:
        self.adapt_coeff = kwargs['adapt_coefficients']
        # assert that it is True if adaptive timesteps are being used:
        if self._models.adaptive_steps:
            assert self.adapt_coeff, err_str.format('adapt_coefficients')
        # check for given parameters if True:
        if self.adapt_coeff:
            print(
                'adapt_coefficients is deprecated due to problems with '
                'stability. Even if set to True, coefficients will not be '
                'adapated to the timestep.'
            )
            # assert that norm timestep is given and of correct type:
            assert 'norm_timestep' in kwargs, err_str.format('norm_timestep')
            err_str = (
                self._base_err
                + self._arg_err.format('norm_timestep')
                + '`norm_timestep` must be given as integer or float value.'
            )
            assert isinstance(kwargs['norm_timestep'], (int, float)), err_str
            # save norm timestep:
            self.norm_timestep = kwargs['norm_timestep']
            # save standard k-coefficients as norm coefficients:
            self.norm_kp = self.kp
            self.norm_ki = self.ki
            self.norm_kd = self.kd

        # DERIVATIVE TERM LOW PASS FILTERING
        # assert that info about filtering is given if there is a derivative
        # term
        if 'D' in self.terms:
            # assert if given and type:
            err_str = (
                self._base_err
                + self._arg_err
                + 'The controller contains a derivative term, '
                'thus `filter_derivative=bool` has to be given.\n'
                'If it is set to `True`, the derivative term will be '
                'low pass filtered with the additionally required '
                'cutoff frequency `cutoff_freq=X`, where X is the '
                'frequency in [Hz]. The filtering method is a '
                'first order Butterworth filter.\n'
                'If it is set to `False`, no filtering will be applied '
                'to the derivative term. Oscillations of the '
                'controller are likely, especially when the simulation '
                'timestep is small.'
            )
            assert 'filter_derivative' in kwargs, err_str.format(
                'filter_derivative'
            )
            assert type(kwargs['filter_derivative']) == bool, err_str.format(
                'filter_derivative'
            )
            # if filter is True:
            if kwargs['filter_derivative']:
                # assert that cutoff frequency is given:
                assert 'cutoff_freq' in kwargs, err_str.format('cutoff_freq')
                assert isinstance(
                    kwargs['cutoff_freq'], (int, float)
                ), err_str.format('cutoff_freq')
                # save states:
                self.filt_derivate = True
                self._bw_cutoff = float(kwargs['cutoff_freq'])
                # calculate RC value:
                self._RC = 1 / (2 * np.pi * self._bw_cutoff)
                # calculate alpha starting value:
                self._bw_alpha = self._models.timestep / (
                    self._RC + self._models.timestep
                )
            else:
                self.filt_derivate = False
        else:
            self.filt_derivate = False

        # set anti-windup factor:
        if 'I' in self.terms:
            assert 'anti_windup' in kwargs and isinstance(
                kwargs['anti_windup'], (int, float, str)
            ), err_str_awndp.format('anti_windup')
            if isinstance(kwargs['anti_windup'], (int, float)):
                self._anti_windup = float(kwargs['anti_windup'])
            else:
                awndp = (
                    float(  # get value of auto scaling or set to 1 if none
                        kwargs['anti_windup'].split('_')[1]
                    )
                    if '_' in kwargs['anti_windup']
                    else 1.0
                )
                self._anti_windup = awndp / self.ki
            assert self._anti_windup > 0.0, (
                self._base_err
                + self._arg_err.format('anti_windup')
                + 'Anti windup must be > 0. To disable anti windup, set to '
                'np.inf.'
            )
        else:
            self._anti_windup = np.inf

    def _reset_to_init_cond(self):
        self.cv = 0.0
        self._prev_cv = 0.0
        self._deriv = 0.0
        self.error = 0.0
        self.prev_error = 0.0
        self.prev2_error = 0.0
        self.prev_i_term = 0.0

    def run_controller(self, timestep):
        # check if k-coefficients have to be recalculated due to using adaptive
        # timesteps:
        if self.adapt_coeff:
            # get factor to multiply coefficients with:
            self.coeff_factor = timestep / self.norm_timestep
            self.coeff_factor = 1
            # multiply coefficients with the factor to get timestep adjusted
            # coefficients:
            self.kp = self.norm_kp * self.coeff_factor
            self.ki = self.norm_ki * self.coeff_factor ** 2
            self.kd = self.norm_kd * self.coeff_factor ** 2

        # check if discrete or continuous time controller:
        if self.discrete_time:
            # if integral term is in the controller, the velocity algorithm is
            # used:
            if self.ki != 0:
                # get and limit previous CV by anti windup:
                self._prev_cv_lim = (
                    -self._anti_windup
                    if self.cv < -self._anti_windup
                    else self._anti_windup
                    if self.cv > self._anti_windup
                    else self.cv
                )
                # if derivate term is to be filtered, use butterworth first
                # order low pass to filter it:
                if self.filt_derivate:
                    # save last derivate value
                    self._prev_deriv = self._deriv
                    # get unfiltered new derivate error term:
                    self._deriv_err = (
                        self.error - 2 * self.prev_error + self.prev2_error
                    )
                    # get new alpha coefficient for derivative filtering:
                    self._bw_alpha = timestep / (self._RC + timestep)
                    # filter derivate value
                    self._deriv = self._prev_deriv + self._bw_alpha * (
                        self._deriv_err - self._prev_deriv
                    )
                else:
                    # else if no filtering, just get derivate error term:
                    self._deriv = (
                        self.error - 2 * self.prev_error + self.prev2_error
                    )
                # calculate new control variable with velocity algorithm:
                self.cv = (
                    self._prev_cv_lim
                    + self.kp * (self.error - self.prev_error)
                    + self.ki * timestep * self.error
                    + self.kd / timestep * self._deriv
                )
            else:
                # else if no integral term, use standard P/PD algorithm
                # if derivate term is to be filtered, use butterworth first
                # order low pass to filter it:
                if self.filt_derivate:
                    # save last derivate value
                    self._prev_deriv = self._deriv
                    # get unfiltered new derivate error term:
                    self._deriv_err = self.error - self.prev_error
                    # get new alpha coefficient for derivative filtering:
                    self._bw_alpha = timestep / (self._RC + timestep)
                    # filter derivate value
                    self._deriv = self._prev_deriv + self._bw_alpha * (
                        self._deriv_err - self._prev_deriv
                    )
                else:
                    # else if no filtering, just get derivate error term:
                    self._deriv = self.error - self.prev_error
                # if no integral term, use standard discrete P/PD algorithm
                # with euler backwards differentiation for derivate term:
                self.cv = (
                    self.kp * self.error + self.kd / timestep * self._deriv
                )
            # clip cv value to saturation:
            self._clip_cv_to_saturation()
        else:  # calculate controller output for a continuous time system
            # calculate I-term:
            self.i_term = timestep * self.error + self.prev_i_term
            #        self.i_term = timestep * self.error_sum
            # clip value to anti_windup:
            self.i_term = (
                -self._anti_windup
                if self.i_term < -self._anti_windup
                else self._anti_windup
                if self.i_term > self._anti_windup
                else self.i_term
            )
            # if derivate term is to be filtered, use butterworth first
            # order low pass to filter it:
            if self.filt_derivate:
                # save last derivate value
                self._prev_deriv = self._deriv
                # get unfiltered new derivate error term:
                self._deriv_err = self.error - self.prev_error
                # get new alpha coefficient for derivative filtering:
                self._bw_alpha = timestep / (self._RC + timestep)
                # filter derivate value
                self._deriv = self._prev_deriv + self._bw_alpha * (
                    self._deriv_err - self._prev_deriv
                )
            else:
                # else if no filtering, just get derivate error term:
                self._deriv = self.error - self.prev_error
            # calculate D-term:
            self.d_term = self._deriv / timestep
            # calculate control variable:
            self.cv = (
                self.kp * self.error
                + self.ki * self.i_term  # allow direction inversion:
                + self.kd * self.d_term
            ) * self._invert
            # clip value to max/min:
            self._clip_cv_to_saturation()

        # check if controller is depending on another controller:
        self._subcontroller_check(pid=True)


class BangBang(Controls):
    def __init__(self, name, master_cls, **kwargs):
        # initialize general controller functions:
        super().__init__(name, master_cls, **kwargs)

        # get hysteresis
        err_str = (
            self._base_err
            + self._arg_err.format('hysteresis')
            + 'A hysteresis must be passed to a bang-bang controller with '
            '`hysteresis=X`, where X is an integer or float value. The '
            'controller will activate the actuator as soon as the '
            'process variable (PV) is lower than the setpoint (SP) minus half '
            'the hysteresis (`PV < SP - hysteresis / 2`) and deactivate the '
            'actuator when the PV is higher than the SP plus half the '
            'hysteresis (`PV > SP + hysteresis / 2`).\n'
            'If a negative hysteresis is given, this behavior will be '
            'reversed. The controller will activate the actuator when '
            '`PV > SP + abs(hysteresis) / 2` and deactivate the controller '
            'when `PV < SP - abs(hysteresis) / 2`.'
        )
        assert 'hysteresis' in kwargs and isinstance(
            kwargs['hysteresis'], (int, float)
        ), err_str
        if kwargs['hysteresis'] >= 0.0:  # check for reversed behavior
            self._reversed = False
        else:
            self._reversed = True
        self._hysteresis = float(abs(kwargs['hysteresis']))  # save pos. hyst.

    def _reset_to_init_cond(self):
        self.cv = 0.0
        self._prev_cv = 0.0
        self._deriv = 0.0
        self.error = 0.0
        self.prev_error = 0.0
        self.prev2_error = 0.0
        self.prev_i_term = 0.0

    def run_controller(self, timestep):
        if self.pv < (self.sp - self._hysteresis / 2):
            if self._reversed:  # reversed action
                self.cv = self._off_state  # switch off (to off state)
            else:
                self.cv = self._cv_sat_max  # switch on (to upper sat lim.)
        elif self.pv > (self.sp + self._hysteresis / 2):
            if self._reversed:  # reversed action
                self.cv = self._cv_sat_max  # switch on (to upper sat lim.)
            else:
                self.cv = self._off_state  # switch off (to off state)

        # allow direction inversion:
        self._cv_invert()

        # check if controller is depending on another controller:
        self._subcontroller_check()


class TwoSensors(Controls):
    def __init__(self, name, master_cls, **kwargs):
        # initialize general controller functions:
        super().__init__(name, master_cls, **kwargs)

        # check if slope was initialized and print warning if true
        if self._slope:
            err_slope = (
                '\nA slope was set for two sensor controller `{0}`. Slopes '
                'with discrete step (0-1) controllers such as Bang-Bang- or '
                'two sensor controllers may cause problems when used as '
                'sub- or master-controllers, f.i. a two sensor controller '
                'for a CHP plant and a PID-sub-controller for the pump.\n'
                'Make sure that controllers do not interfer and set '
                '`silence_slope_warning=True` to controller `{0}` to use a '
                'slope for this controller.\n'.format(name)
            )
            assert (
                'silence_slope_warning' in kwargs
                and kwargs['silence_slope_warning']
            ), err_slope

        # get off sensor and on value (on sensor is PV origin and off value is
        # the SP):
        ers = (
            self._base_err
            + self._arg_err.format('on_sensor_part and/or on_sensor_port')
            + 'switch on sensor part and port must be given'
        )
        assert 'on_sensor_part' in kwargs and 'on_sensor_port' in kwargs, ers
        (
            self._on_sens_part,
            self._on_sens_idx,
        ) = self._Controls__check_args(  # check for existence of part and idx
            'on_sensor_part',
            'on_sensor_port',
            'add_control',
            on_sensor_part=kwargs['on_sensor_part'],
            on_sensor_port=kwargs['on_sensor_port'],
        )
        self.on_sensor_val = self._models.parts[self._on_sens_part].T[
            self._on_sens_idx : self._on_sens_idx + 1 : 1
        ]

        erac = (
            self._base_err
            + self._arg_err.format(
                'activation_value and/or '
                'activation_sign and/or deactivation_sign'
            )
            + 'activation value for on sensor and inequality sign.'
            '`lower` for act. means activation when current value is lower than '
            'act. value. Vice versa for `greater`.\n'
            '`lower` for deact. means deactivation when current value of '
            'controlled part/port is lower than setpoint. Vice versa for'
            '`greater`.'
        )
        assert (
            'activation_value' in kwargs
            and 'activation_sign' in kwargs
            and 'deactivation_sign' in kwargs
        ), erac
        assert isinstance(kwargs['activation_value'], (int, float)), erac
        assert isinstance(kwargs['activation_sign'], str), erac
        assert isinstance(kwargs['deactivation_sign'], str), erac

        assert kwargs['activation_sign'] in ('greater', 'lower'), erac
        assert kwargs['deactivation_sign'] in ('greater', 'lower'), erac
        self._act_val = float(kwargs['activation_value'])
        self._act_sgn = 'le' if kwargs['activation_sign'] == 'lower' else 'ge'
        self._deact_sgn = (
            'le' if kwargs['deactivation_sign'] == 'lower' else 'ge'
        )

    def _reset_to_init_cond(self):
        self.cv = 0.0
        self._prev_cv = 0.0
        self._deriv = 0.0
        self.error = 0.0
        self.prev_error = 0.0
        self.prev2_error = 0.0
        self.prev_i_term = 0.0

    def run_controller(self, timestep):
        # controller activation
        if self._act_sgn == 'le':
            if self.on_sensor_val[0] < self._act_val:
                self.cv = self._cv_sat_max
        else:  # if greater
            if self.on_sensor_val[0] > self._act_val:
                self.cv = self._cv_sat_max
        # controller deactivation
        if self._deact_sgn == 'ge':
            if self.pv[0] > self.sp[0]:
                self.cv = self._off_state
        else:  # if lesser
            if self.pv[0] < self.sp[0]:
                self.cv = self._off_state

        # allow direction inversion:
        self._cv_invert()

        # check if controller is depending on another controller:
        self._subcontroller_check()


class ModelPredCHP(Controls):
    """
    Model predictive controller for electricity profile led CHP plant.

    Attributes
    ----------
    name : str
        Unique controller identifier.
    master_cls : class instance
        Instance of the simulation environment to which to add the controller.

    """

    def __init__(self, name, master_cls, **kwds):
        # initialize general controller functions:
        super().__init__(name, master_cls, **kwds)

        # get off sensor and on value (on sensor is PV origin and off value is
        # the SP):
        ers = (
            self._base_err
            + self._arg_err.format('on_sensor_part and/or on_sensor_port')
            + 'emergency switch ON sensor part and port must be given'
        )
        assert 'on_sensor_part' in kwds and 'on_sensor_port' in kwds, ers
        (
            self._on_sens_part,
            self._on_sens_idx,
        ) = self._Controls__check_args(  # check for existence of part and idx
            'on_sensor_part',
            'on_sensor_port',
            'add_control',
            on_sensor_part=kwds['on_sensor_part'],
            on_sensor_port=kwds['on_sensor_port'],
        )
        self.on_sensor_val = self._models.parts[self._on_sens_part].T[
            self._on_sens_idx : self._on_sens_idx + 1 : 1
        ]

        erac = (
            self._base_err
            + self._arg_err.format(
                'activation_value and/or '
                'activation_sign and/or deactivation_sign'
            )
            + 'activation value for emergency on sensor and inequality sign.'
            '`lower` for act. means activation when current value is lower than '
            'act. value. Vice versa for `greater`.\n'
            '`lower` for deact. means deactivation when current value of '
            'controlled part/port is lower than setpoint. Vice versa for'
            '`greater`.'
        )
        assert (
            'activation_value' in kwds
            and 'activation_sign' in kwds
            and 'deactivation_sign' in kwds
        ), erac
        assert isinstance(kwds['activation_value'], (int, float)), erac
        assert isinstance(kwds['activation_sign'], str), erac
        assert isinstance(kwds['deactivation_sign'], str), erac

        assert kwds['activation_sign'] in ('greater', 'lower'), erac
        assert kwds['deactivation_sign'] in ('greater', 'lower'), erac
        self._act_val = float(kwds['activation_value'])
        self._act_sgn = 'le' if kwds['activation_sign'] == 'lower' else 'ge'
        self._deact_sgn = (
            'le' if kwds['deactivation_sign'] == 'lower' else 'ge'
        )

        # check for and set optimization params:
        self._emerg_prms = {  # default parameters
            'use_hysteresis': True,
            'hysteresis': 1.0,
            'full_power_offset': 2.0,
        }
        err_ep = (
            self._base_err
            + self._arg_err.format('emergency_cntrl')
            + '`emergency_cntrl` must be given. Either set to '
            '`emergency_cntrl=\'default\'` to use default parameters '
            '(see below) or pass a dictionary with values to change.\n\n'
            'Parameters:\n'
            '    - `use_hysteresis`: Use a hysteresis to switch off emergency '
            '**activatation** of the CHP plant? F.i. if the setpoint is 65°C, '
            'then the CHP plant will switch off at `65°C+hysteresis`. For '
            'act. sign `greater` this is `65°C-hysteresis`. Only applies to '
            'CHP activation, not deactivation! Must be bool.\n'
            '    - `hysteresis`: Hysteresis to use. Must be int, float >= 0.\n'
            '    - `full_power_offset`: Offset to activation value for '
            'emergency-on-sensor above/below (depending on act. sign) the '
            'CHP plant will go to full power, completely overwriting the '
            'modelpredictive optimization. '
            'This means that, for act. sign `lower`, the CHP plant will go '
            'into \'mixed emergency mode\' when the PV is '
            '`(act.val. - full_power_offset) <= PV <= act.val.` and into '
            'full power emergency mode for '
            '`PV < (act.val. - full_power_offset)`.'
            'For act. sign `greater`, full power mode is active when '
            '`PV > (act.val. + full_power_offset)`. '
            '**Mind the changed SIGN!**.'
            'In \'mixed emergency mode\' the CHP is forced on, but the '
            'modulation is a 50:50 mix of full power and modelpredictive '
            'output. Full power mode clips the modulation to the maximum '
            'value.\n'
            'Value must be int, float >= 0.'
            '\n\nDefaults:\n{0}'.format(repr(self._emerg_prms))
        )
        assert 'emergency_cntrl' in kwds and isinstance(
            kwds['emergency_cntrl'], (str, dict)
        ), err_ep
        if isinstance(kwds['emergency_cntrl'], dict):  # update values
            self._emerg_prms.update(kwds['emergency_cntrl'])
        else:  # take defaults
            assert kwds['emergency_cntrl'] == 'default', err_ep

        # check and store/process variables
        assert isinstance(self._emerg_prms['use_hysteresis'], bool), err_ep
        self._emerg_use_hyst = self._emerg_prms['use_hysteresis']
        assert (
            isinstance(self._emerg_prms['hysteresis'], (int, float))
            and self._emerg_prms['hysteresis'] >= 0.0
        ), err_ep
        self._hyst_val = self._emerg_prms['hysteresis']
        assert (
            isinstance(self._emerg_prms['full_power_offset'], (int, float))
            and self._emerg_prms['full_power_offset'] >= 0.0
        ), err_ep
        self._act_val_fp_diff = self._emerg_prms['full_power_offset']
        self._act_val_full_power = (
            self._act_val - self._act_val_fp_diff
            if self._act_sgn == 'le'
            else self._act_val + self._act_val_fp_diff
        )

        # get CHP parameters:
        chp_params = {
            'pel_chp': (
                'Electric CHP power in Watt at full throttle operation. '
                'int, float > 0.'
            ),
            'pth_chp': (
                'Thermal CHP power in Watt at full throttle operation. '
                'int, float > 0.'
            ),
            'eta_el': (
                'Electric efficiency of the CHP plant at full throttle '
                'operation. 0 < float < 1.'
            ),
            'mod_range': (
                'CHP electric power modulation range as a tuple with '
                '(lower_limit, upper_limit), typically (.5, 1.).'
            ),
        }
        err_chp = (
            self._base_err
            + self._arg_err.format('chp_params')
            + '`chp_params` must be given. The CHP plant parameters as '
            'described in:\n{0}.\n\n'
            'Replace the description in the dict with the values to set. '
            'Since these values are used in the outer optimization loop, non '
            'exact values, f.i. disregarding dependency of efficiency on the '
            'modulation, are sufficient.'.format(repr(chp_params))
        )
        assert 'chp_params' in kwds, err_chp
        # check that all keys are given
        assert np.all(
            [key in kwds['chp_params'] for key in chp_params.keys()]
        ), err_chp
        # assert that all are in the correct range and store them:
        self.pel_chp_max = kwds['chp_params']['pel_chp']
        assert (
            isinstance(self.pel_chp_max, (int, float))
            and self.pel_chp_max > 0.0
        ), '`chp_params[\'pel_chp\']` not int/float or not > 0.'
        self.pth_chp = kwds['chp_params']['pth_chp']
        assert (
            isinstance(self.pth_chp, (int, float)) and self.pth_chp > 0.0
        ), '`chp_params[\'pth_chp\']` not int/float or not > 0.'
        self.eta_el = kwds['chp_params']['eta_el']
        assert (
            isinstance(self.eta_el, float) and 0.0 < self.eta_el < 1.0
        ), '`chp_params[\'eta_el\']` not float or not 0. < eta < 1.'
        self._modrange_chp = kwds['chp_params']['mod_range']
        assert (
            isinstance(self._modrange_chp, (tuple, list))
            and len(self._modrange_chp) == 2
            and (0.0 <= self._modrange_chp[0] <= self._modrange_chp[1] <= 1.0)
        ), (
            '`chp_params[\'mod_range\']` not tuple or not '
            '0. <= lower_limit <= upper_limit <= 1.'
        )
        # also make a shortcut, for low LOC coding
        self._mrc = self._modrange_chp

        # check for and set optimization params:
        self._opt_params = {  # default parameters
            'opt_timeframe': '2d',
            'opt_every': '15min',
            'increase_continuity': True,
            'max_iter': 500,
        }
        err_op = (
            self._base_err
            + self._arg_err.format('optimization_params')
            + '`opt_params` must be given. Either set to '
            '`opt_params=\'default\' to use default parameters '
            '(see below) or pass a dictionary with values to change.\n\n'
            'Parameters:\n'
            '    - `opt_timeframe`: The timeframe over which the optimization '
            'is performed every `opt_every` period. Must be str.\n'
            '    - `opt_every`: How often is the optimization algorithm '
            'performed? Must be str.\n'
            '    - `increase_continuity`: Try to increase result continuity '
            'to reduce CHP plant switching by incorporating the previous '
            'step in the result post processing. Must be bool.\n'
            '    - `max_iter`: Max. number of iterations for the opt. '
            'algorithm, after which the initial guess will be chosen. Must be '
            'int.\n'
            '\n\nDefaults:\n{0}'.format(repr(self._opt_params))
        )
        assert 'opt_params' in kwds and isinstance(
            kwds['opt_params'], (str, dict)
        ), err_op
        if isinstance(kwds['opt_params'], dict):
            self._opt_params.update(kwds['opt_params'])
        else:
            assert kwds['opt_params'] == 'default', err_op
        # get values in seconds as floats (and timedelta if needed again)
        self._opt_tf_sec = pd.to_timedelta(
            self._opt_params['opt_timeframe']
        ).total_seconds()
        self._opt_every_td = pd.to_timedelta(self._opt_params['opt_every'])
        self._opt_every = self._opt_every_td.total_seconds()
        # get bool checker for continuity:
        self._inc_cont = self._opt_params['increase_continuity']
        assert isinstance(self._inc_cont, bool), err_op
        # size of the control array to optimize
        self._optsize = int(np.ceil(self._opt_tf_sec / self._opt_every))
        # incremental for timesteps after which an optimization must be
        # performed. reversed tolist allows popping the last element as soon as
        # optimization is done. append one more step with total tf length to
        # make sure the list is never empty.
        self._opt_steps = np.arange(
            0.0, master_cls.timeframe, self._opt_every
        )[::-1].tolist()
        self._opt_steps.insert(0, master_cls.timeframe)
        # make arrays to save optimization results to (fill with nan, since
        # zeros can be valid results):
        self._opt_chp_mod = np.full(
            (len(self._opt_steps), self._optsize), np.nan, dtype=np.float64
        )
        # only last row first col is zero to enable checks in first iter
        self._opt_chp_mod[-1, 0] = 0.0
        # post processed results
        self._opt_chp_mod_post = self._opt_chp_mod.copy()
        # status and cost results
        self._opt_status = np.full(
            len(self._opt_steps), np.nan, dtype=np.float64
        )
        self._opt_cost = np.full_like(self._opt_status, np.nan)
        # step counter and sim time of the opt.
        self._opt_steptime = np.full(
            (len(self._opt_steps), 2), np.nan, dtype=np.float64
        )

        # get all inputs and timeseries for optimization
        # heat capacities of the TES
        err_tessoc = (
            self._base_err
            + self._arg_err.format('tes_soc_minmax')
            + '`tes_soc_minmax` must be given as a tuple, defining the minimum '
            '(fully discharged) and maximum (fully charged) energy content '
            'in the TES in **kWh**.\n'
            'Calculation: '
            'E = sum((theta - theta_ref) * V / n * rho * cp) * conv\n'
            'with the TES temperature in each cell `theta`, the reference '
            'temperature `theta_ref`=0°C, the total TES '
            'volume `V`, the number of cells `n` of the temperature array, '
            'and the mean density and spec. heat cap. `rho` and `cp`. The '
            'mean values of rho and cp should at least be calculated with '
            '`(rho(theta) + rho(theta_ref))/2` for increased accuracy. '
            '`conv=1/3.6e6` is needed to convert E from J to kWh.'
        )
        assert (
            'tes_soc_minmax' in kwds
            and isinstance(kwds['tes_soc_minmax'], tuple)
            and len(kwds['tes_soc_minmax']) == 2
        ), err_tessoc
        self._tes_caps = (kwds['tes_soc_minmax'][0], kwds['tes_soc_minmax'][1])
        assert isinstance(self._tes_caps[0], (int, float)) and isinstance(
            self._tes_caps[1], (int, float)
        ), '`tes_soc_minmax` values are not numeric.'
        assert 0 <= self._tes_caps[0] < self._tes_caps[1], (
            '`tes_soc_minmax` values must be >= 0 and the second value must '
            'be greater than the first.'
        )
        # TES caps will be adapted when emergency control hits. thus copy to
        # mutable array
        self._tes_caps_adpt = np.array(self._tes_caps, dtype=np.float64)
        # scalars to store last adpt result for access while restoring
        self._tes_cap_adpt_lo = self._tes_caps[0]
        self._tes_cap_adpt_hi = self._tes_caps[1]
        # time of last adaption, for [lo, hi], init. with zeros
        self._time_last_cap_adpt = np.zeros(2, dtype=np.float64)
        # list for results of adaptions and restoring, setup:
        # (timestep, id, lo, hi)  -> id=1 == restore, id=-1 == adapt
        self._tes_caps_res = []
        # list for TES SOCs depending on caps, setup:
        # (timestep, soc, diff_to_lo, diff_to_hi)
        self._tes_socs = []

        # check for and set tes capacity restore param:
        self.__tes_rstr_dflt = float(24 * 3600)  # default parameters
        err_trt = (
            self._base_err
            + self._arg_err.format('tes_cap_restore_time')
            + '`tes_cap_restore_time` must be given. Either set to '
            '`tes_cap_restore_time=\'default\' to use default parameter '
            '(see below) or define an int/float value > 0.\n'
            'The TES capacity will be adapted to the current state each time '
            'the emergency control overrides the modelpredictive control. '
            'After each adaption, the TES capacity will slowly revert to '
            'its initial state. `tes_cap_restore_time` defines the time in '
            '[s], after which the TES capacity is fully reverted.\nThe '
            'default is {0}s(=={1}h).'.format(
                self.__tes_rstr_dflt, self.__tes_rstr_dflt / 3600
            )
        )
        assert 'tes_cap_restore_time' in kwds and isinstance(
            kwds['tes_cap_restore_time'], (str, int, float)
        ), err_trt
        if isinstance(kwds['tes_cap_restore_time'], str):
            assert kwds['tes_cap_restore_time'] == 'default', err_trt
            self._cap_restore_time = self.__tes_rstr_dflt
        else:  # set input value
            assert kwds['tes_cap_restore_time'] > 0.0, err_trt
            self._cap_restore_time = float(kwds['tes_cap_restore_time'])

        # grab reference to TES:
        err_tes = (
            self._base_err
            + self._arg_err.format('tes_part')
            + '`tes_part` defining the part-name of the TES to consider in the '
            'SOC-optimization has to be given.'
        )
        assert 'tes_part' in kwds, err_tes
        self._tes_part = self._Controls__check_args(  # check for existence
            'tes_part', caller='add_control', tes_part=kwds['tes_part']
        )
        self._tes_part_ref = self._models.parts[self._tes_part]
        # get view to temperature array and calculate constant factor from
        # volume per cell and average values for rho and cp:
        self._tes_temp = self._tes_part_ref.T[:]
        self._tes_const = self._tes_part_ref.V_cell * 985.0 * 4191.0
        self._tes_cell_vol = self._tes_part_ref.V_cell  # also save this ref.

        # power profiles for the optimizing the production schedula:
        self.__err_prof = (
            self._base_err
            + self._arg_err.format('opt_profiles')
            + 'The power demand profiles in __**Watt**__ for the optimization '
            'of the long term production schedule '
            'must be given as a pd.DataFrame with a DatetimeIndex. '
            'Using average daily profiles is recommended, since these '
            'profiles will be used for predicting an optimal load schedule.\n'
            'The index period (pandas: freq) should be substantially shorter '
            'than the value `opt_every={0:.2f}s` (set with `opt_params`). '
            'The following columns are required:\n'
            '    - `heat_dmd`: heat demand in W. Primarily PHW and '
            'recirculation. Space heating only if supplied by the CHP plant.\n'
            '    - `pel_baseload`: Electric baseload power of the building in '
            'W. This is the electric power, which the CHP plant owner can use '
            'directly, thus a reduced EEG-Umlage has to be paid, while still '
            'earning the full reselling-price and KWKG-Verguetung. This '
            'typically includes f.i. elevator, public lights, carpark '
            'ventilation etc.. SET TO ZERO, if there is no baseload of this '
            'type.'
            '    - `pel_user`: Electric power demand in W of users participating '
            'in a Mieterstrommodell. Set to zero if none.'
            '    - `pel_user_noms`: Electric power demand in W of users NOT '
            'participating in a Mieterstrommodell. Set to zero if none. This '
            'value will be ignored during the optimization.'
        ).format(self._opt_every)
        assert 'opt_profiles' in kwds, self.__err_prof
        # process the profiles:
        self._process_profiles(kwds['opt_profiles'], which='opt', **kwds)

        # power profiles for controlling the CHP plant in each step:
        self.__err_prof_ctrl = (
            self._base_err
            + self._arg_err.format('ctrl_profiles')
            + 'The power demand profiles in __**Watt**__ for the control and '
            'optimization at each timestep (short term) '
            'must be given as a pd.DataFrame with a DatetimeIndex. These '
            'profiles are used to control the CHP plant at each step, thus '
            'non-averaged real measurement data profiles are recommended.\n'
            'The index period (pandas: freq) must be 1s.'
            'The following columns are required:\n'
            '    - `pel_baseload`: Electric baseload power of the building in '
            'W. This is the electric power, which the CHP plant owner can use '
            'directly, thus a reduced EEG-Umlage has to be paid, while still '
            'earning the full reselling-price and KWKG-Verguetung. This '
            'typically includes f.i. elevator, public lights, carpark '
            'ventilation etc.. SET TO ZERO, if there is no baseload of this '
            'type.'
            '    - `pel_user`: Electric power demand in W of users participating '
            'in a Mieterstrommodell. Set to zero if none.'
            '    - `pel_user_noms`: Electric power demand in W of users NOT '
            'participating in a Mieterstrommodell. Set to zero if none. This '
            'value will be ignored during the optimization.'
        ).format(self._opt_every)
        assert 'ctrl_profiles' in kwds, self.__err_prof_ctrl
        # process the profiles:
        self._process_profiles(kwds['ctrl_profiles'], which='ctrl', **kwds)

        # check for and set cost params:
        self._cost_params = {  # default parameters
            'gaspreis': 3.84,
            'esteuerrueck_gas': 0.55,
            'netzneg': 0.0,
            'eex_baseload': 'Q3 2019',
            'strompreis_zukauf': 23.5,
            'aufschlag_verkauf': 0.8,
            'penalty_wvk': 0.0,
            'flat_instandh': True,
            'instandh_cost': 'infer',
            'penalty_pow_instandh': 0.25,
        }
        err_cp = (
            self._base_err
            + self._arg_err.format('costs')
            + '`costs` must be given. Either set to '
            '`costs=\'default\' to use default parameters '
            '(see below) or pass a dictionary with values to change. '
            'Currently values are the net values, that is including energy '
            'taxes, el. power taxes etc. but without sales taxes.\n'
            '    - `gaspreis`: Gaspreis in Cent/kWh_gas.\n'
            '    - `esteuerrueck_gas`: Gas tax refund in Cent/kWh_gas, f.i. '
            'according to EnergieStG § 53a Absatz (6) and § 2 Absatz 3 Satz 1 '
            'Nummer 4 (default).\n'
            '    - `netzneg`: Refund of avoided Netznutzungsentgelte for '
            'electric power in Cent/kWh_el. Scrapped starting 01.01.2023 '
            '(default).\n'
            '    - `eex_baseload`: The EEX baseload (also KWK-Index) price '
            'must either be given as the price in Cent/kWh_el as a '
            'float/int > 0 OR as a string defining the quarter of the year '
            'to take the value from. The quarterly values are extracted from '
            'table `data_tables/phelix-quarterly-data-data-data.xls`. Update '
            'this table to include new values. Default is \'Q3 2019\'=3.745.\n'
            '    - `strompreis_zukauf`: Price to buy electricity from an '
            'energy supply company in Cent/kWh_el.\n'
            '    - `aufschlag_verkauf`: Surcharge for reselling bought '
            'electricity to a third party in Cent/kWh_el. Also defines the '
            'selling price of electricity produced in the CHP plant with '
            '`strompreis_zukauf + aufschlag_verkauf`.\n'
            '    - `penalty_wvk`: Penalty for reselling electric power in '
            'Cent/kWh_el. May be introduced, to incorporate a different '
            'selling price for electric power produced in the CHP plant and '
            'bought electric power. The re-selling electric power price is '
            'calculated with:'
            '`strompreis_zukauf + aufschlag_verkauf - penalty_wvk`.\n'
            '    - `flat_instandh`: Bool flag if to set if using a flat '
            'maintenance tariff. If True, the value set to `instandh_cost` '
            'will be used to calculate the cost per operating hour, '
            'independent of the CHP plant modulation. If False, an '
            'approximation by ASUE for the cost per kWh will be used, with a '
            'penalty of `modulation**penalty_pow_instandh`.\n'
            '    - `instandh_cost`: Flat CHP plant maintenance cost, '
            'independent of the modulation. If given as float, the value '
            'will be in Cent/h, where h is the operation hour. '
            'If set to `\'infer\'`, the value will be infered from a table '
            'of usual market terms depending on the nominal electric CHP '
            'power. . Only used if `flat_instandh=True`.\n'
            '    - `penalty_pow_instandh`: Penalize the maintenance cost '
            'at low modulations with f.i. `0.6**penalty_pow_instandh` for '
            '60% modulation. Only applies if `flat_instandh=False`. Setting '
            'to 1 disables penalty.\n'
            '\nDefaults:\n{0}'.format(repr(self._cost_params))
        )
        assert 'costs' in kwds and isinstance(
            kwds['costs'], (str, dict)
        ), err_cp
        if isinstance(kwds['costs'], dict):
            self._cost_params.update(kwds['costs'])
        else:
            assert kwds['costs'] == 'default', err_cp
        # gaspreis
        self._gaspreis = self._cost_params['gaspreis']
        assert (
            isinstance(self._gaspreis, (int, float)) and self._gaspreis >= 0.0
        ), '`gaspreis` must be a scalar value >= 0.'
        # energiesteuerrueckerstattung
        self._esr_gas = self._cost_params['esteuerrueck_gas']
        assert (
            isinstance(self._esr_gas, (int, float)) and self._esr_gas >= 0.0
        ), '`esteuerrueck_gas` must be a scalar value >= 0.'
        # vermiedene netznutzungsentgelte
        self._vnneg = self._cost_params['netzneg']
        assert (
            isinstance(self._vnneg, (int, float)) and self._vnneg >= 0.0
        ), '`netzneg` must be a scalar value >= 0.'
        # eex baseload preis
        assert isinstance(self._cost_params['eex_baseload'], (int, float, str))
        if isinstance(self._cost_params['eex_baseload'], str):
            self._eex = ModelPredCHP._load_eex_baseload(
                quartal=self._cost_params['eex_baseload'], reload_xls=True
            )
        else:
            assert self._cost_params['eex_baseload'] <= 1000
            self._eex = self._cost_params['eex_baseload']
        # strompreis zukauf
        self._spz = self._cost_params['strompreis_zukauf']
        assert (
            isinstance(self._spz, (int, float)) and self._spz >= 0.0
        ), '`strompreis_zukauf` must be a scalar value >= 0.'
        # aufschlag weiterverkauf strom
        self._awvk = self._cost_params['aufschlag_verkauf']
        assert (
            isinstance(self._awvk, (int, float)) and self._awvk >= 0.0
        ), '`aufschlag_verkauf` must be a scalar value >= 0.'
        # penalty weiterverkauf strom
        self._pwvk = self._cost_params['penalty_wvk']
        assert (
            isinstance(self._pwvk, (int, float)) and self._pwvk >= 0.0
        ), '`penalty_wvk` must be a scalar value >= 0.'
        # flat instandhaltung
        self._flat_instandh = self._cost_params['flat_instandh']
        assert isinstance(
            self._flat_instandh, bool
        ), '`flat_instandh` must be a bool.'
        # flat instandhaltungskosten
        assert isinstance(
            self._cost_params['instandh_cost'], (float, str)
        ), '`instandh_cost` must be float or `infer`.'
        if self._cost_params['instandh_cost'] == 'infer':
            self._instandh_cost = ModelPredCHP.infer_instandh(self.pel_chp_max)
        else:
            self._instandh_cost = self._cost_params['instandh_cost']
            assert 0 <= self._instandh_cost, 'Maintenance cost must be >= 0.'
        # penalty non-flat instandh for low modulation
        self._ppic = self._cost_params['penalty_pow_instandh']
        assert (
            isinstance(self._ppic, float) and self._ppic > 0.0
        ), '`penalty_wvk` must be a scalar float > 0.'

        # bound modulation variable to 0-1 range
        self._opt_bounds = np.zeros((self._optsize, 2))
        self._opt_bounds[:, 1] = 1

        # bind constraint functions to dict in the way the SLSQP optimizer
        # expects them
        self._opt_constraints = (
            {'type': 'ineq', 'fun': self.tes_cap_before_empty},
            {'type': 'ineq', 'fun': self.tes_cap_before_full},
        )

        # precalculate constants:
        self._unit_fac = 1 / 3.6e6  # conversion from Joule to kWh
        # time conversion factor for cost calculation. Since cost calculation
        # is hour-based, 1/3600 yields the conversion factor:
        self._freq_fac = self._opt_every / 3600.0
        self._precalc_pth_oe_uf = (
            self.pth_chp * self._opt_every * self._unit_fac
        )

        # set number of optimizations incremental counter:
        self.num_opts = 0
        # initialize previous longterm (outer) opt. result to 0:
        self._lt_prev_opt = 0.0
        # initialize previous CV value to 0:
        self._cv_prev_opt = 0.0

        # initialize bool checker for consecutive emergency control to False:
        self._emerg_cnsctv = False
        self._emerg_on = False  # also if emergency control activation
        # and also checker if caps are adapted (or not fully reverted)
        self._caps_adapted = False

        # and register this control for special disk storing:
        self._models._disk_store_utility[
            self.name
        ] = self._special_store_to_disk

    def _special_store_to_disk(self, store, store_kdws):
        index = self._models._disk_store_timevec
        strt_time = index[0]
        # tes cap results
        if len(self._tes_caps_res) == 0:  # add init. value if empty
            self._tes_caps_res.append((0.0, 0, *self._tes_caps))
        tcapr = pd.DataFrame(
            self._tes_caps_res, columns=['time', 'id', 'cap_lo', 'cap_hi']
        )
        tcapr.index = pd.to_timedelta(tcapr['time'], unit='s') + strt_time
        tcapr = tcapr.drop(columns=['time'])  # drop time col, since in idx
        tcapr = tcapr.dropna(how='any', thresh=2)  # drop nan to avoid NaT idx

        # tes SOC results:
        if len(self._tes_socs) == 0:  # add init. value if empty
            self._tes_socs.append((0.0, 0, 0.0, 0.0))
        tsocs = pd.DataFrame(
            self._tes_socs, columns=['time', 'SOC', 'SOC_to_lo', 'SOC_to_hi']
        )
        tsocs.index = pd.to_timedelta(tsocs['time'], unit='s') + strt_time
        tsocs = tsocs.drop(columns=['time'])  # drop time col, since in idx
        tsocs = tsocs.dropna(how='any', thresh=2)  # drop nan to avoid NaT idx

        # remove duplicated indices
        tcapr = tcapr.loc[~tcapr.index.duplicated()]
        tsocs = tsocs.loc[~tsocs.index.duplicated()]

        # optimization results
        # opt step and time:
        ost = pd.DataFrame(self._opt_steptime, columns=['opt_step', 'time'])
        ost.index = pd.to_timedelta(ost['time'], unit='s') + strt_time
        ost = ost.drop(columns=['time'])  # drop time col, since in idx
        # modulation as opt. result and postprocessed opt result. dropna with
        # thresh=2 to avoid dropping rows at the end where partly valid
        # results are, but still drop the last row
        mod = pd.DataFrame(self._opt_chp_mod, index=ost.index).dropna(
            how='any', thresh=2
        )
        mod_pp = pd.DataFrame(self._opt_chp_mod_post, index=ost.index).dropna(
            how='any', thresh=2
        )
        # opt status and cost
        ostat_cost = pd.DataFrame(
            data={
                'step': ost['opt_step'],
                'status': self._opt_status,
                'cost': self._opt_cost,
            },
            index=ost.index,
        ).dropna(how='any')

        # store TES caps and SOCS
        store.put('{0}/tes_cap'.format(self.name), tcapr, **store_kdws)
        store.put('{0}/tes_soc'.format(self.name), tsocs, **store_kdws)
        # store opt. results
        store.put(
            '{0}_opt/opt_res_longterm'.format(self.name), mod, **store_kdws
        )
        store.put(
            '{0}_opt/opt_res_longterm_postproc'.format(self.name),
            mod_pp,
            **store_kdws
        )
        store.put(
            '{0}_opt/opt_stats'.format(self.name), ostat_cost, **store_kdws
        )

    def _process_opt_profiles(self, opt_prfls, **kwds):
        """Process the load schedule optimization profiles."""
        # make all required checks:
        assert isinstance(opt_prfls, pd.DataFrame), self.__err_prof
        assert isinstance(opt_prfls.index, pd.DatetimeIndex), self.__err_prof
        assert not opt_prfls.isna().any().any(), 'NaN found in `opt_profiles`'
        assert set(  # check if all required columns are given
            ['heat_dmd', 'pel_baseload', 'pel_user', 'pel_user_noms']
        ) == set(opt_prfls.columns), self.__err_prof
        # check index frequency:
        prof_freq = getattr(opt_prfls.index, 'freq', None)
        prof_freq = (
            prof_freq
            if prof_freq is not None
            else getattr(opt_prfls.index, 'inferred_freq', None)
        )
        assert prof_freq is not None, '`opt_profiles` no index freq found.'
        # get index period and assert that it is lower than opt every
        # (higher freq)
        try:  # catch inferred freq with multiple == 1
            self._opt_prfl_input_prd = pd.to_timedelta(
                prof_freq
            ).total_seconds()
        except ValueError:
            self._opt_prfl_input_prd = pd.to_timedelta(
                '1{0}'.format(prof_freq)
            ).total_seconds()
        assert self._opt_prfl_input_prd <= self._opt_every, (
            '`opt_profiles` index period (pandas: freq) higher than '
            '`opt_every={0}`.'.format(self._opt_every)
        )

        # check if it is likely that kW profiles have been given
        if (
            opt_prfls['pel_user'].max() < 1000
            and not kwds['silence_unit_warn']
        ):
            raise ValueError(
                'it seems like electric power profiles have '
                'been given in Kilowatt. Please make sure that '
                'Watt are given. To silence this warning, set '
                '`silence_unit_warn=True.`'
            )

        # now that most checks have been done, resample data to opt_every freq.
        # mean + interpolate avoids nan in the data
        opt_prfls_rs = (
            opt_prfls.resample(self._opt_every_td)
            .mean()
            .interpolate(method='time')
        )
        # get length of df and assert that it is longer than the simulation
        # timeframe:
        assert (
            opt_prfls_rs.index[-1] - opt_prfls_rs.index[0]
        ).total_seconds() >= self._models.timeframe, (
            '`opt_profiles` duration must be longer than the simulation '
            'timeframe. This error is raised after resampling to the '
            '`opt_every={0}s` period which typically results in cutting '
            'off the last period. Thus make sure that the profiles are '
            'at least simulation timeframe + opt_every long.'
        )

        # copy opt profiles with all values in Watt:
        self._opt_profiles = opt_prfls_rs.copy()
        # kumulative sum yields Watt*opt_every. To get to Wh, divide by
        # 1h resp. 3600s and by 1k to get kWh:
        self._opt_profiles['heat_dmd_kum'] = (
            self._opt_profiles['heat_dmd'].cumsum() * self._opt_every / 3.6e6
        )

        # make CHP on kumulative power array for time till TES full calculation
        self._chp_on_heat = (
            np.cumsum(np.full(self._optsize, self.pth_chp))
            * self._opt_every
            / 3.6e6
        )

    def _process_ctrl_profiles(self, ctrl_prfls, **kwds):
        """Process the in-step control optimization profiles."""
        # make all required checks:
        assert isinstance(ctrl_prfls, pd.DataFrame), self.__err_prof_ctrl
        assert isinstance(
            ctrl_prfls.index, pd.DatetimeIndex
        ), self.__err_prof_ctrl
        assert (
            not ctrl_prfls.isna().any().any()
        ), 'NaN found in `ctrl_profiles`'
        assert set(  # check if all required columns are given
            ['pel_baseload', 'pel_user', 'pel_user_noms']
        ) == set(ctrl_prfls.columns), self.__err_prof_ctrl
        # check index frequency:
        prof_freq = getattr(ctrl_prfls.index, 'freq', None)
        prof_freq = (
            prof_freq
            if prof_freq is not None
            else getattr(ctrl_prfls.index, 'inferred_freq', None)
        )
        assert prof_freq is not None, '`ctrl_profiles` no index freq found.'
        # get index period and assert that it is lower than opt every
        # (higher freq)
        try:  # catch inferred freq with multiple == 1
            self._opt_prfl_input_prd = pd.to_timedelta(
                prof_freq
            ).total_seconds()
        except ValueError:
            self._opt_prfl_input_prd = pd.to_timedelta(
                '1{0}'.format(prof_freq)
            ).total_seconds()
        assert self._opt_prfl_input_prd <= self._opt_every, (
            '`ctrl_profiles` index period (pandas: freq) higher than '
            '`opt_every={0}`.'.format(self._opt_every)
        )

        # check if it is likely that kW profiles have been given
        if (
            ctrl_prfls['pel_user'].max() < 1000
            and not kwds['silence_unit_warn']
        ):
            raise ValueError(
                'it seems like electric power profiles have '
                'been given in Kilowatt. Please make sure that '
                'Watt are given. To silence this warning, set '
                '`silence_unit_warn=True.`'
            )

        # now that most checks have been done, resample data to opt_every freq.
        # mean + interpolate avoids nan in the data
        ctrl_prfls_rs = (
            ctrl_prfls.resample(self._opt_every_td)
            .mean()
            .interpolate(method='time')
        )
        # get length of df and assert that it is longer than the simulation
        # timeframe:
        assert (
            ctrl_prfls_rs.index[-1] - ctrl_prfls_rs.index[0]
        ).total_seconds() >= self._models.timeframe, (
            '`ctrl_profiles` duration must be longer than the simulation '
            'timeframe. This error is raised after resampling to the '
            '`opt_every={0}s` period which typically results in cutting '
            'off the last period. Thus make sure that the profiles are '
            'at least simulation timeframe + opt_every long.'
        )

        # copy opt profiles with all values in Watt:
        self._ctrl_profiles = ctrl_prfls_rs.copy()
        # kumulative sum yields Watt*opt_every. To get to Wh, divide by
        # 1h resp. 3600s and by 1k to get kWh:
        self._ctrl_profiles['heat_dmd_kum'] = (
            self._ctrl_profiles['heat_dmd'].cumsum() * self._opt_every / 3.6e6
        )

        # make CHP on kumulative power array for time till TES full calculation
        self._chp_on_heat = (
            np.cumsum(np.full(self._optsize, self.pth_chp))
            * self._opt_every
            / 3.6e6
        )

    def _process_profiles(self, prfls, which, **kwds):
        """Process the in-step control optimization profiles."""
        assert which in ('opt', 'ctrl')
        err = self.__err_prof if which == 'opt' else self.__err_prof_ctrl
        # make all required checks:
        assert isinstance(prfls, pd.DataFrame), err
        assert isinstance(prfls.index, pd.DatetimeIndex), err
        assert (
            not prfls.isna().any().any()
        ), 'NaN found in `{0}_profiles`'.format(which)
        # check if all required columns are given
        cols = (
            ['heat_dmd', 'pel_baseload', 'pel_user', 'pel_user_noms']
            if which == 'opt'
            else ['pel_baseload', 'pel_user', 'pel_user_noms']
        )
        assert set(cols) == set(prfls.columns), err
        # check index frequency:
        prof_freq = getattr(prfls.index, 'freq', None)
        prof_freq = (
            prof_freq
            if prof_freq is not None
            else getattr(prfls.index, 'inferred_freq', None)
        )
        assert (
            prof_freq is not None
        ), '`{0}_profiles` no index freq found.'.format(which)

        # check for frequencies, get them as time period
        try:  # catch inferred freq with multiple == 1
            setattr(
                self,
                '_{0}_prfl_input_prd'.format(which),
                pd.to_timedelta(prof_freq).total_seconds(),
            )
        except ValueError:
            setattr(
                self,
                '_{0}_prfl_input_prd'.format(which),
                pd.to_timedelta('1{0}'.format(prof_freq)).total_seconds(),
            )

        if which == 'opt':
            # assert that freq/period is lower than opt every (higher freq)
            assert self._opt_prfl_input_prd <= self._opt_every, (
                '`opt_profiles` index period (pandas: freq) higher than '
                '`opt_every={0}`.'.format(self._opt_every)
            )
        else:
            # assert that freq/period is 1s
            assert (
                self._ctrl_prfl_input_prd == 1.0
            ), '`ctrl_profiles` index period (pandas: freq) not equal to 1s.'

        # check if it is likely that kW profiles have been given
        if prfls['pel_user'].max() < 1000 and not kwds['silence_unit_warn']:
            raise ValueError(
                'It seems like the electric power profiles have been given '
                'in Kilowatt. Please make sure that the unit is Watt. '
                'To silence this warning, set `silence_unit_warn=True.`'
            )

        if which == 'opt':
            # now that most checks have been done, resample data to opt_every
            # freq. mean + interpolate avoids nan in the data
            prfls = (
                prfls.resample(self._opt_every_td)
                .mean()
                .interpolate(method='time')
            )

            # get length of df and assert that it is longer than the simulation
            # timeframe:
            assert (
                prfls.index[-1] - prfls.index[0]
            ).total_seconds() >= self._models.timeframe, (
                '`opt_profiles` duration must be longer than the '
                'simulation timeframe. This error is raised after '
                'resampling to the `opt_every={0}s` period which '
                'typically results in cutting off the last period. Thus '
                'make sure that the profiles are at least simulation '
                'timeframe + opt_every long.'.format(self._opt_every)
            )
        else:
            assert (
                prfls.index[-1] - prfls.index[0]
            ).total_seconds() >= self._models.timeframe, (
                '`ctrl_profiles` duration must be longer than the '
                'simulation timeframe.'
            )

        # copy opt profiles with all values in Watt:
        setattr(self, '_{0}_profiles'.format(which), prfls.copy())

        if which == 'opt':  # some more stuff specific to opt profiles...
            # kumulative sum yields Watt*opt_every. To get to Wh, divide by
            # 1h resp. 3600s and by 1k to get kWh:
            self._opt_profiles['heat_dmd_kum'] = (
                self._opt_profiles['heat_dmd'].cumsum()
                * self._opt_every
                / 3.6e6
            )

            # make CHP on kumulative power array for time till TES full
            # calculation
            self._chp_on_heat = (
                np.cumsum(np.full(self._optsize, self.pth_chp))
                * self._opt_every
                / 3.6e6
            )

    def _reset_to_init_cond(self):
        self.cv = 0.0
        self._prev_cv = 0.0
        self._deriv = 0.0
        self.error = 0.0
        self.prev_error = 0.0
        self.prev2_error = 0.0
        self.prev_i_term = 0.0

    def tes_cap_before_empty(self, x):
        """
        Make sure the TES is **never empty**.

        This is an inequality constraint used as an input to the SLSQP
        optimizer. The optimization algorithm aims to constrain the
        optimization variables such that the return value of inequality
        constraint functions is >= 0. This means, this function will return
        values < 0, **where the TES is empty**.

        `tes_soc_min` is the remaining capacity before the TES is **empty** in
        kWh, `heat_dmd` is the cumulative heat demand in kWh,
        `_precalc_pth_oe_uf` is a constant pre-calculated factor of the nominal
        thermal CHP plant power in Watt, the optimization step size in seconds
        and the unit conversion factor from J to kWh.

        Parameters
        ----------
        x : np.ndarray, int, float,
            Modulation of the CHP plant, from 0-1.

        Returns
        -------
        np.ndarray, int, float
            Remaining TES energy content in kWh before empty at each timestep
            in x.

        """
        return (
            self.tes_soc_min
            - self._heat_dmd
            + np.cumsum(x * self._precalc_pth_oe_uf)
        )

    def tes_cap_before_full(self, x):
        """
        Make sure the TES is **never overly full**.

        This is an inequality constraint used as an input to the SLSQP
        optimizer. The optimization algorithm aims to constrain the
        optimization variables such that the return value of inequality
        constraint functions is >= 0. This means, this function will return
        values < 0, **where the TES is overcharged**.

        `tes_soc_max` is the remaining capacity before the TES is **full** in
        kWh, `heat_dmd` is the cumulative heat demand in kWh,
        `_precalc_pth_oe_uf` is a constant pre-calculated factor of the nominal
        thermal CHP plant power in Watt, the optimization step size in seconds
        and the unit conversion factor from J to kWh.

        Parameters
        ----------
        x : np.ndarray, int, float,
            Modulation of the CHP plant, from 0-1.

        Returns
        -------
        np.ndarray, int, float
            Remaining TES energy content in kWh before full at each timestep
            in x.

        """
        return self.tes_soc_max - (
            np.cumsum(x * self._precalc_pth_oe_uf) - self._heat_dmd
        )

    def _tes_heat_content(self):
        return (
            self._tes_temp
            * self._unit_fac
            * self._tes_cell_vol
            * (_pf.cp_water(self._tes_temp) + _pf.cp_water(0.0))
            / 2
            * (_pf.rho_water(self._tes_temp) + _pf.rho_water(0.0))
            / 2
        ).sum()

    def _emergency_control(self, hysteresis=True):
        # set bool checker for this step to False
        emerg_cntrl = False  # Modelpredictive cntrl overwritten?
        # if hysteresis is set AND this is a consecutive run of the controller
        # (has already been active in the step before), then add hystersis
        # to activation value. (no hysteresis to deact., since this declines
        # too slow).
        if hysteresis and self._emerg_cnsctv:
            self._hyst = self._hyst_val
        else:  # else set to zero
            self._hyst = 0.0

        # controller activation
        if self._act_sgn == 'le':
            if self.on_sensor_val[0] < (self._act_val + self._hyst):
                # set cv (activate chp plant) and opt now (activate third step
                # opt with a mix of cv sat max AND emergency ctrl)
                self.cv = self._cv_sat_max
                self.opt_chp_now = self._cv_sat_max
                # if an additional threshold is reached, activate double CV to
                # make sure, that full power is used even when mixing with
                # third opt. step. overflow is clipped anyways.
                if self.on_sensor_val[0] < self._act_val_full_power:
                    self.opt_chp_now *= 2
                # set bool checkers
                emerg_cntrl = True
                self._emerg_on = True  # emergency CHP activation
                # adapt lower capacity
                self._adapt_cap(lower=True, check=True)
        else:  # if greater
            if self.on_sensor_val[0] > (self._act_val - self._hyst):
                # set cv (activate chp plant) and opt now (activate third step
                # opt with a mix of cv sat max AND emergency ctrl)
                self.cv = self._cv_sat_max
                self.opt_chp_now = self._cv_sat_max
                # if an additional threshold is reached, activate double CV to
                # make sure, that full power is used even when mixing with
                # third opt. step. overflow is clipped anyways.
                if self.on_sensor_val[0] > self._act_val_full_power:
                    self.opt_chp_now *= 2
                # set bool checkers
                emerg_cntrl = True  # emergency CHP activation
                self._emerg_on = True
                # adapt upper capacity
                self._adapt_cap(lower=False, check=True)

        # controller deactivation
        if self._deact_sgn == 'ge':
            if self.pv[0] > self.sp[0]:
                # set cv (DEactivate chp plant) and opt now (DEactivate all)
                self.cv = self._off_state
                self.opt_chp_now = self._off_state
                # set bool checkers
                emerg_cntrl = True
                self._emerg_on = False  # emergency CHP deactivation
                # adapt upper capacity
                self._adapt_cap(lower=False, check=True)
        else:  # if lesser
            if self.pv[0] < self.sp[0]:
                # set cv (DEactivate chp plant) and opt now (DEactivate all)
                self.cv = self._off_state
                self.opt_chp_now = self._off_state
                # set bool checkers
                emerg_cntrl = True
                self._emerg_on = False  # emergency CHP deactivation
                # adapt lower capacity
                self._adapt_cap(lower=True, check=True)

        # if first activation of emergency control after a time without it
        # activating, set checker for consecutive emergency control in next
        # steps. if already consecutive checked, then do nothing.
        if emerg_cntrl and not self._emerg_cnsctv:
            self._emerg_cnsctv = True
        elif not emerg_cntrl:  # if no more emerg, deact. consecutive check
            self._emerg_cnsctv = False

        # check if opt. third control step. Second is always off when emergency
        # is active, but third is on when pos. emergency is active and off when
        # neg. emergency:
        thrd_step = (emerg_cntrl and self._emerg_on) or not emerg_cntrl

        return emerg_cntrl, thrd_step

    def _adapt_cap(self, lower, check=True):
        if check:  # check if consecutive emergency control?
            # don't adapt if consecutive, to avoid adapting too frequently
            if self._emerg_cnsctv:  # if consecutive, return without adj.
                return None
        # calculate current TES inner energy
        tes_cap_now = self._tes_heat_content()
        # adapt lower or upper cap?
        if lower:
            self._tes_caps_adpt[0] = tes_cap_now
            self._tes_cap_adpt_lo = tes_cap_now  # backup this newly adj. value
            # save time of last adaption:
            self._time_last_cap_adpt[0] = self._models.time_sim
            cid = -2  # adapt lower cap ID
        else:
            self._tes_caps_adpt[1] = tes_cap_now
            self._tes_cap_adpt_hi = tes_cap_now  # backup this newly adj. value
            # save time of last adaption:
            self._time_last_cap_adpt[1] = self._models.time_sim
            cid = -1  # adapt lower cap ID

        # store adaption results to list of tuples: (timestep, id, (lo, hi))
        self._tes_caps_res.append(
            (self._models.time_sim, cid, *self._tes_caps_adpt)
        )
        # set bool checker for adapted caps to True
        self._caps_adapted = True

    def _longterm_opt_pred(self):
        # save step (which is effectively also the index) and the time of the
        # current optimization
        self._opt_steptime[self.num_opts, :] = (
            self.num_opts,
            self._models.time_sim,
        )
        # calculate current TES SOC
        self._tes_soc_now = self._tes_heat_content()
        # self._tes_soc_now = (
        #     self._tes_temp * self._tes_const).sum() * self._unit_fac
        # calculate TES SOC till empty (soc min) and full (soc max):
        self.tes_soc_min = self._tes_soc_now - self._tes_caps_adpt[0]
        self.tes_soc_max = self._tes_caps_adpt[1] - self._tes_soc_now
        # store these values for later access:
        self._tes_socs.append(
            (
                self._models.time_sim,
                self._tes_soc_now,
                self.tes_soc_min,
                self.tes_soc_max,
            )
        )
        # and save them for access to check for errors...
        # extract data in the correct length (optsize) and at the
        # correct time (num opts, since profiles are resampled to
        # opt_every) from df. save heat_dmd to instance to allow access
        # in constraint functions:
        pel_baseload, pel_user_ms, self._heat_dmd = (
            self._opt_profiles[self.num_opts : self.num_opts + self._optsize]
            .loc[:, ['pel_baseload', 'pel_user', 'heat_dmd_kum']]
            .T.values
        )
        # start current kumulative heat demand from current timestep:
        self._heat_dmd -= self._heat_dmd[0]

        # set optimization starting points to follow the power profile
        # as closely as possible
        x_init = (pel_user_ms + pel_baseload) / self.pel_chp_max
        x_init[x_init > 1.0] = 1.0  # limit to 0-1 range
        x_init[x_init < self._modrange_chp[0]] = 0.0
        # pre-optimize x-init:
        # x_init = self._preopt_x0(
        #     x_init, constraint_tes_min, constraint_tes_max)
        x_init_po = self._preopt_x0(x_init)

        # start optimizer
        self._res_optimizer = _sopt.minimize(
            ModelPredCHP._minimizer_input,
            x0=x_init_po,
            method='SLSQP',
            args=(
                pel_baseload,
                pel_user_ms,
                self.pel_chp_max,
                self.eta_el,
                self._mrc,
                self._freq_fac,
                self._flat_instandh,
                self._gaspreis,
                self._esr_gas,
                self._vnneg,
                self._eex,
                self._spz,
                self._awvk,
                self._instandh_cost,
                self._pwvk,
                self._ppic,
            ),
            # cut bounds to the size which fits the remaining array size
            bounds=self._opt_bounds[: pel_baseload.size],
            constraints=self._opt_constraints,
            # increase ftol to adjust it for the large values of the cost
            # function (up to -10k)
            options={'maxiter': 200, 'ftol': 2e-3, 'disp': False},
        )

        # get size of opt array to be able to broadcast data even
        # if the last opt steps are shorter:
        szopt = self._res_optimizer.x.size

        # adjust results by Pel to deal with badly scaled CHP plants.
        # save post processed optimization results, index to szopt to
        # broadcast cases where size shrinks (end of sim)
        self._opt_chp_mod_post[
            self.num_opts, :szopt
        ] = self._longterm_adj_by_soc(self._res_optimizer.x)

        # check for opt status:
        # - 0: success
        # - 3: too many iterations. accept even if not perfect
        # - 4: ineq constraints not compatible. may happen if CHP Pth, Pel
        #      and dmds are not scaled in a reasonable range relative to
        #      each other -> good, since not a problem of the opt, but of
        #      the chosen boundary conditions.
        # - 8: pos dir. der. ls. (result out of bounds, f.i. when CHP Pth
        #      is lower than heat dmd, then optimum is at x>1) -->also good
        # - 9: iter limit exceeded: still accept it, even if not perfect
        if self._res_optimizer.status in (0, 3, 4, 8, 9):
            # save optimization results, index to szopt to broadcast cases
            # where size shrinks (end of sim)
            self._opt_chp_mod[
                self.num_opts, :szopt
            ] = self._res_optimizer.x.copy()
            self._opt_status[self.num_opts] = self._res_optimizer.status
            self._opt_cost[self.num_opts] = self._res_optimizer.fun
            # extract post optimization result for the current opt step until
            # next opt:
            # self.opt_chp_now = self._res_optimizer.x[0]
            self.opt_chp_now = self._opt_chp_mod_post[self.num_opts, 0]
            # limit to 0-1 range AND to the modulation range. Also try to bring
            # continuity to the opt. if selected
            self._longterm_clip_w_cont()
        else:
            # IF NO GOOD OPT WAS FOUND
            # set CHP mode for current step based on current pel load
            self.opt_chp_now = (  # as a single-cell array to allow adjusting
                pel_user_ms[0:1] + pel_baseload[0:1]
            ) / self.pel_chp_max
            # adjust by Pel
            self.opt_chp_now = self._longterm_adj_by_soc(self.opt_chp_now)[0]
            # limit to 0-1 range AND to the modulation range. Also try to bring
            # continuity to the opt. if selected
            self._longterm_clip_w_cont()
            # set opt values to nan and store result to post proc. opt.
            self._opt_chp_mod[self.num_opts, :] = np.nan
            self._opt_chp_mod_post[self.num_opts, 0] = self.opt_chp_now
            # set the optimization status to 99
            self._opt_status[self.num_opts] = 99
            # get current cost with explicitly set x value
            self._opt_cost[self.num_opts] = ModelPredCHP._minimizer_input(
                np.array([self.opt_chp_now]),
                pel_baseload[0:1],
                pel_user_ms[0:1],
                self.pel_chp_max,
                self.eta_el,
                self._mrc,
                self._freq_fac,
                self._flat_instandh,
                self._gaspreis,
                self._esr_gas,
                self._vnneg,
                self._eex,
                self._spz,
                self._awvk,
                self._instandh_cost,
                self._pwvk,
                self._ppic,
            )

    def _longterm_clip_array(self, x, atol, copy=False):
        if copy:
            x = x.copy()
        # clip results to modulation range, including abs. tolerance
        x[x < (self._mrc[0] - atol)] = 0.0
        x[(x < self._mrc[0]) & (x != 0.0)] = 0.5
        x[x > 1.0] = 1.0
        if copy:
            return x

    def _longterm_adj_by_soc(self, x, atol=1e-2):
        # save x size to index SOC values
        xsz = x.size
        # clip a copy of the results to mod. range for TES SOC evaluation
        x_clppd = self._longterm_clip_array(x, atol, copy=True)
        # check if TES is overfull in any step and if so, skip the rest of this
        # method and return a copy of the original opt. output
        if np.any(self.tes_cap_before_full(x_clppd)[:xsz] < 0.0):
            return x.copy()
        # loop over TES SOC for max. 5 steps and increase mod.
        i = 0
        # unclipped version as result array
        new_x_unclp = x.copy()
        # clipped to mod range version to calculate TES SOCs
        new_x = new_x_unclp.copy()
        while (i < 5) and np.any(self.tes_cap_before_empty(new_x)[:xsz] < 0.0):
            # if so, scale and clip the optimization results stepwise until
            # max is almost 1.
            # x_clppd /= (x_clppd.max()**.5)
            # unclipped new x for scaling in each step
            new_x_unclp /= new_x_unclp.max() ** 0.5
            # clip for TES SOC calculation
            new_x = self._longterm_clip_array(new_x_unclp, atol, copy=True)
            i += 1
        # if still any below 0, simply set max val to 1
        if np.any(self.tes_cap_before_empty(new_x)[:xsz] < 0.0):
            new_x_unclp /= new_x_unclp.max()
            new_x = self._longterm_clip_array(new_x_unclp, atol, copy=True)
        # if TES __full__ is below 0 in any step, revert all changes!
        if np.any(self.tes_cap_before_full(new_x)[:xsz] < 0.0):
            new_x_unclp = x.copy()
        # and return (while clipping to 0,1 for safety)
        return np.clip(new_x_unclp, 0, 1)

    def _longterm_clip_w_cont(self, atol=1e-2):
        # limit to 0-1 range AND to the modulation range. Either
        # set it to zero or lower mod range limit, depending on which
        # value is closer.

        # Also try to bring some continuity to the optimization by making
        # the choice depending on the last opt step result
        # (could be refarctored by:
        #    if not self._inc_cont or self._lt_prev_opt >= self._mrc[0]:
        #  but it is much more self explaining like this...)
        if self._inc_cont:
            if self._lt_prev_opt >= self._mrc[0]:
                self.opt_chp_now = (
                    self.opt_chp_now
                    if self._mrc[0] <= self.opt_chp_now <= 1.0
                    else 1.0
                    if self.opt_chp_now > 1.0
                    else self._mrc[0]
                    if self.opt_chp_now >= self._mrc[0] / 2
                    else 0.0
                )
            else:
                self.opt_chp_now = (
                    self.opt_chp_now
                    if self._mrc[0] <= self.opt_chp_now <= 1.0
                    else 1.0
                    if self.opt_chp_now > 1.0
                    # if not in mod.-range but less than atol lower than range
                    else self._mrc[0]
                    if self.opt_chp_now
                    >= (self._mrc[0] - atol)  # add abs. tol. for this clipping
                    else 0.0
                )
        else:
            self.opt_chp_now = (
                self.opt_chp_now
                if self._mrc[0] <= self.opt_chp_now <= 1.0
                else 1.0
                if self.opt_chp_now > 1.0
                else self._mrc[0]
                if self.opt_chp_now >= self._mrc[0] / 2
                else 0.0
            )

    def _longterm_finalize_step(self):
        # increase optimization counter:
        self.num_opts += 1
        # now pop the last opt step element to mark this as done:
        self._opt_steps.pop(-1)
        # and also readjust caps:
        self._restore_adptd_caps()
        # save this step's post processed opt. result to enable bringing
        # continuity to the control
        self._lt_prev_opt = self.opt_chp_now

    def _restore_adptd_caps(self):
        # skip if no caps have been adapted (or already fully reverted)
        if not self._caps_adapted:
            return None
        # how long is it, that the caps were adapted?
        self._time_since_cap_adpt = (
            self._models.time_sim - self._time_last_cap_adpt
        )
        # how long, as a factor of the time it takes to restore the caps?
        self._fctr_cap_rstr = (
            self._time_since_cap_adpt / self._cap_restore_time
        )
        # if all are restored, revert bool checker and return
        if np.all(self._fctr_cap_rstr >= 1.0):
            self._caps_adapted = False
            return None
        # just for safety: limit factor to 1 to avoid over-compensating:
        self._fctr_cap_rstr[self._fctr_cap_rstr > 1.0] = 1.0

        # restore lower and upper caps:
        self._tes_caps_adpt[0] = (
            self._tes_cap_adpt_lo * (1.0 - self._fctr_cap_rstr[0])
            + self._tes_caps[0] * self._fctr_cap_rstr[0]
        )
        self._tes_caps_adpt[1] = (
            self._tes_cap_adpt_hi * (1.0 - self._fctr_cap_rstr[1])
            + self._tes_caps[1] * self._fctr_cap_rstr[1]
        )

        # store restore results to list of tuples: (timestep, id, lo, hi)
        self._tes_caps_res.append(
            (self._models.time_sim, 1.0, *self._tes_caps_adpt)
        )

    def _shortterm_opt(self):
        # extract data to reduce the amount of repeated slicing. this time
        # the control dataframe is used for in-step control. since the
        # index is in second-periods, indexing with the rounded int index
        # is ok.
        pel_baseload_ctrl, pel_user_ms_ctrl, = (
            self._ctrl_profiles.iloc[int(round(self._models.time_sim))]
            .loc[['pel_baseload', 'pel_user']]
            .values
        )
        # make a bounded scalar optimization over the cost function of the
        # current real demand
        self._res_opt_step3 = _sopt.minimize_scalar(
            ModelPredCHP._minimizer_input_scalar,
            method='bounded',
            args=(
                pel_baseload_ctrl,
                pel_user_ms_ctrl,
                self.pel_chp_max,
                self.eta_el,
                self._mrc,
                self._freq_fac,
                self._flat_instandh,
                self._gaspreis,
                self._esr_gas,
                self._vnneg,
                self._eex,
                self._spz,
                self._awvk,
                self._instandh_cost,
                self._pwvk,
                self._ppic,
            ),
            bounds=self._mrc,
            # increase xatol to adjust it for the value range of the
            # cost function (up to -50) and to reduce calculation time
            # (from 1e-5 -> 1e-4 23% speed increase, 1e-5 -> 2e-3 50%)
            options={'maxiter': 50, 'xatol': 2e-3, 'disp': False},
        )

    def run_controller(self, timestep):
        # FIRST CONTROL STEP: Ascertain heat supply. Running constantly.
        emerg_cntrl, thrd_step = self._emergency_control(
            hysteresis=self._emerg_use_hyst
        )

        # SECOND CONTROL STEP: When to run the CHP plant? Running periodically.
        # if emergency control was not active AND another opt_every seconds
        # have passed, run the optimization!
        if not emerg_cntrl and self._models.time_sim >= self._opt_steps[-1]:
            # run long term optimization
            self._longterm_opt_pred()
            # finalize step
            self._longterm_finalize_step()
        elif self._models.time_sim >= self._opt_steps[-1]:
            # if emergncy control but still next opt step reached. save status
            # with 100 is emerg activation and -100 for deact. This only
            # considers the last step, so if there were emerg. changes in
            # between, this will be neglected!
            self._opt_status[self.num_opts] = 100 if self._emerg_on else -100
            # still finalize step, since counter etc. still need to increase
            # to avoid a delay in opt. profiles
            self._longterm_finalize_step()

        # THIRD CONTROL STEP: direct opt. following Pel. Running constantly.
        # only run opt ctrl if CHP modelpred is on and no OR positive
        # emergency control
        if thrd_step and self.opt_chp_now > 0.0:
            # run short term optimization
            self._shortterm_opt()
            # since this does not opt. for heat loads, make a mix of the long
            # and short term optimization result and cache to variable for
            # clean/unaltered access in next step. If NO emerg. control act.,
            # then take 2/3 shortterm result and 1/3 longterm result,
            # else IF emerg. control take 50:50.
            if not emerg_cntrl:  # NO emerg. control activation
                self._cv_prev_opt = (
                    self._res_opt_step3.x * 2 + self.opt_chp_now
                ) / 3
            else:  # emerg. control ACTivation (not valid for deact.!)
                self._cv_prev_opt = (
                    self._res_opt_step3.x + self.opt_chp_now
                ) / 2
            self.cv = self._cv_prev_opt
        elif not emerg_cntrl:  # if step 2 opt is zero but no emergncy cntrl
            # else just set it to zero to deactivate the CHP plant
            self._cv_prev_opt = 0.0
            self.cv = 0.0
        # else for emergency control DEACTIVATION do nothing

        # clip value to max/min:
        self._clip_cv_to_saturation()

        # allow direction inversion:
        self._cv_invert()

        # check if controller is depending on another controller:
        self._subcontroller_check()

    def _preopt_x0(self, x0):
        # copy to avoid altering input
        x0_opt = x0.copy()
        # where is the min or max constraint violated?
        cons_min_vltd = self.tes_cap_before_empty(x0_opt) < 0
        cons_max_vltd = self.tes_cap_before_full(x0_opt) < 0
        # if any violations, optimize
        if np.any(cons_max_vltd) or np.any(cons_min_vltd):
            i = 0
            # loop while any violations exit
            while np.any(self.tes_cap_before_full(x0_opt) < 0) or np.any(
                self.tes_cap_before_empty(x0_opt) < 0
            ):
                # optimize where tes min violation
                if np.any(self.tes_cap_before_empty(x0_opt) < 0):
                    # where is tes min violated? here not compared to fully
                    # empty (<0), but instead with a safety factor so that
                    # lower cap than 10% of the tes is considererd as empty.
                    cmiv = self.tes_cap_before_empty(x0_opt) < (
                        self.tes_soc_min * 0.1
                    )  # safety factor
                    x0_opt[cmiv] *= 1.05
                    x0_opt[(x0_opt < 0.5) & cmiv] = 0.5
                    x0_opt[x0_opt > 1.0] = 1.0
                # optimize where tes max violation
                if np.any(self.tes_cap_before_full(x0_opt) < 0):
                    # where is tes max violated?
                    xio_cmv = x0_opt[cons_max_vltd]
                    xinit_vltd_as = np.argsort(xio_cmv)
                    minus_arr = np.linspace(-0.2, -0.01, xinit_vltd_as.size)
                    xio_cmv[xinit_vltd_as] = xio_cmv[xinit_vltd_as] + minus_arr
                    xio_cmv[xio_cmv < 0.5] = 0.0
                    x0_opt[cons_max_vltd] = xio_cmv
                i += 1
                # stop if more than 100 iterations and leave the opt fully to
                # the real optimizer
                if i > 100:
                    return x0
        return x0_opt

    @staticmethod
    def _minimizer_input(
        x,
        pel_baseload,
        pel_dmd_user_ms,
        pel_chp_max,
        eta_el,
        modrange_chp=(0.5, 1.0),
        freq_fac=0.25,
        flat_instandh=True,
        gaspreis=3.84,
        esteuerrueck_gas=0.55,
        netzneg=0.0,
        eex_baseload=3.745,
        strompreis_zukauf=25.0,
        aufschlag_verkauf=0.8,
        instandh_cost=69.0,
        penalty_wvk=0.0,
        penalty_pow_instandh=0.25,
        return_details=False,
    ):
        """
        Input to minimizer to minimize CHP cost.

        Heat is excluded, since it can be assumed that all heat is either
        directly consumed or stored until it can be consumed, with the heat
        prize being constant.

        All methods are static methods to enable calling them from outside the
        simulation environment for checking/tracing.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        pel_baseload : TYPE
            DESCRIPTION.
        pel_dmd_user_ms : TYPE
            DESCRIPTION.
        pel_chp_max : int, float
            Maximum electric power of the CHP plant in W.
        eta_el : TYPE
            DESCRIPTION.
        modrange_chp : TYPE, optional
            DESCRIPTION. The default is (.5, 1.).
        freq_fac : int, float, optional
            Frequency conversion factor for the cost calculation. The cost
            function is calculated for 1h values. If x is spaced in 15min
            steps, the cost funtion has to be converted to 15min. The freq_fac
            is x_resolution/1h. The default for 15min is .25.
        flat_instandh : TYPE, optional
            DESCRIPTION. The default is True.
        gaspreis : float, int, optional
            Gaspreis in ct/kWh. The default is 3.84. Source: Hilger/Isarwatt.
        esteuerrueck_gas : float, int, optional
            Einergiesteuerrückerstattung Erdgas in ct/kWh, nach
            § 53a Absatz (6) EnergieStG und  § 2 Absatz 3 Satz 1 Nummer 4.
            The default is 0.55.
        netzneg : float, int, optional
            Vermiedene Netzentgelte in ct/kWh. According to the new NEMoG,
            StromNEV and EnWG the vermiedene Netzentgelte will be non-existend
            following the 01.01.2023 (==0), see the default. Otherwise .388 is
            a value found in LITRES 2016 study. The default is 0.
        eex_baseload : float, int, optional
            EEX-Baseload price, also called KWK-Index, in ct/kWh. The default
            is 3.745 and corresponds to the value of Q3 2019.
        strompreis_zukauf : float, int, optional
            Strompreis in ct/kWh for buying electric power. The default
            is 25.
        instandh_cost : float, int, optional
            Upkeep cost of the CHP plant in ct/hour. **Only used, if
            `flat_instandh=True`**, else ASUE approximations.
            The default is 69.
        aufschlag_verkauf : float, int, optional
            Addition to the strompreis_zukauf in ct/kWh when reselling the
            bought el. power AND selling produced electrical power.
            The resulting selling price is
            strompreis_zukauf + aufschlag_verkauf.
            The default is .8.
        penalty_wvk : float, int, optional
            Penalty for reselling electric power in ct/kWh. The default is 0..
        penalty_pow_instandh : float, optional
            Penalty for lower modulation values for the Instandhaltungskosten.
            Vaues are penalized with modulation**penalty. Only used if
            flat_instand=False. The default is 0.25.
        return_details : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        cost.

        """
        # cut to valid range, leaving a 40% margin at lower modulation bound
        x[x < modrange_chp[0] * 0.6] = 0
        x[x > modrange_chp[1]] = 1

        # aktuelle BHKW-Leistung
        curr_chp_pow = x * pel_chp_max
        # aktuelle BHKW-Gasleistung
        pgas_chp_kW = curr_chp_pow / (eta_el * 1e3)
        # Zukauf wenn demand > Produktion
        # zuerst baseload, da hier reduzierte EEG Umlage
        pel_zukauf_bsld = pel_baseload - curr_chp_pow  # for baseload
        # bei demand < Prod, d.h. Einspeisung, ist Zukauf 0.
        pel_zukauf_bsld[pel_zukauf_bsld < 0.0] = 0.0
        # Eigennutzung für Baseload bei Produktion > bsld, sonst Prod.
        pel_eigen_bsld = np.where(
            curr_chp_pow > pel_baseload, pel_baseload, curr_chp_pow
        )

        # für alle anderen Abnehmer verbleibende CHP Prod:
        chp_pel_rest = curr_chp_pow - pel_eigen_bsld

        # Ueberschuss für Mieter aus Mieterstromprojekt, wenn
        # demand Mieter > (Produktion - Baseload):
        pel_zukauf_ms = pel_dmd_user_ms - chp_pel_rest
        # Wann ist demand kleiner als (Produktion-bsld) (Einspeisung)?
        mask_pel_einsp = pel_zukauf_ms < 0.0
        # Einspeisung ist negativer Zukauf BEI: demand < (Prod-bsld)
        pel_einsp = -pel_zukauf_ms.copy()
        pel_einsp[~mask_pel_einsp] = 0.0  # dmd > (Prod-bsld) -> 0
        # bei demand < (Prod-bsld), d.h. Einspeisung, ist Zukauf 0.
        pel_zukauf_ms[mask_pel_einsp] = 0.0
        # Eigennutzung ist demand wo (Prod-bsld) ausreicht, sonst
        # (Prod-bsld)
        pel_eigen_ms = np.where(
            chp_pel_rest > pel_dmd_user_ms, pel_dmd_user_ms, chp_pel_rest
        )

        # gesamter Zukauf:
        pel_zukauf_tot = pel_zukauf_bsld + pel_zukauf_ms
        # gesamte Eigennutzung:
        pel_eigen_tot = pel_eigen_bsld + pel_eigen_ms

        # Verguetung für Eigennutzung und Einspeisung nach KWKG,
        # kwkg_nach_pmax berechnet Staffelung nach pel_chp_max,
        # ansonsten Staffelung nach aktueller Modulation
        kwkg_nach_pmax = False
        if kwkg_nach_pmax:
            kwkg = ModelPredCHP._kwk_zuschlag(
                np.full(x.shape, pel_chp_max / 1e3),
                ersatz_alt=False,
                as_sum=True,
            )
            # Berechnung in ct/kWh für aktuelle einspeisung/eigennutz:
            kwkg_eigen = kwkg['eig'] / pel_chp_max * pel_eigen_tot
            kwkg_einsp = kwkg['esp'] / pel_chp_max * pel_einsp
        else:
            kwkg_eigen = ModelPredCHP._kwk_zuschlag(
                pel_eigen_tot / 1e3, ersatz_alt=False, as_sum=True
            )['eig']
            kwkg_einsp = ModelPredCHP._kwk_zuschlag(
                pel_einsp / 1e3, ersatz_alt=False, as_sum=True
            )['esp']

        # get cost function:
        strompreis_verkauf = strompreis_zukauf + 0.8
        earnings = {
            'gas': -pgas_chp_kW * gaspreis * freq_fac,
            'esteuer_rueck': (pgas_chp_kW * esteuerrueck_gas * freq_fac),
            'vnetzentg': (pgas_chp_kW * netzneg * freq_fac),
            'instandh': (
                # scaling by root of modulation to penalize
                # modulated service.
                -(x ** penalty_pow_instandh)
                * pel_chp_max
                / 1e3
                * ModelPredCHP._instandh_chp_nonflat(
                    pel_chp_max / 1e3, verfahren='ASUE'
                )
                * freq_fac
                if not flat_instandh
                else np.full(x.shape, -instandh_cost * freq_fac)
            ),
            # EEG Umlage wird zusammengesetzt aus red. EEGU für
            # Baseload und voller EEGU für restlichen Strom
            'eeg_umlage': freq_fac
            / 1e3
            * (
                -pel_eigen_bsld
                * ModelPredCHP._eeg_umlage(
                    pel_chp_max, jahr=2020, verbraucher='eigen'
                )
                - chp_pel_rest
                * ModelPredCHP._eeg_umlage(
                    pel_chp_max, jahr=2020, verbraucher='dritte'
                )
            ),
            # Zukauf für Baseload und Mieterstrom-Teilnehmer.
            # nicht-Teilnehmer sind nicht integriert.
            'zukauf_strom': (
                -pel_zukauf_tot / 1e3 * strompreis_zukauf * freq_fac
            ),
            'weiterverk_strom': (
                pel_zukauf_tot
                / 1e3
                * (strompreis_verkauf - penalty_wvk)
                * freq_fac
            ),
            'einspeis_kwkg': kwkg_einsp * freq_fac,
            'einspeis_eex': pel_einsp / 1e3 * eex_baseload * freq_fac,
            'eigennutz_kwkg': kwkg_eigen * freq_fac,
            'eigennutz_verk': (
                pel_eigen_tot / 1e3 * strompreis_verkauf * freq_fac
            ),
            # 'heat': curr_chp_pow * 2 * 6 / 1e3 * freq_fac,
        }
        # if Vollwartungsvertrag mit 69ct/Betriebsstunde
        if flat_instandh:
            earnings['instandh'][x == 0] = 0.0

        # get cost as negative of the sum of all earnings
        cost = -sum(earnings.values())
        if return_details:
            return earnings, cost
        else:
            return np.sum(cost)

    @staticmethod
    def _minimizer_input_scalar(
        x,
        pel_baseload,
        pel_dmd_user_ms,
        pel_chp_max,
        eta_el,
        modrange_chp=(0.5, 1.0),
        freq_fac=0.25,
        flat_instandh=True,
        gaspreis=3.84,
        esteuerrueck_gas=0.55,
        netzneg=0.0,
        eex_baseload=3.745,
        strompreis_zukauf=25.0,
        aufschlag_verkauf=0.8,
        instandh_cost=69.0,
        penalty_wvk=0.0,
        penalty_pow_instandh=0.25,
        return_details=False,
    ):
        """
        Input to minimizer to minimize CHP cost.

        Heat is excluded, since it can be assumed that all heat is either
        directly consumed or stored until it can be consumed, with the heat
        prize being constant.

        All methods are static methods to enable calling them from outside the
        simulation environment for checking/tracing.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        pel_baseload : TYPE
            DESCRIPTION.
        pel_dmd_user_ms : TYPE
            DESCRIPTION.
        pel_chp_max : int, float
            Maximum electric power of the CHP plant in W.
        eta_el : TYPE
            DESCRIPTION.
        modrange_chp : TYPE, optional
            DESCRIPTION. The default is (.5, 1.).
        freq_fac : int, float, optional
            Frequency conversion factor for the cost calculation. The cost
            function is calculated for 1h values. If x is spaced in 15min
            steps, the cost funtion has to be converted to 15min. The freq_fac
            is x_resolution/1h. The default for 15min is .25.
        flat_instandh : TYPE, optional
            DESCRIPTION. The default is True.
        gaspreis : float, int, optional
            Gaspreis in ct/kWh. The default is 3.84. Source: Hilger/Isarwatt.
        esteuerrueck_gas : float, int, optional
            Einergiesteuerrückerstattung Erdgas in ct/kWh, nach
            § 53a Absatz (6) EnergieStG und  § 2 Absatz 3 Satz 1 Nummer 4.
            The default is 0.55.
        netzneg : float, int, optional
            Vermiedene Netzentgelte in ct/kWh. According to the new NEMoG,
            StromNEV and EnWG the vermiedene Netzentgelte will be non-existend
            following the 01.01.2023 (==0), see the default. Otherwise .388 is
            a value found in LITRES 2016 study. The default is 0.
        eex_baseload : float, int, optional
            EEX-Baseload price, also called KWK-Index, in ct/kWh. The default
            is 3.745 and corresponds to the value of Q3 2019.
        strompreis_zukauf : float, int, optional
            Strompreis in ct/kWh for buying electric power. The default
            is 25.
        instandh_cost : float, int, optional
            Upkeep cost of the CHP plant in ct/hour. **Only used, if
            `flat_instandh=True`**, else ASUE approximations.
            The default is 69.
        aufschlag_verkauf : float, int, optional
            Addition to the strompreis_zukauf in ct/kWh when reselling the
            bought el. power AND selling produced electrical power.
            The resulting selling price is
            strompreis_zukauf + aufschlag_verkauf.
            The default is .8.
        penalty_wvk : float, int, optional
            Penalty for reselling electric power in ct/kWh. The default is 0..
        penalty_pow_instandh : float, optional
            Penalty for lower modulation values for the Instandhaltungskosten.
            Vaues are penalized with modulation**penalty. Only used if
            flat_instand=False. The default is 0.25.
        return_details : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        cost.

        """
        # aktuelle BHKW-Leistung
        curr_chp_pow = x * pel_chp_max
        # aktuelle BHKW-Gasleistung
        pgas_chp_kW = curr_chp_pow / (eta_el * 1e3)
        # Zukauf wenn demand > Produktion
        # zuerst baseload, da hier reduzierte EEG Umlage
        pel_zukauf_bsld = pel_baseload - curr_chp_pow  # for baseload
        # bei demand < Prod, d.h. Einspeisung, ist Zukauf 0.
        pel_zukauf_bsld = 0.0 if pel_zukauf_bsld < 0.0 else pel_zukauf_bsld
        # Eigennutzung für Baseload bei Produktion > bsld, sonst Prod.
        pel_eigen_bsld = (
            pel_baseload if curr_chp_pow > pel_baseload else curr_chp_pow
        )

        # für alle anderen Abnehmer verbleibende CHP Prod:
        chp_pel_rest = curr_chp_pow - pel_eigen_bsld

        # Ueberschuss für Mieter aus Mieterstromprojekt, wenn
        # demand Mieter > (Produktion - Baseload):
        pel_zukauf_ms = pel_dmd_user_ms - chp_pel_rest
        # Wann ist demand kleiner als (Produktion-bsld) (Einspeisung)?
        mask_pel_einsp = pel_zukauf_ms < 0.0
        # Einspeisung ist negativer Zukauf BEI: demand < (Prod-bsld)
        pel_einsp = -pel_zukauf_ms
        # dmd > (Prod-bsld) -> 0
        pel_einsp = 0.0 if not mask_pel_einsp else pel_einsp
        # bei demand < (Prod-bsld), d.h. Einspeisung, ist Zukauf 0.
        pel_zukauf_ms = 0.0 if mask_pel_einsp else pel_zukauf_ms
        # Eigennutzung ist demand wo (Prod-bsld) ausreicht, sonst
        # (Prod-bsld)
        pel_eigen_ms = (
            pel_dmd_user_ms if chp_pel_rest > pel_dmd_user_ms else chp_pel_rest
        )

        # gesamter Zukauf:
        pel_zukauf_tot = pel_zukauf_bsld + pel_zukauf_ms
        # gesamte Eigennutzung:
        pel_eigen_tot = pel_eigen_bsld + pel_eigen_ms

        # Verguetung für Eigennutzung und Einspeisung nach KWKG,
        # kwkg_nach_pmax berechnet Staffelung nach pel_chp_max,
        # ansonsten Staffelung nach aktueller Modulation
        kwkg_nach_pmax = False
        if kwkg_nach_pmax:
            kwkg = ModelPredCHP._kwk_zuschlag(
                np.array([pel_chp_max]), ersatz_alt=False, as_sum=True
            )
            # Berechnung in ct/kWh für aktuelle einspeisung/eigennutz:
            kwkg_eigen = kwkg['eig'] / pel_chp_max * pel_eigen_tot
            kwkg_einsp = kwkg['esp'] / pel_chp_max * pel_einsp
        else:
            kwkg_eigen = ModelPredCHP._kwk_zuschlag(
                np.array([pel_eigen_tot / 1e3]), ersatz_alt=False, as_sum=True
            )['eig']
            kwkg_einsp = ModelPredCHP._kwk_zuschlag(
                np.array([pel_einsp / 1e3]), ersatz_alt=False, as_sum=True
            )['esp']

        # get cost function:
        strompreis_verkauf = strompreis_zukauf + 0.8
        earnings = {
            'gas': -pgas_chp_kW * gaspreis * freq_fac,
            'esteuer_rueck': (pgas_chp_kW * esteuerrueck_gas * freq_fac),
            'vnetzentg': (pgas_chp_kW * netzneg * freq_fac),
            'instandh': (
                # scaling by root of modulation to penalize
                # modulated service.
                -(x ** penalty_pow_instandh)
                * pel_chp_max
                / 1e3
                * ModelPredCHP._instandh_chp_nonflat(
                    pel_chp_max / 1e3, verfahren='ASUE'
                )
                * freq_fac
                if not flat_instandh
                else -instandh_cost * freq_fac
            ),
            # EEG Umlage wird zusammengesetzt aus red. EEGU für
            # Baseload und voller EEGU für restlichen Strom
            'eeg_umlage': freq_fac
            / 1e3
            * (
                -pel_eigen_bsld
                * ModelPredCHP._eeg_umlage(
                    pel_chp_max, jahr=2020, verbraucher='eigen'
                )
                - chp_pel_rest
                * ModelPredCHP._eeg_umlage(
                    pel_chp_max, jahr=2020, verbraucher='dritte'
                )
            ),
            # Zukauf für Baseload und Mieterstrom-Teilnehmer.
            # nicht-Teilnehmer sind nicht integriert.
            'zukauf_strom': (
                -pel_zukauf_tot / 1e3 * strompreis_zukauf * freq_fac
            ),
            'weiterverk_strom': (
                pel_zukauf_tot
                / 1e3
                * (strompreis_verkauf - penalty_wvk)
                * freq_fac
            ),
            'einspeis_kwkg': kwkg_einsp * freq_fac,
            'einspeis_eex': pel_einsp / 1e3 * eex_baseload * freq_fac,
            'eigennutz_kwkg': kwkg_eigen * freq_fac,
            'eigennutz_verk': (
                pel_eigen_tot / 1e3 * strompreis_verkauf * freq_fac
            ),
            # 'heat': curr_chp_pow * 2 * 6 / 1e3 * freq_fac,
        }
        # if Vollwartungsvertrag mit 69ct/Betriebsstunde
        if flat_instandh:
            earnings['instandh'] = 0.0 if x == 0 else earnings['instandh']

        # get cost as negative of the sum of all earnings
        cost = -sum(earnings.values())
        if return_details:
            return earnings, cost
        else:
            return sum(cost)

    @staticmethod
    def _kwk_zuschlag(
        pel_chp, as_sum=False, ersatz_alt=False, strompreis_negativ=False
    ):
        """
        Calculate KWK Zuschlag according to KWKG Novelle 2016.

        See finances_chp.py for more information.

        Parameters
        ----------
        pel_chp : TYPE
            DESCRIPTION.
        as_sum : TYPE, optional
            DESCRIPTION. The default is False.
        ersatz_alt : TYPE, optional
            DESCRIPTION. The default is False.
        strompreis_negativ : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        dict
            Einspeise- und Eigennutzungsvergütung.

        """
        if isinstance(pel_chp, (int, float)):
            # call singular value implementation
            _pel_chp = np.array(pel_chp)
        # else check that np.ndarray
        assert isinstance(pel_chp, np.ndarray)
        _pel_chp = np.atleast_1d(pel_chp).copy()

        # Grenzen nach KWKG §7
        kwkg_steps = np.array([0.0, 50.0, 100.0, 250.0, 2000.0, np.inf])
        kwkg_es_verg = np.array(
            [8.0, 6.0, 5.0, 4.4, 3.1]
        )  # Einspeisungsvergütung
        kwkg_en_verg = np.array(
            [4.0, 3.0, 2.0, 1.5, 1.0]
        )  # Eigennutzungsvergütung

        # lidx replaced by just using the max. shape...
        # get highest step number for required shape:
        # lidx = np.argwhere(_pel_chp.max() > kwkg_steps)[-1][0]
        # make array in the correct shape
        pel_per_step = np.zeros((_pel_chp.shape[0], 5))
        # fill each row with the pel chp value
        pel_per_step[:] = _pel_chp[:, None]
        # subtract the step values
        pel_per_step -= kwkg_steps[:-1][None, :]
        # set to zero where lower than zero
        pel_per_step[pel_per_step < 0] = 0.0
        # get difference between steps
        kwkg_clp_steps = np.diff(kwkg_steps)
        # limit each column to the difference between the steps
        pel_per_step = np.clip(pel_per_step, 0.0, kwkg_clp_steps)

        # multiply with the ct/kWh for each step to get the
        kwk_einsp = pel_per_step * kwkg_es_verg
        kwk_eigen = pel_per_step * kwkg_en_verg

        if as_sum:
            kwk_einsp = kwk_einsp.sum(axis=1)
            kwk_eigen = kwk_eigen.sum(axis=1)

        return {'esp': kwk_einsp, 'eig': kwk_eigen}

    @staticmethod
    def _instandh_chp_nonflat(pel_chp_max, verfahren='ASUE'):
        """
        Calculate die Netto-Instandhaltungskosten of the CHP.

        Wird das Verfahren `ASUE` gewählt, werden die
        Netto-Instandhaltungskosten in [Cent/kWh]
        ausgegeben.
        Dieses ist gültig für BHKW zwischen 10kW und 100kW elektrischer
        Leistung. Bei `VDI2067` werden die jährlichen
        (Netto?-)Instandhaltungskosten unabhängig von der erzeugten Energiemenge
        berechnet.

        Parameters:
        -----------
        pel_chp_max : ndarray, float, int
            Elektrische Nennleistung des BHKW.
        verfahren : string, optional
            Verfahren, nach dem die Instandhaltungskosten berechnet werden sollen.
            Mögliche Optionen: `ASUE`, `VDI2067`. Default: `ASUE`

        Returns:
        --------
        instandhaltung_chp : float, np.ndarray
            Netto-Instandhaltungskosten des BHKWs in Cent/kWh.

        Referenz:
        ---------
        ASUE 2014:
            BHKW Kenndaten 2014/2015
            https://asue.de/sites/default/files/asue/themen/blockheizkraftwerke/2014/broschueren/05_10_14_bhkw_kenndaten_leseprobe.pdf
        VDI 2067-1

        """
        if (verfahren != 'ASUE') and (verfahren != 'VDI2067'):
            print(
                'Kein bekanntes Verfahren zur Berechnung der',
                'Instandhaltungskosten gegeben! Es wird auf ASUE',
                'zurückgegriffen (Ausgabe der Kosten in [Cent/kWh]).',
            )
            verfahren = 'ASUE'

        if isinstance(pel_chp_max, (int, float)):
            if verfahren == 'ASUE':
                # Instandhaltungskosten in [Cent/kWh]:
                if 10 <= pel_chp_max <= 100:
                    instandhaltung_chp = 6.6626 * pel_chp_max ** (-0.25)
                elif 1 <= pel_chp_max < 10:
                    instandhaltung_chp = 3.2619 * pel_chp_max ** 0.1866
                else:  # Falls Leistung nicht zwischen 1 und 100kW
                    raise ValueError(
                        'pel_chp_max muss zwischen 1kW < pel < 100kw liegen.'
                    )
            elif verfahren == 'VDI2067':
                # Instandhaltungskosten pro Jahr. 6% der Investitionskosten.
                # Investitionskosten nach ASUE:
                instandhaltung_chp = (
                    0.06 * 5.438 * pel_chp_max ** (-0.351) * 1000 * pel_chp_max
                )
        else:
            if verfahren == 'ASUE':
                assert np.all(pel_chp_max > 1) and np.all(
                    pel_chp_max < 100
                ), 'pel_chp_max muss zwischen 1kW < pel < 100kw liegen.'
                # Instandhaltungskosten in [Cent/kWh]:
                instandhaltung_chp = np.where(
                    (10 <= pel_chp_max) & (pel_chp_max <= 100),
                    6.6626 * pel_chp_max ** (-0.25),
                    3.2619 * pel_chp_max ** 0.1866,
                )
                raise ValueError(
                    'pel_chp_max muss zwischen 1kW < pel < 100kw liegen.'
                )
            elif verfahren == 'VDI2067':
                # Instandhaltungskosten pro Jahr. 6% der Investitionskosten.
                # Investitionskosten nach ASUE:
                instandhaltung_chp = (
                    0.06 * 5.438 * pel_chp_max ** (-0.351) * 1000 * pel_chp_max
                )
        return instandhaltung_chp

    @staticmethod
    def _eeg_umlage(pel_chp, jahr=2020, verbraucher='dritte'):
        """
        EEG-Umlage in [Cent/kWh].

        Parameters:
        -----------
        Pel : ndarray, optional
            Leistung des BHKWs in [kW] als Array in Abhängigkeit der Modulation.
        Verbraucher : string, optional
            Art der Stromverbraucher. Mögliche Optionen sind: `Mieter` und `WEG`.
            Standard ist `Mieter`. Wird für die Berechnung der EEG-Umlage benötigt.
            Wird der Strom an `Mieter` als Letztverbraucher verkauft, so muss die
            volle EEG-Umlage auf den produzierten Strom gezahlt werden. Im Fall von
            `WEG` (Wohnungseigentümergemeinschaft) wird der Strom des BHKWs von den
            Eigentümern des BHKWs verbraucht, daher fällt hier nur die reduzierte
            EEG-Umlage an.

        Returns:
        --------
        float
            EEG-Umlage in [Cent/kWh].

        Referenz:
        ---------
        LITRES 2016:
            Mini-/Mikro-KWK in städtischen Energiesystemen - Eine Analyse von
            Herausforderungen und Erfolgsfaktoren.
        EEG-Novelle 2017:
            Anhebung der anteilig zu zahlenden EEG-Umlage auf 40%.
        Netzbetreiber 2017:
            Anhebung der EEG-Umlage auf 6.88 Cent/kWh.
        Unbestätigt nach bhkw-infozentrum.de:
            Modernisierung bereits befreiter Anlagen bewirkt eine nachfolgend
            20%-ig zu zahlende EEG-Umlage. Bei Erhöhung der KWK-Leistung durch
            Modernisierung folgt eine 40%-ige EEG-Umlage.
        """
        assert verbraucher in ('dritte', 'eigen')
        eeg_umlg = {
            2020: 6.756,
            2019: 6.405,
            2018: 6.792,
            2017: 6.88,
            2016: 6.354,
        }[jahr]

        if isinstance(pel_chp, (int, float)):
            if (pel_chp >= 100) or (verbraucher == 'dritte'):
                eeg_u = eeg_umlg
            elif (pel_chp <= 10) and (verbraucher == 'eigen'):
                eeg_u = 0.0
            else:
                eeg_u = eeg_umlg * 0.4
        else:
            if verbraucher == 'dritte':
                eeg_u = np.full((len(pel_chp),), eeg_umlg)
            else:  # Eigenverbrauch
                eeg_u = np.where(
                    pel_chp >= 100,
                    eeg_umlg,
                    np.where(pel_chp <= 10, 0.0, eeg_umlg * 0.4),
                )
        return eeg_u

    @staticmethod
    def _load_eex_baseload(quartal='Q3 2019', reload_xls=False):
        """
        Yield the EEX-Baseload price in [Cent/kWh].

        Returns
        -------
        float
            EEX-Baseload price in [Cent/kWh].

        Reference
        ---------
        37.6€/MWh: Stand Q4 2016
            https://www.eex.com/de/marktdaten/strom/spotmarkt/kwk-index
            neu:
            https://www.eex.com/de/marktdaten/strom/strom-indizes/kwk-index

        """
        assert isinstance(reload_xls, bool)
        if reload_xls and quartal != 'Q3 2019':  # slow but dynamic output
            df = pd.read_excel(
                './data_tables/phelix-quarterly-data-data-data.xls',
                header=3,
                index_col=0,
                squeeze=True,
            )
            df.index = df.index.str.strip()  # strip trailing whitespaces
            return df.loc[quartal] * 100 / 1e3
        # fast outputs for use in optimization
        elif quartal == 'Q3 2019':
            return 3.745
        else:
            raise NotImplementedError(
                'Quartal nicht als schnelle Implementierung verfügbar. '
                'Hardcoding oder langsame `reload_xls=True` setzen.'
            )

    @staticmethod  # staticmethod to allow access without having an instance
    def infer_instandh(chp_pel, inkl_mwst=False, large_customer_discount=0.95):
        """Infer the **flat** maintenance cost per hour."""
        # make a stepwise interpolation over the table values:
        pel_conti = np.linspace(4.0, 80.0, 100)
        cost_conti = np.interp(
            pel_conti,
            np.array([6.0, 9.0, 15.0, 20.0, 34.0, 50.0, 60.0, 70.0, 80.0]),
            (
                np.array([0.42, 0.49, 0.57, 0.72, 1.14, 1.65, 1.87, 2.0, 2.08])
                * large_customer_discount
                * 1e2
            ),
        )  # * 1e2 -> euro to ct

        # and now make a continuous cubic curve fit out of it
        def cubic_cost(x, a, b, c, d):
            return a * x ** 3 + b * x ** 2 + c * x + d

        # for faster fitting, set known results as initial values...
        popt, pcov = _sopt.curve_fit(
            cubic_cost,
            pel_conti,
            cost_conti,
            p0=[-5.878e-04, 6.248e-02, 7.641e-01, 3.347e-01],
        )

        # return the flat maintenance cost
        instandh_flat = cubic_cost(chp_pel / 1e3, *popt)
        if inkl_mwst:
            instandh_flat *= 1.19
        return instandh_flat
