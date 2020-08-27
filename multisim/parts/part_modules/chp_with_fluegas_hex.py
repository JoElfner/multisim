# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Nov 2019
"""

import numpy as np

from ... import all_parts as _ap
from ... import utility_functions as _ut


def add_chp_with_fghex(
    simenv,
    databases,
    number_rf_ports,
    chp_ff_connection,
    chp_ntrf_connection,
    chp_htrfA_connection,
    chp_htrfB_connection,
    segment_name='1',
    chp_kwds=dict(
        power_electrical=20e3,
        eta_el=0.32733,
        p2h_ratio=0.516796,
        modulation_range=(0.5, 1),
        theta_ff=83.5,
        heat_spread='range',
        heated_cells=(1, 5),
        no_control=False,
        lower_limit=0.0,
        upper_limit=1.0,
        fluegas_flow=70 / 61.1,
        pipe_specs={'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN80'}},
    ),
    chp_ff_pump_ulim=1.0,
    chp_ctrl=dict(
        ctrld_part='tes', on_sens_port=15, setpoint=70.0, off_val=75.0
    ),
    fluegashex_theta_out_water=55.0,
    ht_rf_theta_mix=65.0,
    adjust_pipes=True,
    Tamb=25,
    ctrl_chp_pump_by_ts=False,
    ctrl_hex_by_ts=False,
    ctrl_htrf_by_ts=False,
    fluegas_hex_kwds=dict(hex_regression_pipeline=5, max_gas_Nm3h=70.0),
    **kwds
):
    """
    Add a CHP plant with a condensing flue gas heat exchanger to simenv.


    Parameters
    ----------
    simenv : TYPE
        DESCRIPTION.
    databases : TYPE
        DESCRIPTION.
    chp_kwds : TYPE, optional
        DESCRIPTION. The default is dict(            power_electrical=20e3, modulation_range=(.5, 1),            heat_spread='range', heated_cells=(0, 4), no_control=False,            lower_limit=0., upper_limit=1.).
    fluegas_hex_kwds : TYPE, optional
        DESCRIPTION. The default is dict(hex_regression_pipeline=5).
    **kwds : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    assert isinstance(number_rf_ports, int)

    ps_dn25 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN25'}}
    ps_dn32 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN32'}}
    # ps_dn40 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN40'}}
    # ps_dn50 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN50'}}
    # ps_dn65 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN65'}}
    # ps_dn80 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN80'}}

    # scale all values by the CHP power ratio (to the validation chp th power
    # of 38.7kW)
    chp_power_factor = (
        chp_kwds['power_electrical'] / chp_kwds['p2h_ratio'] / 38.7e3
    )
    pspecs = {  # get ref pipe specs and mult factor for each part
        'pwp_ff': (ps_dn32, chp_power_factor),
        'p3v_rf': (ps_dn32, chp_power_factor),
        'p_rf_lt': (ps_dn25, chp_power_factor),
        'p3v_htrfA': (ps_dn25, chp_power_factor),
        'p_htrfB': (ps_dn25, chp_power_factor),
        'p_htrf': (ps_dn32, chp_power_factor),
        'p_rf': (ps_dn32, chp_power_factor),
        'hex_fg': (ps_dn32, chp_power_factor),
        'chp': (chp_kwds['pipe_specs'], chp_power_factor),
    }
    del chp_kwds['pipe_specs']

    if adjust_pipes:  # adjust pipes by A_i multiplier
        pspecs = _ut.adjust_pipes(pspecs)
    else:  # take pspecs without multiplier
        pspecs = {k: v[0] for k, v in pspecs.items()}

    # chp_pps = ps_dn80

    simenv.add_part(
        _ap.CHPPlant,
        name='CHP{0}'.format(segment_name),
        length=2.5,
        grid_points=6,
        **chp_kwds,
        s_ins=0.2,
        lambda_ins=0.035,
        T_init=71.0,
        T_amb=Tamb,
        material='carbon_steel',
        pipe_specs=pspecs['chp'],
        connect_fluegas_hex=True,
        fg_hex_name='hex_chp{0}_fg'.format(segment_name),
    )
    simenv.add_part(
        _ap.PipeWithPump,
        name='pwp_chp{0}_ff'.format(segment_name),
        length=4,
        grid_points=20,
        s_ins=0.05,
        lambda_ins=0.03,
        material='carbon_steel',
        pipe_specs=pspecs['pwp_ff'],
        T_init=np.linspace(71.3, 82.0, 20),
        T_amb=Tamb,
        start_massflow=0.0,
        lower_limit=0.0,
        upper_limit=chp_ff_pump_ulim,
        # max flow to cool 40kW at 7K spread
        maximum_flow=(
            chp_kwds['power_electrical']
            / chp_kwds['p2h_ratio']
            / 4180
            / (85 - 78)
        ),
    )
    simenv.connect_ports(
        'CHP{0}'.format(segment_name),
        'out',
        'pwp_chp{0}_ff'.format(segment_name),
        'in',
    )

    # design system depending on number of rf ports
    if number_rf_ports == 3:  # if three rf ports
        # RF from HEX to CHP with branch to incorporate HT return flow. Branch is
        # controlled 3W valve to emulate flow control valve over HEX.
        # Only for validation with ang data, else fast ctrl:
        # Controller should be P-Controller with setpoint 52-55 degree
        # (check what s best) Kp should be low, so that it always lags behind
        # (as the thermostate valve does).
        simenv.add_part(
            _ap.PipeWith3wValve,
            name='p3v_chp{0}_rf'.format(segment_name),
            length=2,
            grid_points=10,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p3v_rf'],
            T_init=np.hstack((np.full(3, 38.0), np.full(7, 52.5))),
            T_amb=Tamb,
            valve_location=3,
            ctrl_required=True,
            start_portA_opening=0.15,
            lower_limit=0.01,
            upper_limit=0.9,
        )
        simenv.connect_ports(
            'p3v_chp{0}_rf'.format(segment_name),
            'AB',
            'CHP{0}'.format(segment_name),
            'in',
        )
        # RF from outer system to HEX:
        simenv.add_part(
            _ap.Pipe,
            name='p_chp{0}_rf_lt'.format(segment_name),
            length=7,
            grid_points=30,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_rf_lt'],
            T_init=26.0,
            T_amb=Tamb,
        )
        # RF from TES HT port A (lower HT port) to mix with the RF from the HEX.
        # Valve mixes flow with HT RF port B. P-Controller with low Kp suits the
        # thermostatic valve best. Check for best mixing temperature
        simenv.add_part(
            _ap.PipeWith3wValve,
            name='p3v_chp{0}_rf_htA'.format(segment_name),
            length=7,
            grid_points=30,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p3v_htrfA'],
            T_init=np.hstack(
                # siehe Messdaten für Startwerte
                (
                    np.linspace(49.7, 51.7, 9),  # ht rf a von 49.7 auf 51.7
                    np.linspace(51.7, 54.2, 21),
                )
            ),  # hinter mix, dann auf 54.2
            T_amb=Tamb,
            valve_location=7,
            ctrl_required=True,
            start_portA_opening=0.3,
            lower_limit=0.0,
            upper_limit=1.0,
        )
        simenv.connect_ports(
            'p3v_chp{0}_rf_htA'.format(segment_name),
            'AB',
            'p3v_chp{0}_rf'.format(segment_name),
            'B',
        )
        # add pipe from TES HT RF B to mix with HT RF A
        simenv.add_part(  # Connects to TES and htrf_A
            _ap.Pipe,
            name='p3v_chp{0}_rf_htB'.format(segment_name),
            length=2,
            grid_points=10,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_htrfB'],
            T_init=51.5,
            T_amb=Tamb,
        )
        simenv.connect_ports(
            'p3v_chp{0}_rf_htB'.format(segment_name),
            'out',
            'p3v_chp{0}_rf_htA'.format(segment_name),
            'B',
        )
        # add flue gas hex:
        simenv.add_part(
            _ap.HEXCondPoly,
            name='hex_chp{0}_fg'.format(segment_name),
            material='carbon_steel',
            pipe_specs=pspecs['hex_fg'],
            T_init=50,
            T_amb=Tamb,
            **fluegas_hex_kwds,
            fluegas_flow_range=(0.5, 1.05),
            water_flow_range=(0.0, 1.05),
        )
        simenv.connect_ports(
            'p_chp{0}_rf_lt'.format(segment_name),
            'out',
            'hex_chp{0}_fg'.format(segment_name),
            'water_in',
        )
        simenv.connect_ports(
            'hex_chp{0}_fg'.format(segment_name),
            'water_out',
            'p3v_chp{0}_rf'.format(segment_name),
            'A',
        )
        # boundary conditions for flue gas hex:
        simenv.add_open_port(
            name='BC_fg{0}_in'.format(segment_name),
            constant=True,
            temperature=110,
        )
        simenv.add_open_port(
            name='BC_fg{0}_out'.format(segment_name),
            constant=True,
            temperature=50,
        )
        simenv.connect_ports(
            'BoundaryCondition',
            'BC_fg{0}_in'.format(segment_name),
            'hex_chp{0}_fg'.format(segment_name),
            'fluegas_in',
        )
        simenv.connect_ports(
            'BoundaryCondition',
            'BC_fg{0}_out'.format(segment_name),
            'hex_chp{0}_fg'.format(segment_name),
            'fluegas_out',
        )

        # connect sub-system to outer system:
        simenv.connect_ports(
            'pwp_chp{0}_ff'.format(segment_name),
            'out',
            chp_ff_connection[0],
            chp_ff_connection[1],
        )
        simenv.connect_ports(
            'p_chp{0}_rf_lt'.format(segment_name),
            'in',
            chp_ntrf_connection[0],
            chp_ntrf_connection[1],
        )
        simenv.connect_ports(  # HT RF A (lower ht rf)
            'p3v_chp{0}_rf_htA'.format(segment_name),
            'A',
            chp_htrfA_connection[0],
            chp_htrfA_connection[1],
        )
        simenv.connect_ports(  # HT RF A (upper ht rf)
            'p3v_chp{0}_rf_htB'.format(segment_name),
            'in',
            chp_htrfB_connection[0],
            chp_htrfB_connection[1],
        )
    elif number_rf_ports == 2:
        # RF from HEX to CHP with branch to incorporate HT return flow.
        # Branch is controlled 3W valve to emulate flow control valve over HEX.
        simenv.add_part(
            _ap.PipeWith3wValve,
            name='p3v_chp{0}_rf'.format(segment_name),
            length=2,
            grid_points=10,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p3v_rf'],
            T_init=np.hstack((np.full(3, 38.0), np.full(7, 52.5))),
            T_amb=Tamb,
            valve_location=3,
            ctrl_required=True,
            start_portA_opening=0.15,
            lower_limit=0.01,
            upper_limit=0.9,
        )
        simenv.connect_ports(
            'p3v_chp{0}_rf'.format(segment_name),
            'AB',
            'CHP{0}'.format(segment_name),
            'in',
        )
        # RF from outer system to HEX:
        simenv.add_part(
            _ap.Pipe,
            name='p_chp{0}_rf_lt'.format(segment_name),
            length=7,
            grid_points=30,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_rf_lt'],
            T_init=26.0,
            T_amb=Tamb,
        )
        # RF from TES HT port (ca. where lower/cold ht rf A is in 3-port
        # config.)to mix with the RF from the HEX.
        simenv.add_part(
            _ap.Pipe,
            name='p_chp{0}_rf_ht'.format(segment_name),
            length=7,
            grid_points=30,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_htrf'],
            T_init=np.full(30, 56.0),
            T_amb=Tamb,
        )
        # connect to flue gas hex return flow
        simenv.connect_ports(
            'p_chp{0}_rf_ht'.format(segment_name),
            'out',
            'p3v_chp{0}_rf'.format(segment_name),
            'B',
        )
        # add flue gas hex:
        simenv.add_part(
            _ap.HEXCondPoly,
            name='hex_chp{0}_fg'.format(segment_name),
            material='carbon_steel',
            pipe_specs=pspecs['hex_fg'],
            T_init=50,
            T_amb=Tamb,
            **fluegas_hex_kwds,
            fluegas_flow_range=(0.5, 1.05),
            water_flow_range=(0.0, 1.05),
        )
        simenv.connect_ports(
            'p_chp{0}_rf_lt'.format(segment_name),
            'out',
            'hex_chp{0}_fg'.format(segment_name),
            'water_in',
        )
        simenv.connect_ports(
            'hex_chp{0}_fg'.format(segment_name),
            'water_out',
            'p3v_chp{0}_rf'.format(segment_name),
            'A',
        )
        # boundary conditions for flue gas hex:
        simenv.add_open_port(
            name='BC_fg{0}_in'.format(segment_name),
            constant=True,
            temperature=110,
        )
        simenv.add_open_port(
            name='BC_fg{0}_out'.format(segment_name),
            constant=True,
            temperature=50,
        )
        simenv.connect_ports(
            'BoundaryCondition',
            'BC_fg{0}_in'.format(segment_name),
            'hex_chp{0}_fg'.format(segment_name),
            'fluegas_in',
        )
        simenv.connect_ports(
            'BoundaryCondition',
            'BC_fg{0}_out'.format(segment_name),
            'hex_chp{0}_fg'.format(segment_name),
            'fluegas_out',
        )

        # connect sub-system to outer system:
        simenv.connect_ports(
            'pwp_chp{0}_ff'.format(segment_name),
            'out',
            chp_ff_connection[0],
            chp_ff_connection[1],
        )
        simenv.connect_ports(
            'p_chp{0}_rf_lt'.format(segment_name),
            'in',
            chp_ntrf_connection[0],
            chp_ntrf_connection[1],
        )
        simenv.connect_ports(  # HT RF A (lower ht rf)
            'p_chp{0}_rf_ht'.format(segment_name),
            'in',
            chp_htrfA_connection[0],
            chp_htrfA_connection[1],
        )
    elif number_rf_ports == 1:
        # RF from HEX to CHP. NO branch, since only one rf port chosen!
        simenv.add_part(
            _ap.Pipe,
            name='p_chp{0}_rf'.format(segment_name),
            length=2,
            grid_points=10,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_rf'],
            T_init=np.full(10, 45.0),
            T_amb=Tamb,
        )
        simenv.connect_ports(
            'p_chp{0}_rf'.format(segment_name),
            'out',
            'CHP{0}'.format(segment_name),
            'in',
        )
        # RF from outer system to HEX:
        simenv.add_part(
            _ap.Pipe,
            name='p_chp{0}_rf_lt'.format(segment_name),
            length=7,
            grid_points=30,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_rf'],
            T_init=26.0,
            T_amb=Tamb,
        )
        # add flue gas hex:
        simenv.add_part(
            _ap.HEXCondPoly,
            name='hex_chp{0}_fg'.format(segment_name),
            material='carbon_steel',
            pipe_specs=pspecs['hex_fg'],
            T_init=50,
            T_amb=Tamb,
            **fluegas_hex_kwds,
            fluegas_flow_range=(0.5, 1.05),
            water_flow_range=(0.0, 1.1),
        )
        simenv.connect_ports(
            'p_chp{0}_rf_lt'.format(segment_name),
            'out',
            'hex_chp{0}_fg'.format(segment_name),
            'water_in',
        )
        simenv.connect_ports(
            'hex_chp{0}_fg'.format(segment_name),
            'water_out',
            'p_chp{0}_rf'.format(segment_name),
            'in',
        )
        # boundary conditions for flue gas hex:
        simenv.add_open_port(
            name='BC_fg{0}_in'.format(segment_name),
            constant=True,
            temperature=110,
        )
        simenv.add_open_port(
            name='BC_fg{0}_out'.format(segment_name),
            constant=True,
            temperature=50,
        )
        simenv.connect_ports(
            'BoundaryCondition',
            'BC_fg{0}_in'.format(segment_name),
            'hex_chp{0}_fg'.format(segment_name),
            'fluegas_in',
        )
        simenv.connect_ports(
            'BoundaryCondition',
            'BC_fg{0}_out'.format(segment_name),
            'hex_chp{0}_fg'.format(segment_name),
            'fluegas_out',
        )

        # connect sub-system to outer system:
        simenv.connect_ports(
            'pwp_chp{0}_ff'.format(segment_name),
            'out',
            chp_ff_connection[0],
            chp_ff_connection[1],
        )
        simenv.connect_ports(
            'p_chp{0}_rf_lt'.format(segment_name),
            'in',
            chp_ntrf_connection[0],
            chp_ntrf_connection[1],
        )
    else:
        raise ValueError

    # add control:
    c_pel_chp_ctrld_part = (
        'p3v_chp{0}_rf' if number_rf_ports in (2, 3) else 'p_chp{0}_rf'
    )
    # switch on CHP if on_sensor part/port is below setpoint and switch off
    # when controlled part/port is above off_val --> adjust both to match
    # theta_ff
    simenv.add_control(
        _ap.TwoSensors,
        name='c_pel_chp{0}'.format(segment_name),
        actuator='CHP{0}'.format(segment_name),
        process_CV_mode='direct',
        CV_saturation=(0.0, 1.0),
        controlled_part=c_pel_chp_ctrld_part.format(segment_name),
        controlled_port=-1,
        reference_part='none',
        setpoint=chp_ctrl['off_val'],
        sub_controller=False,
        off_state=0.0,
        time_domain='continuous',
        deadtime=0.0,
        slope=(0, 0),  # (5%/s modulation) deact., checked by chp plant
        on_sensor_part=chp_ctrl['ctrld_part'],
        on_sensor_port=chp_ctrl['on_sens_port'],
        activation_value=chp_ctrl['setpoint'],
        activation_sign='lower',
        deactivation_sign='greater',
        invert=False,
    )

    # upper pump massflow in kg/s depending on the Pel (==Pth/2) of the CHP
    # for Pel=20kW -> Pth = 40kW dm_max=4m3/h --> *.2e-3 -> in kg/s =/3.6
    # pump_max_val = chp_kwds['power_electrical'] * 0.2e-3 / 3.6
    # # calculate diff in massflows for a 5°C increase in ff temp. from 25K
    # # starting spread
    # dm_ff_diff = chp_kwds['power_electrical'] * 2 / 4180 / (
    #     85 - 60
    # ) - chp_kwds['power_electrical'] * 2 / 4180 / (90 - 60)

    if not ctrl_chp_pump_by_ts:
        simenv.add_control(
            _ap.PID,
            name='c_p_chp{0}'.format(segment_name),
            actuator='pwp_chp{0}_ff'.format(segment_name),
            process_CV_mode='part_specific',
            CV_saturation=(0.0, 1.0),
            controlled_part='CHP{0}'.format(segment_name),
            controlled_port=4,
            reference_part='none',
            setpoint=chp_kwds['theta_ff'],
            # sub_controller=True, master_type='controller',
            # master_controller='c_pel_chp{0}'.format(segment_name), dependency_kind='concurrent',
            sub_controller=True,
            master_type='part',
            master_part='CHP{0}'.format(segment_name),
            master_variable='_dQ_heating',
            master_variable_index=0,
            off_state=0.0,
            time_domain='continuous',
            deadtime=0,
            dependency_kind='concurrent',
            # slope=(-pump_max_val / 10, pump_max_val / 10),  # 10s full on-to-off
            slope=(-0.1, 0.1),  # 10s full on-to-off
            terms='PI',
            anti_windup=1.0,
            # loop_tuning='manual', Kp=dm_ff_diff, Ki=dm_ff_diff / 7,
            loop_tuning='ziegler-nichols',
            Kp_crit=0.5,
            T_crit=1.43,  # Kp=.65, Ki=0,
            adapt_coefficients=True,
            norm_timestep=1.0,
            invert=True,
        )
    else:
        simenv.add_control(
            _ap.PID,
            name='c_p_chp{0}'.format(segment_name),
            actuator='pwp_chp{0}_ff'.format(segment_name),
            process_CV_mode='part_specific',
            CV_saturation=(0.0, 1.0),
            controlled_part='CHP{0}'.format(segment_name),
            controlled_port=4,
            reference_part='none',
            setpoint=chp_kwds['theta_ff'],
            # sub_controller=True, master_type='controller',
            # master_controller='c_pel_chp{0}'.format(segment_name), dependency_kind='concurrent',
            sub_controller=True,
            master_type='part',
            master_part='CHP{0}'.format(segment_name),
            master_variable='_dQ_heating',
            master_variable_index=0,
            off_state=0.0,
            time_domain='continuous',
            deadtime=0,
            dependency_kind='concurrent',
            # slope=(-pump_max_val / 10, pump_max_val / 10),  # 10s full on-to-off
            slope=(-0.1, 0.1),  # 10s full on-to-off
            terms='PI',
            anti_windup=1.0,
            # loop_tuning='manual', Kp=dm_ff_diff, Ki=dm_ff_diff / 7,
            loop_tuning='ziegler-nichols',
            Kp_crit=0.5,
            T_crit=1.43,  # Kp=.65, Ki=0,
            adapt_coefficients=True,
            norm_timestep=1.0,
            invert=True,
        )
    # control massflow over fluegas heat exchanger
    # if controlled by timeseries, the hex will have a faster Kp/Ki value
    if not ctrl_hex_by_ts:
        simenv.add_control(
            _ap.PID,
            name='c_3v_chp_hex_fg{0}'.format(segment_name),
            actuator='p3v_chp{0}_rf'.format(segment_name),
            process_CV_mode='part_specific',
            actuator_port='A',
            CV_saturation=(0.05, 0.95),
            controlled_part='p3v_chp{0}_rf'.format(segment_name),
            controlled_port=2,
            reference_part='none',
            setpoint=fluegashex_theta_out_water,
            sub_controller=False,
            off_state=1.0,
            time_domain='continuous',
            deadtime=0,
            slope=(-0.1, 0.1),  # 10s full open-to-closed
            terms='P',
            loop_tuning='manual',
            Kp=0.025,  # Kp=.03,  Ki=.01,
            # anti_windup=10,
            adapt_coefficients=True,
            norm_timestep=1.0,
            invert=True,
        )
    else:
        # Kp = 0.035
        # Ki = 0.0025
        simenv.add_control(
            _ap.PID,
            name='c_3v_chp_hex_fg{0}'.format(segment_name),
            actuator='p3v_chp{0}_rf'.format(segment_name),
            process_CV_mode='part_specific',
            actuator_port='A',
            CV_saturation=(0.05, 0.95),
            controlled_part='p3v_chp{0}_rf'.format(segment_name),
            controlled_port=2,
            reference_part='none',
            setpoint=fluegashex_theta_out_water,
            sub_controller=False,
            off_state=1.0,
            time_domain='continuous',
            deadtime=0,
            slope=(-0.1, 0.1),  # 10s full open-to-closed
            terms='PID',
            loop_tuning='ziegler-nichols',
            rule='classic',
            Kp_crit=0.035 * 1.5,
            T_crit=37.79,
            anti_windup=150.0,
            filter_derivative=False,
            adapt_coefficients=True,
            norm_timestep=1.0,
            invert=True,
        )
    # control mix of massflows through HT rf ports
    # Ermittelung von n min für port A (d.h. minimale Öffnung von HT RF A):
    # Messdaten Mitte CHP Lastgang: T_B=56, T_A=21 T_mix=45.6
    # Messdaten Anfang CHP Lastgang: T_B=52.3, T_A=20.4, T_mix=43
    # T_A überschreitet T_mix bei 65°C, danach T_A = T_mix. D.h. Mindestöffnung
    # für HT RF A immer vorhanden. aus Messerwerten mit
    # T_B * (1-n) + T_A * n = T_mix
    # ergibt sich: n_min = .3 (Mindestöffnung HT RF A), d.h. mit act port =B
    # dann CV sat bei (0, 1-n) = (0, .7)
    print('\n\n\n ggf hier Temperatur als timeseries vorgeben')
    if not ctrl_htrf_by_ts:
        simenv.add_control(
            _ap.PID,
            name='c_3v_chp_htrf{0}'.format(segment_name),
            actuator='p3v_chp{0}_rf_htA'.format(segment_name),
            process_CV_mode='part_specific',
            actuator_port='A',
            CV_saturation=(0.3, 1.0),
            controlled_part='p3v_chp{0}_rf_htA'.format(segment_name),
            controlled_port=8,
            reference_part='none',
            setpoint=ht_rf_theta_mix,
            sub_controller=False,
            off_state=0.0,
            time_domain='continuous',
            deadtime=0,
            slope=(-0.1, 0.1),  # 10s full open-to-closed
            terms='P',
            loop_tuning='manual',
            Kp=0.15,  # Kp=.15, #Ki=.1,
            # anti_windup=25,
            adapt_coefficients=True,
            norm_timestep=1.0,
            invert=True,
        )
    else:
        simenv.add_control(
            _ap.PID,
            name='c_3v_chp_htrf{0}'.format(segment_name),
            actuator='p3v_chp{0}_rf_htA'.format(segment_name),
            process_CV_mode='part_specific',
            actuator_port='A',
            CV_saturation=(0.03, 1.0),
            controlled_part='p3v_chp{0}_rf_htA'.format(segment_name),
            controlled_port=8,
            reference_part='none',
            setpoint=ht_rf_theta_mix,
            sub_controller=False,
            off_state=0.0,
            time_domain='continuous',
            deadtime=0,
            slope=(-0.1, 0.1),  # 10s full open-to-closed
            terms='PID',
            loop_tuning='ziegler-nichols',
            rule='classic',
            Kp_crit=0.035 * 2,
            T_crit=15.0,
            anti_windup=150.0,
            filter_derivative=False,
            adapt_coefficients=True,
            norm_timestep=1.0,
            invert=True,
        )
