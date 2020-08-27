# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Jan 2020
"""

import numpy as np

from ... import all_parts as msp
from ... import utility_functions as ut


def chp_with_fghex(
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
    chp_ctrl=dict(
        ctrld_part='tes',
        on_sensor_port=15,
        setpoint=70.0,
        off_val=75.0,
        chp_mode='heat',
    ),
    fluegashex_theta_out_water=55.0,
    ht_rf_theta_mix=65.0,
    adjust_pipes=True,
    Tamb=25,
    ctrl_chp_pump_by_ts=False,
    ctrl_hex_by_ts=False,
    ctrl_htrf_by_ts=False,
    fluegas_hex_kwds=dict(hex_regression_pipeline='pass sklearn pipe'),
    calculations_df=None,
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
    assert isinstance(number_rf_ports, (int, float))
    assert chp_ctrl['chp_mode'] in ('heat', 'el', 'el_mod')
    if chp_ctrl['chp_mode'] == 'el_mod':
        assert (
            'tes_cap_mpred' in chp_ctrl
            and 'opt_profiles' in chp_ctrl
            and 'ctrl_profiles' in chp_ctrl
        )
        # extract TES caps for modelpredictive control optimization
        tes_cap_min, tes_cap_max = chp_ctrl['tes_cap_mpred']

    # calculate pump flow limits with a maximum allowed temp exceedence of 3K:
    chp_ff_pump_ulim = (
        chp_kwds['power_electrical']
        / chp_kwds['p2h_ratio']
        / ((chp_kwds['theta_ff'] - chp_ctrl['off_val']) * 4180)
    )

    ps_dn25 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN25'}}
    ps_dn32 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN32'}}
    ps_dn40 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN40'}}
    # ps_dn50 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN50'}}
    # ps_dn65 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN65'}}
    ps_dn80 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN80'}}

    # scale all values by the CHP power ratio (to the validation chp th power
    # of 38.7kW) and by the temperature spread compared to vld ref temp spread
    chp_power_factor = (
        chp_kwds['power_electrical']
        / chp_kwds['p2h_ratio']
        / 38.7e3
        * (82.5 - 73.0)
        / (chp_kwds['theta_ff'] - chp_ctrl['off_val'])
    )
    pspecs = {  # get ref pipe specs and mult factor for each part
        'pwp_ff': (ps_dn40, chp_power_factor),
        'p3v_rf': (ps_dn40, chp_power_factor),
        'p_rf_lt': (ps_dn25, chp_power_factor),
        'p3v_htrfA': (ps_dn32, chp_power_factor),
        'p_htrfB': (ps_dn32, chp_power_factor),
        'p_htrf': (ps_dn40, chp_power_factor),
        'p_rf': (ps_dn40, chp_power_factor),
        'hex_fg': (ps_dn40, chp_power_factor),
    }

    if 'pipe_specs' in chp_kwds:
        pspecs['chp'] = (chp_kwds['pipe_specs'], chp_power_factor)
        del chp_kwds['pipe_specs']
    else:
        pspecs['chp'] = (ps_dn80, chp_power_factor)

    if adjust_pipes:  # adjust pipes by A_i multiplier
        pspecs = ut.adjust_pipes(pspecs)
    else:  # take pspecs without multiplier
        pspecs = {k: v[0] for k, v in pspecs.items()}

    # make some basic chp power calculations:
    # core chp pth und eta
    chp_pth_core = chp_kwds['power_electrical'] / chp_kwds['p2h_ratio']
    chp_eta_pth = chp_kwds['eta_el'] / chp_kwds['p2h_ratio']
    # gas power lower and higher heating value
    chp_pgas_lhv = chp_kwds['power_electrical'] / chp_kwds['eta_el']
    chp_pgas_hhv = chp_pgas_lhv * 1.108

    # chp_pps = ps_dn80
    if number_rf_ports != 0.5:  # with fg hex
        simenv.add_part(
            msp.CHPPlant,
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
            fg_hex_name='hex_chp_fg_{0}'.format(segment_name),
            store_results=True,
        )
        # also calculate an estimated total power assuming that the FG HEX can
        # extract 50% of the condensation enthalpy.
        chp_pth_est = (
            chp_pth_core
            # get cond. enth. as a fraction relative to the thermal chp power
            # fraction
            + ((chp_pgas_hhv * chp_eta_pth) - chp_pth_core) * 0.5
        )
    else:  # without fg hex
        simenv.add_part(
            msp.CHPPlant,
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
            connect_fluegas_hex=False,
            store_results=True,
        )
        # here the estimated thermal power is simply the thermal power...
        chp_pth_est = chp_pth_core

    # ff pipe with pump
    simenv.add_part(
        msp.PipeWithPump,
        name='pwp_chp_ff_{0}'.format(segment_name),
        length=4,
        grid_points=15,
        s_ins=0.05,
        lambda_ins=0.03,
        material='carbon_steel',
        pipe_specs=pspecs['pwp_ff'],
        T_init=np.full(15, chp_kwds['theta_ff']),
        T_amb=Tamb,
        start_massflow=0.0,
        lower_limit=0.0,
        upper_limit=chp_ff_pump_ulim,
        # max flow to cool 40kW at 7K spread
        maximum_flow=chp_ff_pump_ulim,
        store_results=(0, 1, 7, 14),
    )
    simenv.connect_ports(
        'CHP{0}'.format(segment_name),
        'out',
        'pwp_chp_ff_{0}'.format(segment_name),
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
            msp.PipeWith3wValve,
            name='p3v_chp_rf_{0}'.format(segment_name),
            length=2,
            grid_points=7,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p3v_rf'],
            T_init=np.hstack((np.full(2, 38.0), np.full(5, 52.5))),
            T_amb=Tamb,
            valve_location=2,
            ctrl_required=True,
            start_portA_opening=0.15,
            lower_limit=0.01,
            upper_limit=0.9,
            store_results=(0, 1, 2, 3, 6),
        )
        simenv.connect_ports(
            'p3v_chp_rf_{0}'.format(segment_name),
            'AB',
            'CHP{0}'.format(segment_name),
            'in',
        )
        # RF from outer system to HEX:
        simenv.add_part(
            msp.Pipe,
            name='p_chp_rf_lt_{0}'.format(segment_name),
            length=7,
            grid_points=30,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_rf_lt'],
            T_init=26.0,
            T_amb=Tamb,
            store_results=(0, 1, 15, 28, 29),
        )
        # RF from TES HT port A (lower HT port) to mix with the RF from the HEX.
        # Valve mixes flow with HT RF port B. P-Controller with low Kp suits the
        # thermostatic valve best. Check for best mixing temperature
        simenv.add_part(
            msp.PipeWith3wValve,
            name='p3v_chp_rf_htA_{0}'.format(segment_name),
            length=7,
            grid_points=20,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p3v_htrfA'],
            T_init=np.hstack(
                # siehe Messdaten für Startwerte
                (
                    np.linspace(49.7, 51.7, 6),  # ht rf a von 49.7 auf 51.7
                    np.linspace(51.7, 54.2, 14),
                )
            ),  # hinter mix, dann auf 54.2
            T_amb=Tamb,
            valve_location=6,
            ctrl_required=True,
            start_portA_opening=0.3,
            lower_limit=0.0,
            upper_limit=1.0,
            store_results=(0, 1, 5, 6, 7, 18, 19),
        )
        simenv.connect_ports(
            'p3v_chp_rf_htA_{0}'.format(segment_name),
            'AB',
            'p3v_chp_rf_{0}'.format(segment_name),
            'B',
        )
        # add pipe from TES HT RF B (upper/hot ht rf) to mix with HT RF A
        simenv.add_part(  # Connects to TES and htrf_A
            msp.Pipe,
            name='p_chp_rf_htB_{0}'.format(segment_name),
            length=2,
            grid_points=8,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_htrfB'],
            T_init=51.5,
            T_amb=Tamb,
            store_results=(0, 1, 4, 6, 7),
        )
        simenv.connect_ports(
            'p_chp_rf_htB_{0}'.format(segment_name),
            'out',
            'p3v_chp_rf_htA_{0}'.format(segment_name),
            'B',
        )
        # add flue gas hex:
        simenv.add_part(
            msp.HEXCondPoly,
            name='hex_chp_fg_{0}'.format(segment_name),
            material='carbon_steel',
            pipe_specs=pspecs['hex_fg'],
            T_init=50,
            T_amb=Tamb,
            **fluegas_hex_kwds,
            fluegas_flow_range=(0.5, 1.05),
            water_flow_range=(0.0, 1.05),
            store_results=True,
        )
        simenv.connect_ports(
            'p_chp_rf_lt_{0}'.format(segment_name),
            'out',
            'hex_chp_fg_{0}'.format(segment_name),
            'water_in',
        )
        simenv.connect_ports(
            'hex_chp_fg_{0}'.format(segment_name),
            'water_out',
            'p3v_chp_rf_{0}'.format(segment_name),
            'A',
        )
        # boundary conditions for flue gas hex:
        simenv.add_open_port(
            name='BC_fg_in_{0}'.format(segment_name),
            constant=True,
            temperature=110,
        )
        simenv.add_open_port(
            name='BC_fg_out_{0}'.format(segment_name),
            constant=True,
            temperature=50,
        )
        simenv.connect_ports(
            'BoundaryCondition',
            'BC_fg_in_{0}'.format(segment_name),
            'hex_chp_fg_{0}'.format(segment_name),
            'fluegas_in',
        )
        simenv.connect_ports(
            'BoundaryCondition',
            'BC_fg_out_{0}'.format(segment_name),
            'hex_chp_fg_{0}'.format(segment_name),
            'fluegas_out',
        )

        # connect sub-system to outer system:
        simenv.connect_ports(
            'pwp_chp_ff_{0}'.format(segment_name),
            'out',
            chp_ff_connection[0],
            chp_ff_connection[1],
        )
        simenv.connect_ports(
            'p_chp_rf_lt_{0}'.format(segment_name),
            'in',
            chp_ntrf_connection[0],
            chp_ntrf_connection[1],
        )
        simenv.connect_ports(  # HT RF A (lower ht rf)
            'p3v_chp_rf_htA_{0}'.format(segment_name),
            'A',
            chp_htrfA_connection[0],
            chp_htrfA_connection[1],
        )
        simenv.connect_ports(  # HT RF A (upper ht rf)
            'p_chp_rf_htB_{0}'.format(segment_name),
            'in',
            chp_htrfB_connection[0],
            chp_htrfB_connection[1],
        )
    elif number_rf_ports == 2:
        # RF from HEX to CHP with branch to incorporate HT return flow.
        # Branch is controlled 3W valve to emulate flow control valve over HEX.
        simenv.add_part(
            msp.PipeWith3wValve,
            name='p3v_chp_rf_{0}'.format(segment_name),
            length=2,
            grid_points=7,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p3v_rf'],
            T_init=np.hstack((np.full(2, 38.0), np.full(5, 52.5))),
            T_amb=Tamb,
            valve_location=2,
            ctrl_required=True,
            start_portA_opening=0.15,
            lower_limit=0.01,
            upper_limit=0.9,
            store_results=(0, 1, 2, 3, 6),
        )
        simenv.connect_ports(
            'p3v_chp_rf_{0}'.format(segment_name),
            'AB',
            'CHP{0}'.format(segment_name),
            'in',
        )
        # RF from outer system to HEX:
        simenv.add_part(
            msp.Pipe,
            name='p_chp_rf_lt_{0}'.format(segment_name),
            length=7,
            grid_points=30,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_rf_lt'],
            T_init=26.0,
            T_amb=Tamb,
            store_results=(0, 1, 15, 28, 29),
        )
        # RF from TES HT port (ca. where lower/cold ht rf A is in 3-port
        # config.)to mix with the RF from the HEX.
        simenv.add_part(
            msp.Pipe,
            name='p_chp_rf_ht_{0}'.format(segment_name),
            length=7,
            grid_points=30,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_htrf'],
            T_init=np.full(30, 56.0),
            T_amb=Tamb,
            store_results=(0, 1, 15, 28, 29),
        )
        # connect to flue gas hex return flow
        simenv.connect_ports(
            'p_chp_rf_ht_{0}'.format(segment_name),
            'out',
            'p3v_chp_rf_{0}'.format(segment_name),
            'B',
        )
        # add flue gas hex:
        simenv.add_part(
            msp.HEXCondPoly,
            name='hex_chp_fg_{0}'.format(segment_name),
            material='carbon_steel',
            pipe_specs=pspecs['hex_fg'],
            T_init=50,
            T_amb=Tamb,
            **fluegas_hex_kwds,
            fluegas_flow_range=(0.5, 1.05),
            water_flow_range=(0.0, 1.05),
            store_results=True,
        )
        simenv.connect_ports(
            'p_chp_rf_lt_{0}'.format(segment_name),
            'out',
            'hex_chp_fg_{0}'.format(segment_name),
            'water_in',
        )
        simenv.connect_ports(
            'hex_chp_fg_{0}'.format(segment_name),
            'water_out',
            'p3v_chp_rf_{0}'.format(segment_name),
            'A',
        )
        # boundary conditions for flue gas hex:
        simenv.add_open_port(
            name='BC_fg_in_{0}'.format(segment_name),
            constant=True,
            temperature=110,
        )
        simenv.add_open_port(
            name='BC_fg_out_{0}'.format(segment_name),
            constant=True,
            temperature=50,
        )
        simenv.connect_ports(
            'BoundaryCondition',
            'BC_fg_in_{0}'.format(segment_name),
            'hex_chp_fg_{0}'.format(segment_name),
            'fluegas_in',
        )
        simenv.connect_ports(
            'BoundaryCondition',
            'BC_fg_out_{0}'.format(segment_name),
            'hex_chp_fg_{0}'.format(segment_name),
            'fluegas_out',
        )

        # connect sub-system to outer system:
        simenv.connect_ports(
            'pwp_chp_ff_{0}'.format(segment_name),
            'out',
            chp_ff_connection[0],
            chp_ff_connection[1],
        )
        simenv.connect_ports(
            'p_chp_rf_lt_{0}'.format(segment_name),
            'in',
            chp_ntrf_connection[0],
            chp_ntrf_connection[1],
        )
        simenv.connect_ports(  # HT RF A (lower ht rf)
            'p_chp_rf_ht_{0}'.format(segment_name),
            'in',
            chp_htrfA_connection[0],
            chp_htrfA_connection[1],
        )
    elif number_rf_ports == 1:
        # RF from HEX to CHP. NO branch, since only one rf port chosen!
        simenv.add_part(
            msp.Pipe,
            name='p_chp_rf_{0}'.format(segment_name),
            length=2,
            grid_points=6,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_rf'],
            T_init=np.full(6, 45.0),
            T_amb=Tamb,
            store_results=(0, 1, 4, 5),
        )
        simenv.connect_ports(
            'p_chp_rf_{0}'.format(segment_name),
            'out',
            'CHP{0}'.format(segment_name),
            'in',
        )
        # RF from outer system to HEX:
        simenv.add_part(
            msp.Pipe,
            name='p_chp_rf_lt_{0}'.format(segment_name),
            length=7,
            grid_points=21,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_rf'],
            T_init=26.0,
            T_amb=Tamb,
            store_results=(0, 1, 10, 19, 20),
        )
        # add flue gas hex:
        simenv.add_part(
            msp.HEXCondPoly,
            name='hex_chp_fg_{0}'.format(segment_name),
            material='carbon_steel',
            pipe_specs=pspecs['hex_fg'],
            T_init=50,
            T_amb=Tamb,
            **fluegas_hex_kwds,
            fluegas_flow_range=(0.5, 1.05),
            water_flow_range=(0.0, 1.05),
            store_results=True,
        )
        simenv.connect_ports(
            'p_chp_rf_lt_{0}'.format(segment_name),
            'out',
            'hex_chp_fg_{0}'.format(segment_name),
            'water_in',
        )
        simenv.connect_ports(
            'hex_chp_fg_{0}'.format(segment_name),
            'water_out',
            'p_chp_rf_{0}'.format(segment_name),
            'in',
        )
        # boundary conditions for flue gas hex:
        simenv.add_open_port(
            name='BC_fg_in_{0}'.format(segment_name),
            constant=True,
            temperature=110,
        )
        simenv.add_open_port(
            name='BC_fg_out_{0}'.format(segment_name),
            constant=True,
            temperature=50,
        )
        simenv.connect_ports(
            'BoundaryCondition',
            'BC_fg_in_{0}'.format(segment_name),
            'hex_chp_fg_{0}'.format(segment_name),
            'fluegas_in',
        )
        simenv.connect_ports(
            'BoundaryCondition',
            'BC_fg_out_{0}'.format(segment_name),
            'hex_chp_fg_{0}'.format(segment_name),
            'fluegas_out',
        )

        # connect sub-system to outer system:
        simenv.connect_ports(
            'pwp_chp_ff_{0}'.format(segment_name),
            'out',
            chp_ff_connection[0],
            chp_ff_connection[1],
        )
        simenv.connect_ports(
            'p_chp_rf_lt_{0}'.format(segment_name),
            'in',
            chp_ntrf_connection[0],
            chp_ntrf_connection[1],
        )
    elif number_rf_ports == 0.5:
        # ONE RF PORT AND NO FLUE GAS HEX!!
        # RF from HEX to CHP. NO branch, since only one rf port chosen!
        simenv.add_part(
            msp.Pipe,
            name='p_chp_rf_{0}'.format(segment_name),
            length=9,
            grid_points=27,
            s_ins=0.05,
            lambda_ins=0.03,
            material='carbon_steel',
            pipe_specs=pspecs['p_rf'],
            T_init=np.full(27, 45.0),
            T_amb=Tamb,
            store_results=(0, 1, 13, 25, 26),
        )
        simenv.connect_ports(
            'p_chp_rf_{0}'.format(segment_name),
            'out',
            'CHP{0}'.format(segment_name),
            'in',
        )
        # connect sub-system to outer system:
        simenv.connect_ports(
            'pwp_chp_ff_{0}'.format(segment_name),
            'out',
            chp_ff_connection[0],
            chp_ff_connection[1],
        )
        simenv.connect_ports(
            'p_chp_rf_{0}'.format(segment_name),
            'in',
            chp_ntrf_connection[0],
            chp_ntrf_connection[1],
        )
    else:
        raise ValueError

    # add control:
    if chp_ctrl['chp_mode'] == 'heat':
        c_pel_chp_ctrld_part = (
            'p3v_chp_rf_{0}' if number_rf_ports in (2, 3) else 'p_chp_rf_{0}'
        )
        # switch on CHP if on_sensor part/port is below setpoint and switch off
        # when controlled part/port is above off_val --> adjust both to match
        # theta_ff
        simenv.add_control(
            msp.TwoSensors,
            name='c_chp_pel_{0}'.format(segment_name),
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
            on_sensor_port=chp_ctrl['on_sensor_port'],
            activation_value=chp_ctrl['setpoint'],
            activation_sign='lower',
            deactivation_sign='greater',
            invert=False,
        )
    elif chp_ctrl['chp_mode'] == 'el_mod':
        c_pel_chp_ctrld_part = (
            'p3v_chp_rf_{0}' if number_rf_ports in (2, 3) else 'p_chp_rf_{0}'
        )
        # electricity led modulated CHP mode including model predictive ctrl
        simenv.add_control(
            msp.ModelPredCHP,
            name='chp_modelpred_{0}'.format(segment_name),
            actuator='CHP{0}'.format(segment_name),
            process_CV_mode='direct',
            CV_saturation=(0.0, 1.0),
            controlled_part=c_pel_chp_ctrld_part.format(segment_name),
            controlled_port=-1,
            reference_part='none',
            setpoint=chp_ctrl['off_val'],
            sub_controller=False,
            chp_params={
                'pel_chp': chp_kwds['power_electrical'],
                'pth_chp': chp_pth_est,
                'eta_el': chp_kwds['eta_el'],
                'mod_range': chp_kwds['modulation_range'],
            },
            opt_params={
                'opt_timeframe': '2d',
                'opt_every': '15min',
                'max_iter': 500,
            },
            tes_soc_minmax=(tes_cap_min, tes_cap_max),
            tes_part='tes',
            opt_profiles=chp_ctrl['opt_profiles'],
            ctrl_profiles=chp_ctrl['ctrl_profiles'],
            costs='default',
            off_state=0.0,
            time_domain='continuous',
            deadtime=0.0,
            slope=(-0.2, 0.2),  # (20%/s modulation)
            on_sensor_part=chp_ctrl['ctrld_part'],
            on_sensor_port=chp_ctrl['on_sensor_port'],
            activation_value=chp_ctrl['setpoint'],
            activation_sign='lower',
            deactivation_sign='greater',
            invert=False,
            emergency_cntrl={
                'use_hysteresis': True,
                'hysteresis': 1.0,
                'full_power_offset': 2.0,
            },  # go into full power 2°C below act val SP
            tes_cap_restore_time='default',  # time to restore cap to input val
        )

    # pump controls
    if not ctrl_chp_pump_by_ts:
        simenv.add_control(
            msp.PID,
            name='c_chp_pump_{0}'.format(segment_name),
            actuator='pwp_chp_ff_{0}'.format(segment_name),
            process_CV_mode='part_specific',
            CV_saturation=(0.0, 1.0),
            controlled_part='CHP{0}'.format(segment_name),
            controlled_port=4,
            reference_part='none',
            setpoint=chp_kwds['theta_ff'],
            # sub_controller=True, master_type='controller',
            # master_controller='c_chp_pel_{0}'.format(segment_name), dependency_kind='concurrent',
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
            anti_windup='auto_.1',
            # stabil aber ein bisschen langsam:
            # loop_tuning='ziegler-nichols', Kp_crit=.6, T_crit=2.5,
            # loop_tuning='tune', Kp=.68,
            loop_tuning='ziegler-nichols',
            Kp_crit=0.65,
            T_crit=40 / 16,
            adapt_coefficients=True,
            norm_timestep=1.0,
            invert=True,
        )
    else:
        simenv.add_control(
            msp.PID,
            name='c_chp_pump_{0}'.format(segment_name),
            actuator='pwp_chp_ff_{0}'.format(segment_name),
            process_CV_mode='part_specific',
            CV_saturation=(0.0, 1.0),
            controlled_part='CHP{0}'.format(segment_name),
            controlled_port=4,
            reference_part='none',
            setpoint=chp_kwds['theta_ff'],
            # sub_controller=True, master_type='controller',
            # master_controller='c_chp_pel_{0}'.format(segment_name), dependency_kind='concurrent',
            sub_controller=True,
            master_type='part',
            master_part='CHP{0}'.format(segment_name),
            master_variable='_dQ_heating',
            master_variable_index=0,
            dependency_kind='concurrent',
            off_state=0.0,
            time_domain='continuous',
            deadtime=0,
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
    # only if 2 or 3 rf ports are to be simulated
    if number_rf_ports in (2, 3):
        if not ctrl_hex_by_ts:
            simenv.add_control(
                msp.PID,
                name='c_chp_3v_hex_fg_{0}'.format(segment_name),
                actuator='p3v_chp_rf_{0}'.format(segment_name),
                process_CV_mode='part_specific',
                actuator_port='A',
                CV_saturation=(0.05, 0.95),
                controlled_part='p3v_chp_rf_{0}'.format(segment_name),
                controlled_port=1,
                reference_part='none',
                setpoint=fluegashex_theta_out_water,
                sub_controller=False,
                off_state=1.0,
                time_domain='continuous',
                deadtime=0,
                slope=(-0.1, 0.1),  # 10s full open-to-closed
                # terms='PID', loop_tuning='manual', Kp=0.025,  #Kp=.03,  Ki=.01,
                # anti_windup=10,
                terms='PID',
                loop_tuning='ziegler-nichols',
                rule='classic',
                Kp_crit=0.035 * 1.3,
                T_crit=37.79,
                anti_windup=150.0,
                filter_derivative=False,
                adapt_coefficients=True,
                norm_timestep=1.0,
                invert=True,
            )
        else:
            # Kp = 0.035
            # Ki = 0.0025
            simenv.add_control(
                msp.PID,
                name='c_chp_3v_hex_fg_{0}'.format(segment_name),
                actuator='p3v_chp_rf_{0}'.format(segment_name),
                process_CV_mode='part_specific',
                actuator_port='A',
                CV_saturation=(0.05, 0.95),
                controlled_part='p3v_chp_rf_{0}'.format(segment_name),
                controlled_port=1,
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
    # only if 3 rf ports are to be simulated
    if number_rf_ports == 3:
        # act_3v_htrf = 'p3v_chp_rf_htA_{0}'.format(segment_name)
        if not ctrl_htrf_by_ts:
            simenv.add_control(
                msp.PID,
                name='c_chp_3v_htrf_{0}'.format(segment_name),
                actuator='p3v_chp_rf_htA_{0}'.format(segment_name),
                process_CV_mode='part_specific',
                actuator_port='A',
                CV_saturation=(0.3, 1.0),
                controlled_part='p3v_chp_rf_htA_{0}'.format(segment_name),
                controlled_port=7,
                reference_part='none',
                setpoint=ht_rf_theta_mix,
                sub_controller=False,
                off_state=0.0,
                time_domain='continuous',
                deadtime=0,
                slope=(-0.05, 0.05),  # 10s full open-to-closed
                # terms='P', loop_tuning='manual', Kp=0.15,  # Kp=.15, #Ki=.1,
                # anti_windup=25,
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
        else:
            simenv.add_control(
                msp.PID,
                name='c_chp_3v_htrf_{0}'.format(segment_name),
                actuator='p3v_chp_rf_htA_{0}'.format(segment_name),
                process_CV_mode='part_specific',
                actuator_port='A',
                CV_saturation=(0.03, 1.0),
                controlled_part='p3v_chp_rf_htA_{0}'.format(segment_name),
                controlled_port=7,
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

    # save calculations to df and return if given:
    if calculations_df is not None:
        calculations_df.loc[
            'max_pump_flow', 'CHP{0}'.format(segment_name)
        ] = chp_ff_pump_ulim
        calculations_df.loc[
            'pipe_scaling', 'CHP{0}'.format(segment_name)
        ] = chp_power_factor
        return calculations_df


def gasboiler(
    simenv,
    gasb_pth,
    lhs,
    rc_flow,
    phw_dmd,
    space_heating=None,
    n_rf_ports=1,
    segment_name='1',
    adjust_pipes=True,
    ff_connector={'part': 'port'},
    ltrf_connector={'part': 'port'},
    htrf_connector={'part': 'port'},
    gasboiler_ctrl=dict(
        on_sens_part='tes',
        on_sens_port=5,
        setpoint=70.0,
        off_sens_part='tes',
        off_sens_port=10,
        off_val=75.0,
        theta_ff_gasb=75.0,
    ),
    Tamb=25.0,
    calculations_df=None,
):
    # ps_dn20 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN20'}}
    # ps_dn25 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN25'}}
    # ps_dn32 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN32'}}
    # ps_dn40 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN40'}}
    ps_dn50 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN50'}}
    # ps_dn65 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN65'}}
    # ps_dn80 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN80'}}
    ps_dn125 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN125'}}

    # scale all values by the gasboiler thermal power and the temperature
    # spread with reference to the ref min. temp. spread of 15K
    gasb_factor = (
        gasb_pth
        / 150e3
        / (gasboiler_ctrl['theta_ff_gasb'] - 60.0)
        * (75 - 60.0)
    )
    # get ref pipe specs and mult factor for each part
    pspecs = {
        'pwp_gasb_ff': (ps_dn50, gasb_factor),
        'p_gasb_core': (ps_dn125, gasb_factor),
        'p_gasb_rf': (ps_dn50, gasb_factor),
        'pb_gasb_rf': (ps_dn50, gasb_factor),
    }

    if adjust_pipes:  # adjust pipes by A_i multiplier
        pspecs = ut.adjust_pipes(pspecs)
    else:  # take pspecs without multiplier
        pspecs = {k: v[0] for k, v in pspecs.items()}

    # scale pump flow limit to full flow at 15K spread at full power
    pmp_flow_lim = gasb_pth / (4180 * (gasboiler_ctrl['theta_ff_gasb'] - 60.0))

    simenv.add_part(  # Rohr von Gasboiler
        msp.PipeWithPump,
        name='pwp_gasb_ff_{0}'.format(segment_name),
        length=2.5,
        grid_points=8,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        material='carbon_steel',
        pipe_specs=pspecs['pwp_gasb_ff'],
        T_amb=Tamb,
        T_init=np.full(8, gasboiler_ctrl['theta_ff_gasb']),
        start_massflow=0.0,
        lower_limit=0,
        upper_limit=pmp_flow_lim,
        maximum_flow=pmp_flow_lim,
        store_results=(0, 1, 4, 7),
    )
    simenv.connect_ports(
        list(ff_connector.keys())[0],
        list(ff_connector.values())[0],
        'pwp_gasb_ff_{0}'.format(segment_name),
        'out',
    )

    # GASBOILER CORE
    simenv.add_part(  # Rohr von Gasboiler
        msp.HeatedPipe,
        name='ph_gasb_{0}'.format(segment_name),
        length=2.5,
        grid_points=6,
        insulation_thickness=0.2,
        insulation_lambda=0.035,
        material='carbon_steel',
        pipe_specs=pspecs['p_gasb_core'],
        T_amb=Tamb,
        T_init=np.full(6, 75.0),
        heat_spread='range',
        heated_cells=[1, 5],
        no_control=False,
        lower_limit=0.0,
        upper_limit=gasb_pth,
        store_results=True,
    )
    simenv.connect_ports(
        'pwp_gasb_ff_{0}'.format(segment_name),
        'in',
        'ph_gasb_{0}'.format(segment_name),
        'out',
    )

    if n_rf_ports == 1:
        simenv.add_part(  # Rohr von Gasboiler
            msp.Pipe,
            name='p_gasb_rf_{0}'.format(segment_name),
            length=2.5,
            grid_points=8,
            insulation_thickness=0.1,
            insulation_lambda=0.035,
            material='carbon_steel',
            pipe_specs=pspecs['p_gasb_rf'],
            T_amb=Tamb,
            T_init=np.full(8, 25.0),
            store_results=(0, 1, 4, 6, 7),
        )
        simenv.connect_ports(
            'ph_gasb_{0}'.format(segment_name),
            'in',
            'p_gasb_rf_{0}'.format(segment_name),
            'out',
        )
        simenv.connect_ports(
            'p_gasb_rf_{0}'.format(segment_name),
            'in',
            list(ltrf_connector.keys())[0],
            list(ltrf_connector.values())[0],
        )
    elif n_rf_ports == 2:
        raise NotImplementedError
    else:
        raise ValueError

    # add control:
    # switch on gas boiler if on_sensor part/port is below setpoint and
    # switch off when controlled part/port is above off_val
    # --> adjust both to match theta_ff
    simenv.add_control(  # control on/off state of gas boiler
        msp.TwoSensors,
        name='c_gasb_pth_{0}'.format(segment_name),
        actuator='ph_gasb_{0}'.format(segment_name),
        process_CV_mode='direct',
        CV_saturation=(0.0, gasb_pth),
        controlled_part=gasboiler_ctrl['off_sens_part'],
        controlled_port=gasboiler_ctrl['off_sens_port'],
        reference_part='none',
        on_sensor_part=gasboiler_ctrl['on_sens_part'],
        on_sensor_port=gasboiler_ctrl['on_sens_port'],
        setpoint=gasboiler_ctrl['off_val'],
        sub_controller=False,
        off_state=0,
        time_domain='continuous',
        deadtime=0,
        slope=(-gasb_pth / 10, gasb_pth / 10),
        activation_value=gasboiler_ctrl['setpoint'],
        activation_sign='lower',
        deactivation_sign='greater',
        invert=False,
        silence_slope_warning=True,
    )

    simenv.add_control(  # pump controls temperature at gas boiler outlet
        msp.PID,
        name='c_gasb_pump_{0}'.format(segment_name),
        actuator='pwp_gasb_ff_{0}'.format(segment_name),
        process_CV_mode='part_specific',
        CV_saturation=(0.0, 1.0),
        controlled_part='ph_gasb_{0}'.format(segment_name),
        controlled_port=4,
        reference_part='none',
        setpoint=gasboiler_ctrl['theta_ff_gasb'],
        sub_controller=True,
        master_type='part',
        master_part='ph_gasb_{0}'.format(segment_name),
        master_variable='_dQ_heating',
        master_variable_index=0,
        dependency_kind='concurrent',
        off_state=0,
        time_domain='continuous',
        deadtime=0,
        slope=(-0.1, 0.1),
        terms='PI',
        anti_windup='auto_.3',
        # loop_tuning='manual', Kp=dm_ff_diff, Ki=dm_ff_diff / 7,
        # loop_tuning='ziegler-nichols', Kp_crit=.5, T_crit=1.43,  # Kp=.65, Ki=0,
        # loop_tuning='tune', Kp=.55,  further reduced kp to .5 to reduce OS
        loop_tuning='ziegler-nichols',
        Kp_crit=0.5,
        T_crit=31 / 13,
        adapt_coefficients=True,
        norm_timestep=1.0,
        invert=True,
    )

    # save calculations to df and return if given:
    if calculations_df is not None:
        calculations_df.loc[
            'max_pump_flow', 'Gasboiler_{0}'.format(segment_name)
        ] = pmp_flow_lim
        calculations_df.loc[
            'pipe_scaling', 'Gasboiler_{0}'.format(segment_name)
        ] = gasb_factor
        return calculations_df
