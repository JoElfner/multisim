# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 09:51:38 2020

@author: Johannes
"""

import numpy as np
import pandas as pd

from ... import precomp_funs as _pf
from ... import all_parts as _msp
from ... import utility_functions as _ut


def lowex_phwrc(
    simenv,
    lhs,
    phw_dmd,
    rc_flow,
    segment_name='1',
    adjust_pipes=True,
    ff_connector={'part': 'port'},
    rf_connector={'part': 'port'},
    rc_connector={'part': 'port'},
    Tamb=25.0,
    databases=None,
    calculations_df=None,
):
    assert (
        'part' not in ff_connector
        and 'part' not in rf_connector
        and 'part' not in rc_connector
    ) and (
        'port' not in ff_connector.values()
        and 'port' not in rf_connector.values()
        and 'port' not in rc_connector.values()
    ), (
        'Pass forward, return and rc flow connectors as '
        '`{part: port}` dicts,'
        ' where the key:value pairs are existing in the simenv.'
    )
    # calculate all kind of size-scaled variables:
    # no. of plates rel. to the ref. wfl from PHD Zeisberger
    hex_phw_plates = int(round(61 / 1607 * lhs.wfl))
    # no. of plates rel. to the mean of LabData and the mean of the given flow
    hex_rc_plates = int(round(8 / 0.11555245060523046 * rc_flow.mean()))
    # check for correct n plates with n passes
    passes_hex_phw = 2
    passes_hex_rc = 1
    # adjust hex phw n plates to correct values
    if passes_hex_phw > 1:
        is_even = (((hex_phw_plates - 1) / passes_hex_phw) % 2) == 0
        while not is_even:
            hex_phw_plates += 1
            even_no = (hex_phw_plates - 1) / passes_hex_phw
            is_even = (even_no % 2) == 0
    else:
        is_even = (hex_phw_plates % 2) == 0
        hex_phw_plates = hex_phw_plates + 1 if not is_even else hex_phw_plates
    # adjust hex rc n plates to correct values
    if passes_hex_rc > 1:
        is_even = (((hex_rc_plates - 1) / passes_hex_rc) % 2) == 0
        while not is_even:
            hex_rc_plates += 1
            even_no = (hex_rc_plates - 1) / passes_hex_rc
            is_even = (even_no % 2) == 0
    else:
        is_even = (hex_rc_plates % 2) == 0
        hex_rc_plates = hex_rc_plates + 1 if not is_even else hex_rc_plates
    # add consumption side:
    theta_phw = 60.0
    theta_rc = 55.0  # return flow from the building
    theta_cw = 13.0  # cold water inflow
    theta_rc_rf = 57.0  # return flow from HEX from RC heating
    # setpoints for controllers:
    sp_theta_mix = 65.0

    # ps_dn20 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN20'}}
    ps_dn25 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN25'}}
    ps_dn32 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN32'}}
    # ps_dn40 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN40'}}
    ps_dn50 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN50'}}
    # ps_dn65 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN65'}}
    # ps_dn80 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN80'}}

    # scale all values by the PHW and RC flow factors, relative to the mean
    # values of the Lab validation
    phw_flow_factor = phw_dmd.mean() / 0.020426520552047665
    rc_flow_factor = rc_flow.mean() / 0.11555245060523046
    # if a part is affected by both flows, scale relevance of phw to RC by
    # 1/3 to 2/3
    pspecs = {
        'p3v_phwrc_ff_mix': (
            ps_dn32,
            (phw_flow_factor * 1 / 3 + rc_flow_factor * 2 / 3),
        ),
        'pb_phwrc_ff': (
            ps_dn32,
            (phw_flow_factor * 1 / 3 + rc_flow_factor * 2 / 3),
        ),
        'pwp_phw_ff': (ps_dn25, phw_flow_factor),
        'pwp_rc_ff': (ps_dn25, rc_flow_factor),
        'hex_phw': (ps_dn25, phw_flow_factor),
        'hex_rc': (ps_dn25, rc_flow_factor),
        'pb_rc_rf': (ps_dn25, rc_flow_factor),
        'p_phw_rf': (ps_dn25, phw_flow_factor),
        'p_rc_to_mix': (ps_dn25, rc_flow_factor),
        'p_rc_dmd': (ps_dn25, rc_flow_factor),
        'pb_phwrc_dmd': (
            ps_dn50,
            (phw_flow_factor * 1 / 3 + rc_flow_factor * 2 / 3),
        ),
        'pwp_rc': (ps_dn25, rc_flow_factor),
        'pwp_cw': (ps_dn32, phw_flow_factor),
    }

    if adjust_pipes:  # adjust pipes by A_i multiplier
        pspecs = _ut.adjust_pipes(pspecs)
    else:  # take pspecs without multiplier
        pspecs = {k: v[0] for k, v in pspecs.items()}

    # load databases:
    databases = _ut.database_import()

    # VL ZU VERBRAUCHERN MIT MISCHVENTIL UND AUFSPLITTUNG
    simenv.add_part(  # Rohr mit Mischventil zum Mischen der VL Temp TWE und Zirk
        _msp.PipeWith3wValve,
        name='p3v_phwrc_ff_mix_{0}'.format(segment_name),
        length=2.5,
        grid_points=8,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        material='carbon_steel',
        pipe_specs=pspecs['p3v_phwrc_ff_mix'],
        T_amb=Tamb,
        T_init=np.hstack((np.full(6, 80.0), np.full(2, sp_theta_mix))),
        valve_location=6,
        lower_limit=0,
        upper_limit=1,
        start_portA_opening=0.32,
        store_results=(0, 1, 5, 6, 7),
    )
    simenv.connect_ports(
        list(ff_connector.keys())[0],
        list(ff_connector.values())[0],
        'p3v_phwrc_ff_mix_{0}'.format(segment_name),
        'A',
    )

    simenv.add_part(  # Rohr mit Branch fuer VL zu TWE und Zirk
        _msp.PipeBranched,
        name='pb_phwrc_ff_{0}'.format(segment_name),
        length=1.0,
        grid_points=3,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['pb_phwrc_ff'],
        material='carbon_steel',
        T_init=sp_theta_mix,
        T_amb=Tamb,
        new_ports={'out_rc': [2, 'index']},
        store_results=True,
    )
    simenv.connect_ports(
        'p3v_phwrc_ff_mix_{0}'.format(segment_name),
        'AB',
        'pb_phwrc_ff_{0}'.format(segment_name),
        'in',
    )

    # ROHRE MIT PUMPEN ZU DEN VERBRAUCHER-HEX (primary side)
    simenv.add_part(  # Prim.-Seite FF PHW to HEX
        _msp.PipeWithPump,
        name='pwp_phw_ff_{0}'.format(segment_name),
        length=0.5,
        grid_points=2,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['pwp_phw_ff'],
        material='carbon_steel',
        T_init=sp_theta_mix,
        T_amb=Tamb,
        start_massflow=phw_dmd[0],
        ctrl_required=True,
        lower_limit=0.0,
        upper_limit=0.5 * phw_flow_factor,
        maximum_flow=0.5 * phw_flow_factor,
        store_results=True,
    )
    simenv.connect_ports(
        'pb_phwrc_ff_{0}'.format(segment_name),
        'out',
        'pwp_phw_ff_{0}'.format(segment_name),
        'in',
    )
    simenv.add_part(  # Prim.-Seite FF RC to HEX
        _msp.PipeWithPump,
        name='pwp_rc_ff_{0}'.format(segment_name),
        length=0.5,
        grid_points=2,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['pwp_rc_ff'],
        material='carbon_steel',
        T_init=sp_theta_mix,
        T_amb=Tamb,
        ctrl_required=True,
        start_massflow=rc_flow
        / (sp_theta_mix - theta_rc_rf)
        * (theta_phw - theta_rc),
        lower_limit=0.0,
        upper_limit=0.2 * rc_flow_factor,
        maximum_flow=0.2 * rc_flow_factor,
        store_results=True,
    )
    simenv.connect_ports(
        'pb_phwrc_ff_{0}'.format(segment_name),
        'out_rc',
        'pwp_rc_ff_{0}'.format(segment_name),
        'in',
    )

    # HEAT EXCHANGERS FOR PHW AND RC CONSUMPTION:
    simenv.add_part(  # numeric (implicit) PHW heat exchanger
        _msp.HexNum,
        name='hex_phw_{0}'.format(segment_name),
        HEx_type='plate_hex',
        passes=passes_hex_phw,
        flow_type='counter',
        grid_points=20,
        plates=hex_phw_plates,
        max_channels='supply',
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['hex_phw'],
        T_init=35.0,
        T_amb=Tamb,
        **databases['heat_exchangers'].Plate.SWEP.B25T,
        Reynolds_correction=10000,
        solve_implicit=True,
        implicit_root_diff=_pf.hexnum_imp_root_diff,
        store_results=True,
    )
    simenv.connect_ports(
        'pwp_phw_ff_{0}'.format(segment_name),
        'out',
        'hex_phw_{0}'.format(segment_name),
        'sup_in',
    )
    simenv.add_part(  # non-numeric (empiric NTU) RC heat exchanger
        _msp.HeatExchanger,
        name='hex_rc_{0}'.format(segment_name),
        HEx_type='plate_hex',
        passes=passes_hex_rc,
        plates=hex_rc_plates,
        max_channels='supply',
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['hex_rc'],
        T_init=60.0,
        T_amb=Tamb,
        **databases['heat_exchangers'].Plate.SWEP.BX8T,
        Reynolds_correction=7750,
        store_results=True,
    )
    simenv.connect_ports(
        'pwp_rc_ff_{0}'.format(segment_name),
        'out',
        'hex_rc_{0}'.format(segment_name),
        'sup_in',
    )

    # RETURN FLOW FROM RC HEAT EXCHANGER:
    # Rohr mit Branch zur Einbindung des Zirk RL in Beimisch
    simenv.add_part(
        _msp.PipeBranched,
        name='pb_rc_rf_{0}'.format(segment_name),
        length=5,
        grid_points=20,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['pb_rc_rf'],
        material='carbon_steel',
        T_init=theta_rc_rf,
        T_amb=Tamb,
        new_ports={'rc_to_mix': [5, 'index']},
        store_results=(0, 1, 4, 5, 6, 18, 19),
    )
    simenv.connect_ports(
        'hex_rc_{0}'.format(segment_name),
        'sup_out',
        'pb_rc_rf_{0}'.format(segment_name),
        'in',
    )
    simenv.connect_ports(
        'pb_rc_rf_{0}'.format(segment_name),
        'out',
        list(rc_connector.keys())[0],
        list(rc_connector.values())[0],
    )
    # Zwischenstück um Laufzeit und reale Rohrlänge zu repräsentieren
    simenv.add_part(  # Rohr von rf RC split zu Beimischventil
        _msp.Pipe,
        name='p_rc_to_mix{0}'.format(segment_name),
        length=1.5,
        grid_points=6,
        pipe_specs=pspecs['p_rc_to_mix'],
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        material='carbon_steel',
        T_init=theta_rc_rf,
        T_amb=Tamb,
        store_results=(0, 1, 3, 5),
    )
    simenv.connect_ports(
        'pb_rc_rf_{0}'.format(segment_name),
        'rc_to_mix',
        'p_rc_to_mix{0}'.format(segment_name),
        'in',
    )
    simenv.connect_ports(
        'p_rc_to_mix{0}'.format(segment_name),
        'out',
        'p3v_phwrc_ff_mix_{0}'.format(segment_name),
        'B',
    )
    # RETURN FLOW FROM PHW HEAT EXCHANGER:
    simenv.add_part(
        _msp.Pipe,
        name='p_phw_rf_{0}'.format(segment_name),
        length=2.5,
        grid_points=10,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['p_phw_rf'],
        material='carbon_steel',
        T_init=theta_cw,
        T_amb=Tamb,
        store_results=(0, 1, 5, 9),
    )
    simenv.connect_ports(
        'hex_phw_{0}'.format(segment_name),
        'sup_out',
        'p_phw_rf_{0}'.format(segment_name),
        'in',
    )
    simenv.connect_ports(
        'p_phw_rf_{0}'.format(segment_name),
        'out',
        list(rf_connector.keys())[0],
        list(rf_connector.values())[0],
    )

    # CONSUMER SIDE (INTO THE HOUSE)
    # pipe from RC hex to general pipe into building
    simenv.add_part(  # only for getting a separate PV for RC controller
        _msp.Pipe,
        name='p_rc_dmd_{0}'.format(segment_name),
        length=0.8,
        grid_points=2,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['p_rc_dmd'],
        material='carbon_steel',
        T_init=theta_phw - 0.1,
        T_amb=Tamb,
        store_results=True,
    )
    simenv.connect_ports(
        'hex_rc_{0}'.format(segment_name),
        'dmd_out',
        'p_rc_dmd_{0}'.format(segment_name),
        'in',
    )
    # forward flow into building after hex with BC
    simenv.add_part(  # forward flow from hexs into building
        _msp.PipeBranched,
        name='pb_phwrc_dmd_{0}'.format(segment_name),
        length=1,
        grid_points=6,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        material='carbon_steel',
        pipe_specs=pspecs['pb_phwrc_dmd'],
        T_init=theta_phw - 0.1,
        T_amb=Tamb,
        new_ports={'in_rc_flow': [1, 'index']},
        store_results=(0, 1, 2, 3, 5),
    )
    simenv.connect_ports(
        'hex_phw_{0}'.format(segment_name),
        'dmd_out',
        'pb_phwrc_dmd_{0}'.format(segment_name),
        'in',
    )
    simenv.connect_ports(
        'p_rc_dmd_{0}'.format(segment_name),
        'out',
        'pb_phwrc_dmd_{0}'.format(segment_name),
        'in_rc_flow',
    )
    simenv.add_open_port(  # rf from RC heating
        'BC_phwrc_theta_ff_{0}'.format(segment_name),
        constant=True,
        temperature=theta_phw,
    )
    simenv.connect_ports(
        'pb_phwrc_dmd_{0}'.format(segment_name),
        'out',
        'BoundaryCondition',
        'BC_phwrc_theta_ff_{0}'.format(segment_name),
    )
    # RC from building and boundary conditions (temp + flow)
    simenv.add_part(  # RC return from building, set with BC flow
        _msp.PipeWithPump,
        name='pwp_rc_{0}'.format(segment_name),
        length=0.8,
        grid_points=2,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['pwp_rc'],
        material='carbon_steel',
        T_init=theta_rc,
        T_amb=Tamb,
        start_massflow=rc_flow,
        ctrl_required=False,
        const_val=rc_flow,
        store_results=True,
    )
    simenv.connect_ports(
        'pwp_rc_{0}'.format(segment_name),
        'out',
        'hex_rc_{0}'.format(segment_name),
        'dmd_in',
    )
    simenv.add_open_port(  # rf from RC heating
        'BC_rc_theta_rf_{0}'.format(segment_name),
        constant=True,
        temperature=theta_rc,
    )
    simenv.connect_ports(
        'BoundaryCondition',
        'BC_rc_theta_rf_{0}'.format(segment_name),
        'pwp_rc_{0}'.format(segment_name),
        'in',
    )
    # PHW cold water inflow and boundary conditions (temp + flow)
    simenv.add_part(  # PHW DMD on cold water pumpt, set with BC flow
        _msp.PipeWithPump,
        name='pwp_cw_{0}'.format(segment_name),
        length=1,
        grid_points=5,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['pwp_cw'],
        material='carbon_steel',
        T_init=theta_cw,
        T_amb=Tamb,
        start_massflow=phw_dmd[0],
        ctrl_required=False,
        time_series=phw_dmd,
        store_results=(0, 3, 4),
    )
    simenv.add_open_port(  # rf from RC heating
        'BC_phw_theta_cw_{0}'.format(segment_name),
        constant=True,
        temperature=theta_cw,
    )
    simenv.connect_ports(
        'BoundaryCondition',
        'BC_phw_theta_cw_{0}'.format(segment_name),
        'pwp_cw_{0}'.format(segment_name),
        'in',
    )
    simenv.connect_ports(
        'pwp_cw_{0}'.format(segment_name),
        'out',
        'hex_phw_{0}'.format(segment_name),
        'dmd_in',
    )

    # ----> CONTROLS <----
    # Control Mischventil FF PHW+RC:
    simenv.add_control(
        _msp.PID,
        name='c_3v_mix{0}'.format(segment_name),
        terms='PI',
        actuator='p3v_phwrc_ff_mix_{0}'.format(segment_name),
        controlled_part='p3v_phwrc_ff_mix_{0}'.format(segment_name),
        controlled_port=7,
        actuator_port='A',
        process_CV_mode='part_specific',
        reference_part='none',
        time_domain='discrete',
        sub_controller=False,
        off_state=0.0,
        setpoint=sp_theta_mix,
        deadtime=0,
        loop_tuning='manual',
        Kp=0.04,
        Ki=1 / 100,
        adapt_coefficients=True,
        norm_timestep=1.0,
        anti_windup=1.0,
        CV_saturation=(0.0, 1.0),
        slope=(0.0, 0.0),
        invert=False,
    )
    simenv.add_control(
        _msp.PID,
        name='c_pump_phw_{0}'.format(segment_name),
        terms='PI',
        actuator='pwp_phw_ff_{0}'.format(segment_name),
        controlled_part='pb_phwrc_dmd_{0}'.format(segment_name),
        controlled_port=1,
        process_CV_mode='part_specific',
        reference_part='none',
        time_domain='continuous',
        sub_controller=True,
        master_type='part',
        master_part='pwp_cw_{0}'.format(segment_name),
        master_variable='dm',
        master_variable_index=0,
        dependency_kind='concurrent',
        off_state=0,
        setpoint=theta_phw,
        deadtime=0,
        loop_tuning='ziegler-nichols',
        # loop_tuning='tune', Kp=.25,
        # T_crit=33, rule='classic',
        Kp_crit=0.23,
        T_crit=270 / 18,
        rule='pessen-int',
        filter_derivative=False,
        cutoff_freq=0.1,
        adapt_coefficients=True,
        norm_timestep=1,  # norm_timestep=0.47,
        anti_windup='auto',
        CV_saturation=(0, 1.0),
        slope=(0, 0),
        invert=False,
        #  slope=(-1, 1), invert=False,
    )
    simenv.add_control(
        _msp.PID,
        name='c_pump_rc{0}'.format(segment_name),
        terms='PI',
        actuator='pwp_rc_ff_{0}'.format(segment_name),
        controlled_part='p_rc_dmd_{0}'.format(segment_name),
        controlled_port=1,
        process_CV_mode='part_specific',
        reference_part='none',
        time_domain='discrete',
        sub_controller=False,
        off_state=0,
        setpoint=theta_phw,
        deadtime=0,
        loop_tuning='manual',
        Kp=0.005,
        Ki=1 / 600,
        adapt_coefficients=True,
        norm_timestep=1,
        anti_windup=1.0,
        CV_saturation=(0, 1.0),
        slope=(0, 0),
        invert=False,
    )

    # save calculations to df and return if given:
    if calculations_df is not None:
        calculations_df.loc[
            'hex_phw_n_plates', 'LowEx{0}'.format(segment_name)
        ] = hex_phw_plates
        calculations_df.loc[
            'hex_rc_n_plates', 'LowEx{0}'.format(segment_name)
        ] = hex_rc_plates
        calculations_df.loc[
            'pipe_scaling_phw', 'LowEx{0}'.format(segment_name)
        ] = phw_flow_factor
        calculations_df.loc[
            'pipe_scaling_rc', 'LowEx{0}'.format(segment_name)
        ] = rc_flow_factor
        calculations_df.loc[
            'pipe_scaling_sum', 'LowEx{0}'.format(segment_name)
        ] = (phw_flow_factor * 1 / 3 + rc_flow_factor * 2 / 3)
        calculations_df.loc[
            'theta_phw', 'LowEx{0}'.format(segment_name)
        ] = theta_phw
        calculations_df.loc[
            'theta_rc', 'LowEx{0}'.format(segment_name)
        ] = theta_rc
        calculations_df.loc[
            'theta_coldwater', 'LowEx{0}'.format(segment_name)
        ] = theta_cw
        calculations_df.loc[
            'theta_rc_rf', 'LowEx{0}'.format(segment_name)
        ] = theta_rc_rf
        calculations_df.loc[
            'theta_mix', 'LowEx{0}'.format(segment_name)
        ] = sp_theta_mix

        return calculations_df


def speicherladesys(
    simenv,
    lhs,
    phw_dmd,
    rc_flow,
    theta_ff_phwstore=67.5,
    store_gp=40,
    store_dia=1.0,
    store_shell=3e-3,
    store_init=60.0,
    store_ports={
        'ff_tes': [0.0, 'volume'],
        'rc': [round(0.4 * 40), 'index'],
        'cw': [-2, 'index'],
    },
    segment_name='1',
    adjust_pipes=True,
    ff_connector={'part': 'port'},
    rf_connector={'part': 'port'},
    Tamb=25.0,
    databases=None,
    calculations_df=None,
):
    # assert correct connector setup
    assert ('part' not in ff_connector and 'part' not in rf_connector) and (
        'port' not in ff_connector.values()
        and 'port' not in rf_connector.values()
    ), (
        'Pass forward, return and rc flow connectors as '
        '`{part: port}` dicts,'
        ' where the key:value pairs are existing in the simenv.'
    )

    # calculate all kind of size-scaled variables:
    # no. of plates rel. to the ref. wfl from PHD Zeisberger of 1607m2, plus
    # a reduction factor of 3, since no extreme peaks as for PHW must be
    # covered. for a peak power of 150kW
    # (97% percentile of phw_dmds for 5k sqm), this is a peak power of the
    # PHW store HEX of 50kW --> slightly oversized for a 5k sqm building, where
    # with year 1982 standards the constant flow will be around 24kW.
    hex_store_plates = int(round(61 / 1607 * lhs.wfl / 7))
    # check for correct n plates with n passes
    passes_hex_store = 1
    # adjust hex n plates to correct values
    if passes_hex_store > 1:
        is_even = (((hex_store_plates - 1) / passes_hex_store) % 2) == 0
        while not is_even:
            hex_store_plates += 1
            even_no = (hex_store_plates - 1) / passes_hex_store
            is_even = (even_no % 2) == 0
    else:
        is_even = (hex_store_plates % 2) == 0
        hex_store_plates = (
            hex_store_plates + 1 if not is_even else hex_store_plates
        )

    # add consumption side:
    theta_phw = 60.0
    theta_rc = 55.0  # return flow from the building
    theta_cw = 13.0  # cold water inflow

    # setpoints for controllers:
    sp_theta_mix_ff_store = theta_ff_phwstore
    theta_ff_diff = 0.15  # small theta diff for forward flow to smooth control
    theta_chrg_diff = -1.5  # same for storage charging SP

    # define controls for storage charging:
    store_control_idx = int((0.35 * store_gp))  # floor of 30% of volume
    sp_store_control = theta_phw + theta_chrg_diff  # slightly lower than phw

    # scale all values by the PHW and RC flow factors, relative to the mean
    # values of the Lab validation
    phw_mean, rc_mean = phw_dmd.mean(), rc_flow.mean()
    phw_flow_factor = phw_mean / 0.020426520552047665
    rc_flow_factor = rc_mean / 0.11555245060523046
    # get mean total phw+rc power:
    mean_phwrc_power = (
        phw_mean * (theta_phw - theta_cw) + rc_mean * (theta_phw - theta_rc)
    ) * 4180
    # mean total phw+rc scaled relative to 20kW as flow factor
    mean_dmd_flow_factor = mean_phwrc_power / 20e3

    # get max pump flows with a security factor of 5 to mean power. slightly
    # reduced rf temp for store pump due to HEX temp drop
    pump_ff_hex_max = (
        mean_phwrc_power * 5 / (4180 * (theta_ff_phwstore - 55.0))
    )
    pump_store_max = (
        mean_phwrc_power * 5 / (4180 * ((theta_phw + theta_ff_diff) - 53.0))
    )
    # and mean pump flow (also starting pump flow)
    pump_ff_hex_mean = mean_phwrc_power / (4180 * (theta_ff_phwstore - 45.0))
    pump_store_mean = mean_phwrc_power / (
        4180 * ((theta_phw + theta_ff_diff) - 43.0)
    )

    # ps_dn20 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN20'}}
    ps_dn25 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN25'}}
    ps_dn32 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN32'}}
    # ps_dn40 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN40'}}
    ps_dn50 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN50'}}
    # ps_dn65 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN65'}}
    # ps_dn80 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN80'}}

    # if a part is affected by both flows, scale relevance of phw to RC by
    # 1/3 to 2/3
    pspecs = {
        # supply side
        'p3v_ff_hex': (ps_dn25, mean_dmd_flow_factor * 1.1),  # from 1.2
        'p_ff_hex': (ps_dn25, mean_dmd_flow_factor * 1.03),  # from 1.
        'pb_rf_hex': (ps_dn25, mean_dmd_flow_factor * 0.95),  # from .85
        'hex_store': (ps_dn25, mean_dmd_flow_factor),
        'p_ff_store': (ps_dn32, mean_dmd_flow_factor * 0.92),  # from .95
        'pwp_rf_store': (ps_dn32, mean_dmd_flow_factor),
        # demand side
        'p_phwrc_dmd': (
            ps_dn50,
            (phw_flow_factor * 1 / 3 + rc_flow_factor * 2 / 3),
        ),
        'pwp_rc': (ps_dn25, rc_flow_factor),
        'pwp_cw': (ps_dn32, phw_flow_factor),
    }

    if adjust_pipes:  # adjust pipes by A_i multiplier
        pspecs = _ut.adjust_pipes(pspecs)
    else:  # take pspecs without multiplier
        pspecs = {k: v[0] for k, v in pspecs.items()}

    # load databases:
    databases = _ut.database_import()

    # PHW Storage:
    simenv.add_part(
        _msp.Tes,
        name='sps_store_phw_{0}'.format(segment_name),
        volume=lhs.v_stor_phw,
        grid_points=store_gp,
        outer_diameter=store_dia,
        shell_thickness=store_shell,
        new_ports=store_ports,
        insulation_thickness=0.2,
        insulation_lambda=0.035,
        T_init=store_init,
        T_amb=Tamb,
        material='carbon_steel',
        pipe_specs={'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN40'}},
        store_results=True,
    )

    # VL zu PHW Store heat exhanger
    simenv.add_part(  # Rohr mit Mischventil: VL Temp Store auf setpoint
        _msp.PipeWith3wValve,
        name='sps_p3v_ff_hex_{0}'.format(segment_name),
        length=1.5625,
        grid_points=5,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        material='carbon_steel',
        pipe_specs=pspecs['p3v_ff_hex'],
        T_amb=Tamb,
        T_init=np.hstack(
            (np.full(3, lhs.theta_ff), np.full(2, sp_theta_mix_ff_store))
        ),
        valve_location=3,
        lower_limit=0.0,
        upper_limit=1.0,
        start_portA_opening=0.7,
        store_results=(0, 2, 3, 4),
    )
    simenv.connect_ports(  # connect to CHP TES
        list(ff_connector.keys())[0],
        list(ff_connector.values())[0],
        'sps_p3v_ff_hex_{0}'.format(segment_name),
        'A',
    )
    simenv.add_part(  # Rohr mit Pumpe zu PHW store charging HEX
        _msp.PipeWithPump,
        name='sps_pwp_ff_hex_{0}'.format(segment_name),
        length=0.625,
        grid_points=2,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        material='carbon_steel',
        pipe_specs=pspecs['p_ff_hex'],
        T_amb=Tamb,
        T_init=lhs.theta_ff,
        start_massflow=pump_ff_hex_mean,
        lower_limit=0.0,
        upper_limit=pump_ff_hex_max,
        maximum_flow=pump_ff_hex_max,
        store_results=(1,),
    )
    simenv.connect_ports(  # connect to pipe with pump
        'sps_p3v_ff_hex_{0}'.format(segment_name),
        'AB',
        'sps_pwp_ff_hex_{0}'.format(segment_name),
        'in',
    )
    # RL von PHW Store loading HEX
    # Rohr mit Branch. RL von Store HEX und zu Mischeventil
    simenv.add_part(
        _msp.PipeBranched,
        name='sps_pb_rf_hex_{0}'.format(segment_name),
        length=2.5,
        grid_points=8,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['pb_rf_hex'],
        material='carbon_steel',
        T_init=40.0,
        T_amb=Tamb,
        new_ports={'out_mix_sps_ff': [2, 'index']},
        store_results=True,
    )
    simenv.connect_ports(  # connect to CHP TES
        'sps_pb_rf_hex_{0}'.format(segment_name),
        'out',
        list(rf_connector.keys())[0],
        list(rf_connector.values())[0],
    )
    # Verbindungsrohr von RL mit Branch zu 3v:
    simenv.add_part(
        _msp.Pipe,
        name='sps_p_rf_hex_to_mix_{0}'.format(segment_name),
        length=0.625,
        grid_points=2,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['pb_rf_hex'],
        material='carbon_steel',
        T_init=40.0,
        T_amb=Tamb,
        store_results=(0,),
    )
    simenv.connect_ports(  # connect to CHP TES
        'sps_pb_rf_hex_{0}'.format(segment_name),
        'out_mix_sps_ff',
        'sps_p_rf_hex_to_mix_{0}'.format(segment_name),
        'in',
    )
    simenv.connect_ports(  # connect to CHP TES
        'sps_p_rf_hex_to_mix_{0}'.format(segment_name),
        'out',
        'sps_p3v_ff_hex_{0}'.format(segment_name),
        'B',
    )

    # HEX for loading the PHW Store:
    simenv.add_part(  # numeric (implicit) PHW heat exchanger
        _msp.HexNum,
        name='sps_hex_phw_store_{0}'.format(segment_name),
        HEx_type='plate_hex',
        passes=passes_hex_store,
        flow_type='counter',
        grid_points=20,
        plates=hex_store_plates,
        max_channels='supply',
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['hex_store'],
        T_init=35.0,
        T_amb=Tamb,
        **databases['heat_exchangers'].Plate.SWEP.B10T,
        Reynolds_correction=6000,
        solve_implicit=True,
        implicit_root_diff=_pf.hexnum_imp_root_diff,
        store_results=True,
    )
    simenv.connect_ports(  # connect hex to supply side
        'sps_pwp_ff_hex_{0}'.format(segment_name),
        'out',
        'sps_hex_phw_store_{0}'.format(segment_name),
        'sup_in',
    )
    simenv.connect_ports(
        'sps_hex_phw_store_{0}'.format(segment_name),
        'sup_out',
        'sps_pb_rf_hex_{0}'.format(segment_name),
        'in',
    )

    # Pipes between HEX and PHW Store:
    simenv.add_part(  # from HEX to store
        _msp.Pipe,
        name='sps_p_ff_store_{0}'.format(segment_name),
        length=2.5,
        grid_points=8,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['p_ff_store'],
        material='carbon_steel',
        T_init=theta_phw + theta_ff_diff,
        T_amb=Tamb,
        store_results=(0, 7),
    )
    simenv.connect_ports(  # connect to HEX
        'sps_hex_phw_store_{0}'.format(segment_name),
        'dmd_out',
        'sps_p_ff_store_{0}'.format(segment_name),
        'in',
    )
    simenv.connect_ports(  # connect to PHW Store
        'sps_p_ff_store_{0}'.format(segment_name),
        'out',
        'sps_store_phw_{0}'.format(segment_name),
        'ff_tes',
    )
    simenv.add_part(  # from store to HEX
        _msp.PipeWithPump,
        name='sps_pwp_rf_store_{0}'.format(segment_name),
        length=2.5,
        grid_points=8,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['pwp_rf_store'],
        material='carbon_steel',
        T_init=40.0,
        T_amb=Tamb,
        start_massflow=pump_store_mean,
        lower_limit=0.0,
        upper_limit=pump_store_max,
        maximum_flow=pump_store_max,
        store_results=(0, 7),
    )
    simenv.connect_ports(  # connect to PHW store
        'sps_store_phw_{0}'.format(segment_name),
        'out',
        'sps_pwp_rf_store_{0}'.format(segment_name),
        'in',
    )
    simenv.connect_ports(  # connect to HEX
        'sps_pwp_rf_store_{0}'.format(segment_name),
        'out',
        'sps_hex_phw_store_{0}'.format(segment_name),
        'dmd_in',
    )

    # CONSUMER SIDE (INTO THE HOUSE)
    # forward flow into building with PHW + RC
    simenv.add_part(  # forward flow from hexs into building
        _msp.Pipe,
        name='sps_p_phwrc_dmd_{0}'.format(segment_name),
        length=1,
        grid_points=6,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        material='carbon_steel',
        pipe_specs=pspecs['p_phwrc_dmd'],
        T_init=theta_phw - 0.1,
        T_amb=Tamb,
        store_results=(0, 1, 5),
    )
    simenv.connect_ports(  # connect to store
        'sps_store_phw_{0}'.format(segment_name),
        'in',
        'sps_p_phwrc_dmd_{0}'.format(segment_name),
        'in',
    )
    simenv.add_open_port(  # create constant temp. PHW BC
        'BC_phwrc_theta_ff_{0}'.format(segment_name),
        constant=True,
        temperature=theta_phw,
    )
    simenv.connect_ports(  # connect demand ff pipe to BC
        'sps_p_phwrc_dmd_{0}'.format(segment_name),
        'out',
        'BoundaryCondition',
        'BC_phwrc_theta_ff_{0}'.format(segment_name),
    )

    # RC from building and boundary conditions (temp + flow)
    simenv.add_part(  # RC return from building, set with BC flow
        _msp.PipeWithPump,
        name='sps_pwp_rc_{0}'.format(segment_name),
        length=0.8,
        grid_points=2,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['pwp_rc'],
        material='carbon_steel',
        T_init=theta_rc,
        T_amb=Tamb,
        start_massflow=rc_flow,
        ctrl_required=False,
        const_val=rc_flow,
        store_results=True,
    )
    # connect to PHW store
    simenv.connect_ports(
        'sps_pwp_rc_{0}'.format(segment_name),
        'out',
        'sps_store_phw_{0}'.format(segment_name),
        'rc',
    )
    simenv.add_open_port(  # create RC temp. boundary condition (constant)
        'BC_rc_theta_rf_{0}'.format(segment_name),
        constant=True,
        temperature=theta_rc,
    )
    simenv.connect_ports(  # connect to RC boundary condition
        'BoundaryCondition',
        'BC_rc_theta_rf_{0}'.format(segment_name),
        'sps_pwp_rc_{0}'.format(segment_name),
        'in',
    )

    # PHW cold water inflow and boundary conditions (temp + flow)
    simenv.add_part(  # PHW DMD on cold water pumpt, set with BC flow
        _msp.PipeWithPump,
        name='sps_pwp_cw_{0}'.format(segment_name),
        length=1,
        grid_points=5,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        pipe_specs=pspecs['pwp_cw'],
        material='carbon_steel',
        T_init=theta_cw,
        T_amb=Tamb,
        start_massflow=phw_dmd[0],
        ctrl_required=False,
        time_series=phw_dmd,
        store_results=(0, 3, 4),
    )
    simenv.add_open_port(  # create CW temp. boundary condition (constant)
        'BC_phw_theta_cw_{0}'.format(segment_name),
        constant=True,
        temperature=theta_cw,
    )
    simenv.connect_ports(  # connect BC to CW pipe
        'BoundaryCondition',
        'BC_phw_theta_cw_{0}'.format(segment_name),
        'sps_pwp_cw_{0}'.format(segment_name),
        'in',
    )
    simenv.connect_ports(  # connect CW pipe to PHW store
        'sps_pwp_cw_{0}'.format(segment_name),
        'out',
        'sps_store_phw_{0}'.format(segment_name),
        'cw',
    )

    # ----> CONTROLS <----
    # Control Mischventil FF HEX fuer Speicherladung:
    simenv.add_control(
        _msp.PID,
        name='sps_c_3v_hex_ff_{0}'.format(segment_name),
        terms='PI',
        actuator='sps_p3v_ff_hex_{0}'.format(segment_name),
        controlled_part='sps_p3v_ff_hex_{0}'.format(segment_name),
        controlled_port=4,
        actuator_port='A',
        process_CV_mode='part_specific',
        reference_part='none',
        time_domain='discrete',
        sub_controller=False,
        off_state=0.0,
        setpoint=sp_theta_mix_ff_store,
        deadtime=0,
        loop_tuning='manual',
        Kp=0.04,
        Ki=1 / 100,
        adapt_coefficients=True,
        norm_timestep=1.0,
        anti_windup=1.0,
        CV_saturation=(0.0, 1.0),
        slope=(0.0, 0.0),
        invert=False,
    )
    simenv.add_control(  # control pump on prim. side of charging HEX
        _msp.PID,
        name='sps_c_pump_hex_{0}'.format(segment_name),
        terms='PI',
        actuator='sps_pwp_ff_hex_{0}'.format(segment_name),
        controlled_part='sps_p_ff_store_{0}'.format(segment_name),
        controlled_port=1,
        process_CV_mode='part_specific',
        reference_part='none',
        time_domain='continuous',
        sub_controller=True,
        master_type='part',
        # only run pump to hex if there is a storage charging massflow
        master_part='sps_pwp_rf_store_{0}'.format(segment_name),
        master_variable='dm',
        master_variable_index=0,
        dependency_kind='concurrent',
        off_state=0,
        setpoint=theta_phw + theta_ff_diff,
        deadtime=0,
        loop_tuning='ziegler-nichols',
        # loop_tuning='tune', Kp=.25,
        # T_crit=33, rule='classic',
        # Kp_crit=0.23, T_crit=270 / 18, rule='some-os',
        Kp_crit=0.25,
        T_crit=270 / 18,
        rule='classic',  # from.kp=.25
        filter_derivative=False,
        cutoff_freq=0.1,
        adapt_coefficients=True,
        norm_timestep=1,  # norm_timestep=0.47,
        anti_windup='auto_.6',
        CV_saturation=(0, 1.0),
        slope=(0, 0),
        invert=False,
    )
    simenv.add_control(  # control pump on prim. side of charging HEX
        _msp.PID,
        name='sps_c_pump_store_{0}'.format(segment_name),
        terms='PI',
        actuator='sps_pwp_rf_store_{0}'.format(segment_name),
        controlled_part='sps_store_phw_{0}'.format(segment_name),
        controlled_port=store_control_idx,
        process_CV_mode='part_specific',
        reference_part='none',
        time_domain='continuous',
        sub_controller=False,
        setpoint=sp_store_control,
        deadtime=0,
        off_state=0.0,
        loop_tuning='ziegler-nichols',
        # loop_tuning='tune', Kp=100.,
        # T_crit=33, rule='classic',
        # Kp_crit=0.23, T_crit=270 / 18, rule='some-os',
        # Kp_crit=10., T_crit=500., rule='some-os',
        Kp_crit=9.0,
        T_crit=600.0,
        rule='some-os',
        filter_derivative=False,
        cutoff_freq=0.1,
        adapt_coefficients=True,
        norm_timestep=1,  # norm_timestep=0.47,
        anti_windup='auto_.25',  # alt: .7, 500, .4, auch gut: .7, 600, .4
        CV_saturation=(0, 1.0),
        slope=(0, 0),
        invert=False,
        #  slope=(-1, 1), invert=False,
    )
    # simenv.add_control(
    #     _msp.PID, name='c_pump_rc{0}'.format(segment_name), terms='PI',
    #     actuator='pwp_rc_ff_{0}'.format(segment_name),
    #     controlled_part='p_rc_dmd_{0}'.format(segment_name), controlled_port=1,
    #     process_CV_mode='part_specific', reference_part='none',
    #     time_domain='discrete',
    #     sub_controller=False, off_state=0,
    #     setpoint=theta_phw, deadtime=0, loop_tuning='manual', Kp=0.005,
    #     Ki=1/600, adapt_coefficients=True, norm_timestep=1, anti_windup=1.,
    #     CV_saturation=(0, 1.), slope=(0, 0), invert=False,
    # )

    # save calculations to df and return if given:
    if calculations_df is not None:
        calculations_df.loc[
            'hex_store_n_plates', 'SPS{0}'.format(segment_name)
        ] = hex_store_plates
        calculations_df.loc[
            'pipe_scaling_phw', 'SPS{0}'.format(segment_name)
        ] = phw_flow_factor
        calculations_df.loc[
            'pipe_scaling_rc', 'SPS{0}'.format(segment_name)
        ] = rc_flow_factor
        calculations_df.loc[
            'pipe_scaling_sum', 'SPS{0}'.format(segment_name)
        ] = (phw_flow_factor * 1 / 3 + rc_flow_factor * 2 / 3)
        calculations_df.loc[
            'pipe_scaling_store_charging', 'SPS{0}'.format(segment_name)
        ] = mean_dmd_flow_factor
        calculations_df.loc[
            'theta_phw', 'SPS{0}'.format(segment_name)
        ] = theta_phw
        calculations_df.loc[
            'theta_rc', 'SPS{0}'.format(segment_name)
        ] = theta_rc
        calculations_df.loc[
            'theta_coldwater', 'SPS{0}'.format(segment_name)
        ] = theta_cw
        calculations_df.loc[
            'sp_theta_mix_ff_HEX', 'SPS{0}'.format(segment_name)
        ] = sp_theta_mix_ff_store
        calculations_df.loc[
            'pump_ff_HEX_max_massflow', 'SPS{0}'.format(segment_name)
        ] = pump_ff_hex_max
        calculations_df.loc[
            'sp_theta_ff_storage', 'SPS{0}'.format(segment_name)
        ] = (theta_phw + theta_ff_diff)
        calculations_df.loc[
            'pump_storage_max_massflow', 'SPS{0}'.format(segment_name)
        ] = pump_store_max
        calculations_df.loc[
            'store_theta_ctrl_idx', 'SPS{0}'.format(segment_name)
        ] = store_control_idx
        calculations_df.loc[
            'sp_theta_storage_ctrl', 'SPS{0}'.format(segment_name)
        ] = sp_store_control
        calculations_df.loc[
            'store_rc_in_idx', 'SPS{0}'.format(segment_name)
        ] = round(store_ports['rc'][0] / lhs.v_stor_phw * store_gp)

        return calculations_df


def space_heating(
    simenv,
    lhs,
    space_heating,
    aux_gasboiler_pth,
    aux_gasboiler_theta_ff,
    configuration='4w_valve',
    segment_name='1',
    adjust_pipes=True,
    tes_ff_connector={'part': 'port'},
    tes_rf_connector={'part': 'port'},
    Tamb=25.0,
    databases=None,
    calculations_df=None,
):

    # ps_dn25 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN25'}}
    ps_dn32 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN32'}}
    ps_dn40 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN40'}}
    ps_dn50 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN50'}}
    # ps_dn65 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN65'}}
    # ps_dn80 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN80'}}
    ps_dn125 = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN125'}}

    # extract data columns from SH dataframe:
    assert isinstance(space_heating, pd.DataFrame)
    sh_dmd = space_heating.loc[:, 'sh_dmd'].copy()
    sh_theta_ff = space_heating.loc[:, 'theta_ff'].copy()
    sh_theta_rf = space_heating.loc[:, 'theta_rf'].copy()

    # maximum pump flow to satisfy the SH demand with a security of 10%
    # sh_ff_pump_ulim = sh_dmd.max() * 1.1

    # rename to shorter name
    gasb_sh_pth = aux_gasboiler_pth
    gasb_sh_sp = aux_gasboiler_theta_ff

    # ff temp for gasboiler, slightly higher than sh theta ff
    sp_theta_gasb = sh_theta_ff + 2.0
    # but limit to at least 70
    if isinstance(sp_theta_gasb, pd.Series):
        sp_theta_gasb = sp_theta_gasb.where(sp_theta_gasb >= 70.0, 70.0)
    else:
        sp_theta_gasb = sp_theta_gasb if sp_theta_gasb >= 70.0 else 70.0

    # get start value if SH theta ff is given as pd.Series
    if isinstance(sh_theta_ff, pd.Series):
        sh_theta_ff_init = sh_theta_ff.iloc[0]
    else:  # constant
        sh_theta_ff_init = sh_theta_ff
    if isinstance(sh_theta_rf, pd.Series):
        sh_theta_rf_init = sh_theta_rf.iloc[0]
    else:  # constant
        sh_theta_rf_init = sh_theta_rf

    # sp for first mixing valve (TES ff and SH rf):
    # slight plus to ascertan that second receives temperature regardless of
    # losses:
    mixing_valve_delta = 1e-2
    # sp_theta_tesrf = sh_theta_ff + mixing_valve_delta  # array
    sp_theta_tesrf_init = sh_theta_ff_init + 1e-2  # singular value
    tes_theta_ff_ctrl_init = sh_theta_ff_init + mixing_valve_delta

    # get max and mean values for scaling
    sh_theta_ff_max, sh_theta_ff_min = np.max(sh_theta_ff), np.min(sh_theta_ff)
    sh_theta_ff_mean = np.mean(sh_theta_ff)
    sh_theta_rf_max, sh_theta_rf_min = np.max(sh_theta_rf), np.min(sh_theta_rf)
    sh_theta_rf_mean = np.mean(sh_theta_rf)

    # scale all values by the sh power factor relative to ref sh demand
    # take max since spread is the lowest with max
    sh_power_factor = (
        sh_dmd.mean() / (sh_theta_ff_max - sh_theta_rf_max) * (75 - 50)
    )
    gasb_factor = (
        aux_gasboiler_pth
        / 150e3
        / (gasb_sh_sp - sh_theta_rf_mean)
        * (75 - 60.0)
    )
    pspecs = {  # get ref pipe specs and mult factor for each part
        'pwp_ff': (ps_dn40, sh_power_factor),
        'p3v_tes_rf': (ps_dn32, sh_power_factor),
        'p_gasb_ff': (ps_dn50, gasb_factor),
        'ph_gasb_core': (ps_dn125, gasb_factor),
    }

    if adjust_pipes:  # adjust pipes by A_i multiplier
        pspecs = _ut.adjust_pipes(pspecs)
    else:  # take pspecs without multiplier
        pspecs = {k: v[0] for k, v in pspecs.items()}

    # pipe with pump into building:
    simenv.add_part(
        _msp.PipeWithPump,
        name='pwp_sh_ff_{0}'.format(segment_name),
        length=1,
        grid_points=2,
        s_ins=0.05,
        lambda_ins=0.03,
        material='carbon_steel',
        pipe_specs=pspecs['pwp_ff'],
        T_init=sh_theta_ff_init,
        T_amb=Tamb,
        ctrl_required=False,
        start_massflow=sh_dmd.iloc[0],
        # lower_limit=0., upper_limit=sh_ff_pump_ulim,
        # maximum_flow=sh_ff_pump_ulim,
        time_series=sh_dmd,
        store_results=True,
    )
    simenv.add_open_port(  # ff into building
        'BC_sh_theta_ff_{0}'.format(segment_name),
        constant=True,
        temperature=sh_theta_ff.mean(),
    )
    simenv.connect_ports(
        'pwp_sh_ff_{0}'.format(segment_name),
        'out',
        'BoundaryCondition',
        'BC_sh_theta_ff_{0}'.format(segment_name),
    )
    # pipe FROM building, first branch into valve, second branch to gasboiler,
    # end into TES
    simenv.add_part(
        _msp.PipeBranched,
        name='pb_sh_rf_{0}'.format(segment_name),
        length=5,
        grid_points=10,
        s_ins=0.1,
        lambda_ins=0.035,
        material='carbon_steel',
        pipe_specs=pspecs['pwp_ff'],
        T_init=sh_theta_rf_init,
        T_amb=Tamb,
        new_ports={'rf_to_valve': [3, 'index'], 'rf_to_gb': [5, 'index']},
        store_results=(0, 3, 5, 9),
    )
    simenv.add_open_port(  # ff into building
        'BC_sh_theta_rf_{0}'.format(segment_name),
        constant=False,
        temperature=sh_theta_rf,
    )
    simenv.connect_ports(
        'BoundaryCondition',
        'BC_sh_theta_rf_{0}'.format(segment_name),
        'pb_sh_rf_{0}'.format(segment_name),
        'in',
    )
    simenv.connect_ports(  # connect end of RF to TES
        'pb_sh_rf_{0}'.format(segment_name),
        'out',
        list(tes_rf_connector.keys())[0],
        list(tes_rf_connector.values())[0],
    )

    if configuration == '4w_valve':
        # second valve, mixing TES ff with gasboiler ff
        tesgb_3wv_min_open = 0.01  # save for offstate of sub controller
        simenv.add_part(
            _msp.PipeWith3wValve,
            name='p3v_sh_tes_gb_{0}'.format(segment_name),
            length=1.5,
            grid_points=3,
            # slightly higher insulation to reduce losses between mixing vales
            insulation_thickness=0.2,
            insulation_lambda=0.035,
            material='carbon_steel',
            pipe_specs=pspecs['pwp_ff'],
            T_amb=Tamb,
            T_init=np.hstack(
                (
                    np.full(1, tes_theta_ff_ctrl_init),
                    np.full(2, sh_theta_ff_init),
                )
            ),
            valve_location=1,
            lower_limit=tesgb_3wv_min_open,
            upper_limit=0.99,
            start_portA_opening=1,
            store_results=True,
        )
        simenv.connect_ports(
            'p3v_sh_tes_gb_{0}'.format(segment_name),
            'AB',
            'pwp_sh_ff_{0}'.format(segment_name),
            'in',
        )
        # first valve, mixing TES ff with SH rf
        simenv.add_part(
            _msp.PipeWith3wValve,
            name='p3v_sh_tes_rf_{0}'.format(segment_name),
            length=3,
            grid_points=6,
            # high insulation to reduce losses between mixing vales
            insulation_thickness=0.5,
            insulation_lambda=0.0035,
            material='carbon_steel',
            pipe_specs=pspecs['p3v_tes_rf'],
            T_amb=Tamb,
            T_init=tes_theta_ff_ctrl_init,
            valve_location=4,
            lower_limit=0.01,
            upper_limit=0.99,
            start_portA_opening=1,
            store_results=(0, 3, 4, 5),
        )
        simenv.connect_ports(  # connect first and second mix valve
            'p3v_sh_tes_rf_{0}'.format(segment_name),
            'AB',
            'p3v_sh_tes_gb_{0}'.format(segment_name),
            'B',
        )
        simenv.connect_ports(  # connect first valve to tes
            list(tes_ff_connector.keys())[0],
            list(tes_ff_connector.values())[0],
            'p3v_sh_tes_rf_{0}'.format(segment_name),
            'A',
        )
        simenv.connect_ports(  # connect first valve to SH rf
            'p3v_sh_tes_rf_{0}'.format(segment_name),
            'B',
            'pb_sh_rf_{0}'.format(segment_name),
            'rf_to_valve',
        )
    else:  # only 4w valve implemented so far
        raise NotImplementedError()

    # make auxiliary gas boiler:
    # forward flow needs additional pipe to have a slight delay in heat
    # transportation. extended GB core length would result in a pipe which is
    # too big, thus a new pipe with own specs. RF delay is not relevant for
    # controls, thus direct connection.
    simenv.add_part(  # gasboiler forward flow
        _msp.Pipe,
        name='p_sh_gasb_ff_{0}'.format(segment_name),
        length=2.5,
        grid_points=8,
        insulation_thickness=0.1,
        insulation_lambda=0.035,
        material='carbon_steel',
        pipe_specs=pspecs['p_gasb_ff'],
        T_amb=Tamb,
        T_init=np.full(8, 75.0),
        store_results=(0, 1, 4, 7),
    )
    simenv.connect_ports(
        'p3v_sh_tes_gb_{0}'.format(segment_name),
        'A',
        'p_sh_gasb_ff_{0}'.format(segment_name),
        'out',
    )
    # GASBOILER CORE
    aux_gasb_cells = 6  # num of grid points
    aux_gasb_heated_cells = [1, aux_gasb_cells - 1]  # heat spread range
    last_heated_cell = max(aux_gasb_heated_cells) - 1  # last cell for control
    simenv.add_part(  # Rohr von Gasboiler
        _msp.HeatedPipe,
        name='ph_sh_gasb_{0}'.format(segment_name),
        length=2.5,
        grid_points=6,
        insulation_thickness=0.3,
        insulation_lambda=0.035,
        material='carbon_steel',
        pipe_specs=pspecs['ph_gasb_core'],
        T_amb=Tamb,
        T_init=np.full(6, 75.0),
        heat_spread='range',
        heated_cells=[1, 5],
        no_control=False,
        lower_limit=0.0,
        upper_limit=gasb_sh_pth,
        store_results=True,
    )
    simenv.connect_ports(  # connect gasboiler to ff pipe
        'p_sh_gasb_ff_{0}'.format(segment_name),
        'in',
        'ph_sh_gasb_{0}'.format(segment_name),
        'out',
    )
    simenv.connect_ports(  # connect gasboiler rf directly to SH rf
        'ph_sh_gasb_{0}'.format(segment_name),
        'in',
        'pb_sh_rf_{0}'.format(segment_name),
        'rf_to_gb',
    )

    # CONTROLS
    simenv.add_control(  # control mixing valve TES + SH rf
        _msp.PID,
        name='c_sh_3v_tesrf_{0}'.format(segment_name),
        terms='PI',
        actuator='p3v_sh_tes_rf_{0}'.format(segment_name),
        controlled_part='p3v_sh_tes_rf_{0}'.format(segment_name),
        controlled_port=5,
        actuator_port='A',
        process_CV_mode='part_specific',
        reference_part='none',
        time_domain='discrete',
        sub_controller=False,
        off_state=0.0,
        setpoint=sp_theta_tesrf_init,
        deadtime=0,
        loop_tuning='manual',
        Kp=0.04,
        Ki=1 / 100,
        adapt_coefficients=True,
        norm_timestep=1.0,
        anti_windup=1.0,
        CV_saturation=(0.0, 1.0),
        slope=(0.0, 0.0),
        invert=False,
    )
    # control mixing valve TES + Gasboiler FF (After first valve)
    simenv.add_control(
        _msp.PID,
        name='c_sh_3v_tesgb_{0}'.format(segment_name),
        terms='PI',
        actuator='p3v_sh_tes_gb_{0}'.format(segment_name),
        controlled_part='p3v_sh_tes_gb_{0}'.format(segment_name),
        controlled_port=2,
        actuator_port='A',
        process_CV_mode='part_specific',
        reference_part='none',
        time_domain='discrete',
        sub_controller=False,
        off_state=tesgb_3wv_min_open,
        setpoint=sh_theta_ff_init,
        deadtime=0,
        loop_tuning='manual',
        Kp=0.04,
        Ki=1 / 100,
        adapt_coefficients=True,
        norm_timestep=1.0,
        anti_windup=1.0,
        CV_saturation=(0.0, 1.0),
        slope=(0.0, 0.0),
        invert=False,
    )
    # control gasboiler ff temp if controller is on
    kp_sh_gasboiler = gasb_sh_pth * (gasb_sh_sp - sh_theta_ff_mean) / 500
    simenv.add_control(  # control on/off state of gas boiler
        _msp.PID,
        name='c_sh_gasb_pth_{0}'.format(segment_name),
        terms='PI',
        actuator='ph_sh_gasb_{0}'.format(segment_name),
        process_CV_mode='direct',
        controlled_part='ph_sh_gasb_{0}'.format(segment_name),
        controlled_port=last_heated_cell,
        reference_part='none',
        setpoint=gasb_sh_sp,
        sub_controller=True,
        off_state=0.0,
        master_type='controller',
        master_controller='c_sh_3v_tesgb_{0}'.format(segment_name),
        dependency_kind='concurrent',
        time_domain='continuous',
        deadtime=0,
        loop_tuning='manual',
        Kp=kp_sh_gasboiler,
        Ki=kp_sh_gasboiler / 10.0,
        adapt_coefficients=True,
        norm_timestep=1.0,
        anti_windup=gasb_sh_pth / 100,
        slope=(-gasb_sh_pth / 30, gasb_sh_pth / 30),
        CV_saturation=(0.0, gasb_sh_pth),
        invert=False,
        silence_slope_warning=True,
    )

    # save calculations to df and return if given:
    if calculations_df is not None:
        calculations_df.loc[
            'pipe_scaling_sh', 'space_heating{0}'.format(segment_name)
        ] = sh_power_factor
        calculations_df.loc[
            'pipe_scaling_sh_gasboiler',
            'space_heating{0}'.format(segment_name),
        ] = gasb_factor
        calculations_df.loc[
            'theta_sh_ff_max', 'space_heating{0}'.format(segment_name)
        ] = sh_theta_ff_max
        calculations_df.loc[
            'theta_sh_rf_max', 'space_heating{0}'.format(segment_name)
        ] = sh_theta_rf_max
        calculations_df.loc[
            'theta_sh_ff_mean', 'space_heating{0}'.format(segment_name)
        ] = sh_theta_ff_mean
        calculations_df.loc[
            'theta_sh_rf_mean', 'space_heating{0}'.format(segment_name)
        ] = sh_theta_rf_mean
        calculations_df.loc[
            'theta_sh_ff_min', 'space_heating{0}'.format(segment_name)
        ] = sh_theta_ff_min
        calculations_df.loc[
            'theta_sh_rf_min', 'space_heating{0}'.format(segment_name)
        ] = sh_theta_rf_min
        calculations_df.loc[
            'theta_sh_gasb_ff', 'space_heating{0}'.format(segment_name)
        ] = gasb_sh_sp

        return calculations_df
