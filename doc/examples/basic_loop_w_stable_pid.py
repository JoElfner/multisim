# -*- coding: utf-8 -*-

import pandas as pd

import multisim as ms

sp_pid = 40.  # setpoint for the PID in degree celsius
theta_low = 20.0  # degree celsius
theta_high = pd.Series(
    data=50.0, index=pd.date_range('2021-01-01', periods=2000, freq='1s')
)
theta_high.iloc[900:] = 85.0

pipe_specs_var = {
    'in': {'pipe_type': 'EN10255_medium', 'DN': 'DN25'},
    'out': {'pipe_type': 'EN10255_medium', 'DN': 'DN25'},
}
pipe_specs = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN25'}}

general_specs = dict(
    insulation_thickness=1e-2,
    insulation_lambda=0.035,
    T_init=theta_low,
    T_amb=theta_low,
    material='carbon_steel',
    pipe_specs=pipe_specs,
)

my_sim_a = ms.Models()

my_sim_a.set_disksaving(save=False, start_date='infer')

my_sim_a.set_timeframe(timeframe=1800, adaptive_steps=True)

my_sim_a.set_solver(solver='heun', allow_implicit=False)

my_sim_a.add_part(
    part=ms.ap.PipeWith3wValve,
    name='pipe_in',
    length=2.0,
    grid_points=20,  # you will be prompted to enter this after initializing
    valve_location=5,
    start_portA_opening=0.1,
    lower_limit=0.01,
    upper_limit=0.99,
    **general_specs,
)
my_sim_a.add_part(
    part=ms.ap.Tes,
    name='TES',
    volume=2.0,
    grid_points=50,
    outer_diameter=1.0,
    shell_thickness=5e-3,
    new_ports=None,
    **general_specs,
)
my_sim_a.add_part(
    part=ms.ap.PipeWithPump,
    name='pipe_out',
    length=1.0,
    grid_points=10,
    start_massflow=0.1,
    ctrl_required=False,
    const_val=0.1,
    **general_specs,
)

my_sim_a.add_open_port('BC_theta_low', constant=True, temperature=theta_low)
my_sim_a.add_open_port('BC_theta_high', constant=False, temperature=theta_high)
my_sim_a.add_open_port('BC_out', constant=True, temperature=theta_low)

my_sim_a.connect_ports(
    first_part='BoundaryCondition',
    first_port='BC_theta_low',
    scnd_part='pipe_in',
    scnd_port='A',
)
my_sim_a.connect_ports(
    first_part='BoundaryCondition',
    first_port='BC_theta_high',
    scnd_part='pipe_in',
    scnd_port='B',
)
my_sim_a.connect_ports(
    first_part='pipe_in', first_port='AB', scnd_part='TES', scnd_port='in',
)
my_sim_a.connect_ports(
    first_part='TES',
    first_port='out',
    scnd_part='pipe_out',
    scnd_port='in',
)
my_sim_a.connect_ports(
    first_part='pipe_out',
    first_port='out',
    scnd_part='BoundaryCondition',
    scnd_port='BC_out',
)


my_sim_a.add_control(ms.ap.PID, name='pid_valve', actuator='pipe_in', process_CV_mode='part_specific',
                     CV_saturation=(0,1), controlled_part='pipe_in', controlled_port=-1, reference_part='none',
                     setpoint=sp_pid, sub_controller=False, off_state=0., time_domain='continuous', deadtime=0.,
                     slope=(-.1, .1), invert=True,
                     terms='PID', loop_tuning='manual', Kp=.01, Ki=.0005, Kd=.00005,
                     adapt_coefficients=True, norm_timestep=1., filter_derivative=False,
                     anti_windup=.5)

my_sim_a.initialize_sim()

my_sim_a.start_sim()
