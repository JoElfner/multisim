# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import multisim as ms

# define temperatures
sp_pid = 40.0  # setpoint for the PID in degree celsius
theta_low = 20.0  # degree celsius
theta_high = pd.Series(
    data=50.0, index=pd.date_range('2021-01-01', periods=1000, freq='1s')
)
theta_high.iloc[300:] = 85.0
# define pipe specifications for all pipes and ports
pipe_specs = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN25'}}
# set general specifications for all parts
general_specs = dict(
    insulation_thickness=1e-2,
    insulation_lambda=0.035,
    T_init=theta_low,
    T_amb=theta_low,
    material='carbon_steel',
    pipe_specs=pipe_specs,
)

# create simulation environment
my_sim_a = ms.Models()
# set disksaving, simulatiion timeframe and solver
my_sim_a.set_disksaving(save=True, start_date='infer', sim_name='sim_a')
my_sim_a.set_timeframe(timeframe=900, adaptive_steps=True)
my_sim_a.set_solver(solver='heun', allow_implicit=False)

# add parts
my_sim_a.add_part(
    part=ms.ap.PipeWith3wValve,
    name='pipe_in',
    length=2.0,
    grid_points=20,
    valve_location=5,
    start_portA_opening=0.5,
    lower_limit=0.0,
    upper_limit=1.0,
    **general_specs,
)
my_sim_a.add_part(
    part=ms.ap.Tes,
    name='TES',
    volume=0.5,
    grid_points=20,
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
    start_massflow=0.75,
    ctrl_required=False,
    const_val=0.75,
    **general_specs,
)

# add open ports (connections to ambient conditions, connections crossing the
# control volume of the simulation environment)
my_sim_a.add_open_port('BC_theta_low', constant=True, temperature=theta_low)
my_sim_a.add_open_port('BC_theta_high', constant=False, temperature=theta_high)
my_sim_a.add_open_port('BC_out', constant=True, temperature=theta_low)

# connect parts and also boundary conditions to parts
my_sim_a.connect_ports(
    first_part='BoundaryCondition',
    first_port='BC_theta_low',
    scnd_part='pipe_in',
    scnd_port='B',
)
my_sim_a.connect_ports(
    first_part='BoundaryCondition',
    first_port='BC_theta_high',
    scnd_part='pipe_in',
    scnd_port='A',
)
my_sim_a.connect_ports(
    first_part='pipe_in', first_port='AB', scnd_part='TES', scnd_port='in',
)
my_sim_a.connect_ports(
    first_part='TES', first_port='out', scnd_part='pipe_out', scnd_port='in',
)
my_sim_a.connect_ports(
    first_part='pipe_out',
    first_port='out',
    scnd_part='BoundaryCondition',
    scnd_port='BC_out',
)

# add and set PID control to control the 3-way-valve
my_sim_a.add_control(
    ms.ap.PID,
    name='pid_valve',
    actuator='pipe_in',
    process_CV_mode='part_specific',
    CV_saturation=(0.0, 1.0),
    controlled_part='pipe_in',
    controlled_port=-1,
    reference_part='none',
    setpoint=sp_pid,
    sub_controller=False,
    off_state=0.0,
    time_domain='discrete',
    deadtime=0.0,
    slope=(-0.1, 0.1),
    invert=False,
    terms='PID',
    loop_tuning='ziegler-nichols',
    rule='classic',
    Kp_crit=0.025,
    T_crit=5.0,
    filter_derivative=False,
    anti_windup=1.0,
)

# initialize simulation (set up parts and controllers, preallocate arrays,
# calculate topology...)
my_sim_a.initialize_sim()
my_sim_a.start_sim()

# add meters:
meters = ms.Meters(my_sim_a, start_time=theta_high.index[0])
meters.temperature(name='theta_mix', part='pipe_in', cell=-1)
meters.heat_meter(
    name='hm',
    warm_part='pipe_in',
    warm_cell=-1,
    cold_part='pipe_out',
    cold_cell=0,
)
meters.massflow(name='mflow_A', part='pipe_in', cell=0)
meters.massflow(name='mflow_AB', part='pipe_in', cell=-1)

# return results as a dictionary of kind
# {part:{'res': temperatures, 'dm': massflows}}
results = my_sim_a.return_stored_data()


# %% plot results
# plot index is the simulation time in seconds
plot_idx = (
    results['TES']['res'].index - results['TES']['res'].index[0]
).astype(int) / 1e9

plot_idx = results['TES']['res'].index

# %%% plot valve data
fig_valve, (ax_valve_flow, ax_valve_theta) = plt.subplots(
    1, 2, sharex=True, figsize=(16 / 2.54, 6 / 2.54)
)
# plot massflows
ax_valve_flow.plot(
    plot_idx, results['meters']['mflow_A'], label=r'$\dot{m}$ port A'
)
ax_valve_flow.plot(
    plot_idx,
    results['meters']['mflow_AB'].sub(results['meters']['mflow_A'].values),
    label=r'$\dot{m}$ port B',
)
ax_valve_flow.plot(
    plot_idx, results['meters']['mflow_AB'], label=r'$\dot{m}$ total'
)
# plot temperatures
ax_valve_theta.hlines(
    theta_low, plot_idx[0], plot_idx[-1], label=r'$\theta$ port A'
)
ax_valve_theta.plot(
    plot_idx, theta_high.reindex(plot_idx), label=r'$\theta$ port B'
)
ax_valve_theta.plot(
    plot_idx, results['meters']['theta_mix'], label=r'$\theta$ mix'
)
# legends, ax labels, formatting and layout
ax_valve_flow.legend()
ax_valve_theta.legend(loc='center right')
ax_valve_flow.set_ylabel(r'massflow $\dot{m}$ in kg/s')
ax_valve_theta.set_ylabel(r'temperataure $\theta$ in °C')
ax_valve_flow.set_xlabel('simulation time in min:s')
ax_valve_theta.set_xlabel('simulation time in min:s')
ax_valve_flow.xaxis.set_major_formatter(mpl.dates.DateFormatter('%M:%S'))
fig_valve.tight_layout(pad=0.1)

# fig_valve.savefig('./figures/basic_example_valve.svg')

# %% plot TES temperature as heatmap

# resample and select every second point in y-axis to reduce plot size:
tes_heatmap_rs = results['TES']['res'].resample('5s').mean().iloc[:, ::2]

fig_tes = plt.figure(figsize=(16.0 / 2.54, 6.0 / 2.54))
ax_tes = fig_tes.gca()

ms.plotting.heatmap_from_df(
    tes_heatmap_rs,
    ax=ax_tes,
    ylabel=('TES height', 'm'),
    cbar=True,
    cbar_label=(r'Temperature\; $\theta$', '°C'),
    vmin=20.0,
    plt_kwds={'shading': 'gouraud'},
)

# fig_tes.savefig('./figures/basic_example_tes.png', dpi=200)
