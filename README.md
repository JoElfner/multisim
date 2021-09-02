[![PyPi Package](https://badge.fury.io/py/MultiSim.svg)](https://badge.fury.io/py/MultiSim)
[![License](https://img.shields.io/github/license/JoElfner/multisim.svg)](https://github.com/JoElfner/multisim/blob/master/LICENSE)
[![TravisCI Build](https://travis-ci.com/JoElfner/multisim.svg?branch=master)](https://travis-ci.com/JoElfner/multisim)
[![Appveyor Build](https://ci.appveyor.com/api/projects/status/uc42tex22gkcgaxo/branch/master?svg=true)](https://ci.appveyor.com/project/JoElfner/multisim)
[![CodeCoverage](https://codecov.io/gh/JoElfner/multisim/branch/master/graph/badge.svg)](https://codecov.io/gh/JoElfner/multisim)


----------------

# MultiSim - A simulation tool for energy systems

![MultiSim logo](/doc/figures/logo.svg)

MultiSim is a simulation tool for energy systems consisting of multiple different parts like pipes, storages, valves, heat exchangers, etc.
F.i. technical appliances such as space or water heating appliances can be designed, simulated and optimized.

MultiSim was mainly designed to solve the [convection-diffusion-reaction-equation](https://en.wikipedia.org/wiki/Convection_diffusion_equation#General) but can also be utilized to solve other differential equations.
It features an adaptive step size solver, which is capable of yielding a stable solution even when loads fluctuate vigorously.
Furthermore the solver can handle steps in flow variables due to interference of controllers (f.i. PID).

- **Developer/maintainer:** https://www.linkedin.com/in/johannes-elfner

## Short documentation

The short documentation consists of:
* [Installation instructions](#install)
* Description of available [Components](#components)
* [Getting startet](#getting-started)
* Basic [examples](#Examples)
* Summary of the [validation](#Validation) of the simulation software
* [Known limitations and To-do](#known-limitations-and-to-do)

## Install

To install the newest snapshot of MultiSim:
1. Make a local clone of this repository or download the release to the installation path
2. `cd` the console to the download/clone folder
3. `pip install -e .` to install MultiSim in editable mode or `pip install .` if you want a fixed installation.

The newest snapshot of the branch `master` is always **fully operational**, but may include code which will be deprecated in the next release.
If you want to stick to fixed releases, you can stick to the version published on PyPi and install MultiSim with:
1. `pip install MultiSim`

Fixed releases may be **outdated** by several months, so I recommend cloning this repository.

For slightly more detailed building, distribution and installation instructions, see [INSTALL.rst](INSTALL.rst).

## Components

MultiSim supports different types of components, most notably:
* [Parts](#Basic-parts) which require differential equations to be solved (called "basic parts" hereafter)
* [Connectors and Actuators](#Actuators-and-connectors) which can effect flow variables, either controlled by controllers, stationary or time series based
* [Controllers](#Controllers) which control actuators
* Boundary Conditions like in- and outflows to ambience, both stationary and time series based
* [Compound parts](#Compound-parts) consisting of multiple parts, actuators, controllers and boundary conditions
* [Meters](#Meters) to track process variables of specific important parts/cells and perform basic calculations on-the-fly
* [Utility tools](#Utility-tools) for pre- and postprocessing, plotting, loading and storing data, ...

### Basic parts
The following basic parts are currently available:
* Pipe
* Thermal storage (TES)
* Heat exchanger
    * Numeric (by solution of differential equations)
    * Non-numeric *condensing flue gas* heat exchanger based on fitting an ElasticNet regression to measurement data (*very* specific to the type of HEX)

Parts derived by class inheritance of the basic parts:
* Heated pipe
* Branched pipe, pipe with valve, pipe with pump (these are also compound parts)

### Actuators and connectors
The following actuators and connectors can be installed:
* Three way connector
* Mixing valve/splitting valve/three way valve
* Branch
* Pump
* Connector to ambient conditions/boundary conditions

All actuators/connectors can be controlled by controllers, set to a static value or follow a predefined behaviour by defining a time series.

### Controllers
[parts/controllers](multisim/parts/controllers.py) defines the following controllers:
* PID controller
* Bang–bang controller (also 2 step or on–off controller) with a hysteresis
* Two sensor controller (switch on when sensor 1 is > or < than setpoint 1, switch off when sensor 2 is > or < than setpoint 2)
* Model predictive controller (CHP plant specific to optimally follow a predicted electric profile)

All controllers support setting:
* Fixed setpoints and setting the setpoint to a process variable of another part
* Control variable saturation limits
* Part specific control variable post processing like conversion to a specific value range
* Setting slopes to control variable changes
* Linking controllers to make controller action depend on another controller to construct control chains

PID controllers additionally support semi-automatic tuning by Ziegler-Nichols method.
Thus preferred tuning method for PID controllers is Ziegler-Nichols, since the parameters `Kp_crit` and `T_crit` can be passed directly to the controller while specifying the aggressiveness of the PID controller with rules like `classic` or `pessen-int` (Pessen integral rule).

### Compound parts
Compound parts consisting of multiple other parts and controllers can be found in [parts/part_modules](multisim/parts/part_modules).
Part dimensions, such as pipe diameters, and controller coefficients have been fit to a wide range of flow speeds and temperatures, but may be adjusted if controls show instabilities or if the solver requires too many retries to find a stable solution.
The following compound parts can be used:
* Gas boiler
* Chp plant, also with flue gas heat exchanger (based on fitting a model to manufacturer specific measurement data)
* Consumer appliances
    * Space heating
    * State-of-the-art water heating
    * Low exergy water heating

New parts and controllers can be added either by defining completely new classes or by inheriting from existing parts.

### Meters
There is also a list of [**sensors/meters**](multisim/_utility/meters.py) which can be "installed" at any (numeric) cell of each part to track the state of this cell or perform calculations like energy flows, cumulated mass and energy flows etc. on the fly, such as:
* Temperature sensor
* Mass flow sensor
* Heat meter (power, mass flow, volume flow, temperature of hot and cold part, cumulated values)

### Utility tools
The file [utility_functions](multisim/utility_functions.py) provides methods for pre- and post-processing of input/output data and basic plotting/statistic analysis. Also methods to open the `*.hdf5` files, which are used to store the results on disk, are provided.

**But**: `utility_functions.py` requires **heavy** refactoring!! This is scheduled for the next release.

Some parts have already been refactored to [multisim/_utility/](multisim/_utility):
* [Meters](multisim/_utility/meters.py), also see section [Meters](#Meters)
* [plotting](multisim/_utility/plotting.py) provides basic plotting methods for validation and heatmap plots and also some formatting helpers. More will be added soon.
* [Statistical error measures](multisim/_utility/stat_error_measures.py) provides basic error measures useful for validation, such as the (adjusted) coefficient of determination, MSE, RMSE, CV(RMSE), NME, ...

## Getting started
Import MultiSim and create your simulation environment instance with:
```python
import multisim as ms

my_sim = ms.SimEnv()
```

Now simply follow the detailed step-by-step instructions printed to the console.

## Examples
We will cover three basic examples in this section, all covering the temperature control of a three-way-valve flowing into a thermal energy storage (TES):
1. A [stable PID controller](#Stable-PID-controller) (loop tuned with Ziegler-Nichols)
2. An [instable PID controller](#Instable-PID-controller). Stable at first for small steps in the process variable, but instable with persisting oscillations for larger steps.
3. A [bang-bang controller](#Bang-bang controller) to control the pump.

The appliance/setup to simulate is, in all three cases, the following:

![example scheme PID](/doc/examples/figures/example_scheme_PID.svg)

With the temperature of the water flowing into port B of 'pipe_in' describing a step from 50.0 °C to 85.0 °C after 300 s.

### Stable PID controller
This small example covers controlling the mixing temperature of the three-way-valve via a stable PID controller.
The PID controller is tuned using the class Ziegler-Nichols rule.

The full executable example as a Python script can be found at [doc/examples/basic_loop_w_stable_pid.py](doc/examples/basic_loop_w_stable_pid.py).

To set up the simulation environment, first start by loading the required modules and defining boundary conditions like the temperatures and temperature time series:
```python
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
```

Now create an instance of the simulation environment, and set some basic options:
```python
# create simulation environment
my_sim_a = ms.Models()
# set disksaving, simulatiion timeframe and solver
my_sim_a.set_disksaving(save=True, start_date='infer', sim_name='sim_a')
my_sim_a.set_timeframe(timeframe=900, adaptive_steps=True)
my_sim_a.set_solver(solver='heun', allow_implicit=False)
```

Next define specifications for pipes and the ports at each pipe. Dimensions can be set individually for each port. In this case all ports and parts share the same specifications.
```python
# define pipe specifications for all pipes and ports
pipe_specs = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN25'}}
# set general specifications for all parts
general_specs = dict(
    insulation_thickness=1e-2,  # insulation around pipe in meters
    insulation_lambda=0.035,  # insulation heat conductivity in W/(m*K)
    T_init=theta_low,  # initial temperature of all cells
    T_amb=theta_low,  # ambient temperature of all parts
    material='carbon_steel',
    pipe_specs=pipe_specs,
)
```

Now add the parts to the simulation environment:
```python
my_sim_a.add_part(  # add pipe_in with the valve to control
    part=ms.ap.PipeWith3wValve,
    name='pipe_in',
    length=2.0,  # in meters
    grid_points=20,  # number of numeric cells to calculate
    valve_location=5,  # location of the three-way-valve in grid_points
    start_portA_opening=0.5,  # initialize the valve
    lower_limit=0.0,  # lower limit for the valve, can be 0 <= x < 1
    upper_limit=1.0,  # upper limit for the valve, can be 0 < x <= 1
    **general_specs,
)
my_sim_a.add_part(
    part=ms.ap.Tes,  # add a thermal energy storage
    name='TES',
    volume=0.5,  # volume in m**3
    grid_points=20,
    outer_diameter=1.0,  # outer diameter in meters
    shell_thickness=5e-3,  # shell/casing thickness in meters
    new_ports=None,  # add no additional ports/connectors
    **general_specs,
)
my_sim_a.add_part(
    part=ms.ap.PipeWithPump,  # add a pipe with a pump
    name='pipe_out',
    length=1.0,
    grid_points=10,
    start_massflow=0.75,  # initialize massflow in kg/s
    ctrl_required=False,  # set to constant or time series based
    const_val=0.75,  # constant massflow
    **general_specs,
)
```

Add open ports (connections to ambient conditions, connections crossing the
control volume of the simulation environment, other boundary conditions (BC)).
Open ports can be either constant or time series based.
```python
my_sim_a.add_open_port('BC_theta_low', constant=True, temperature=theta_low)
my_sim_a.add_open_port('BC_theta_high', constant=False, temperature=theta_high)
my_sim_a.add_open_port('BC_out', constant=True, temperature=theta_low)
```

Connect parts at ports and also boundary conditions to parts:
```python
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
```

Add and set PID control to control the 3-way-valve.
Nomenclature: setpoint (SP), control variable (CV), process variable (PV).
For this, the critical coefficient `Kp_crit` causing permanent oscillations and the period of the oscillations `T_crit` have to be estimated before. This can be done by increasing `Kp` of a PID controller set to `loop_tuning='manual'` and `terms='P'` until oscillations after a step start pertaining.
```python
my_sim_a.add_control(
    ms.ap.PID,
    name='pid_valve',
    actuator='pipe_in',  # controlled actuator
    process_CV_mode='part_specific',  # allow post-processing of CV in part
    CV_saturation=(0.0, 1.0),  # clip CV
    controlled_part='pipe_in',  # part where the PV is found
    controlled_port=-1,  # port or cell where the PV is found in its part
    reference_part='none',  # use another part as source of the SP
    setpoint=sp_pid,  # use defined constant value
    sub_controller=False,  # controller action is not depending on another ctrl
    off_state=0.0,  # which value shows that the controller is off?
    time_domain='discrete',  # integral and derivative calculation type
    deadtime=0.0,  # in seconds
    slope=(-0.1, 0.1),  # in units/s
    invert=False,  # invert action to allow reversed operation
    terms='PID',  # which coefficients to use
    loop_tuning='ziegler-nichols',  # semi-automatic loop tuning or manual?
    rule='classic',  # loop tuning rule
    Kp_crit=0.025,  # critical Kp value
    T_crit=5.0,  # period of the oscillations in seconds
    filter_derivative=False,  # low pass filter of the derivative term
    anti_windup=1.0,  # anti windup for the integral term
)
```

Initialize simulation (set up parts and controllers, preallocate arrays,
calculate topology...) and run it:
```python
my_sim_a.initialize_sim()
my_sim_a.start_sim()
```

Add meters:
```python
meters = ms.Meters(my_sim_a, start_time=theta_high.index[0])
meters.temperature(name='theta_mix', part='pipe_in', cell=-1)
meters.heat_meter(
    name='hm',
    warm_part='pipe_in',
    warm_cell=-1,
    cold_part='pipe_out',  # massflows will be calculted on the cold cell
    cold_cell=0,
)
meters.massflow(name='mflow_A', part='pipe_in', cell=0)
meters.massflow(name='mflow_AB', part='pipe_in', cell=-1)
```

return results as a dictionary of kind `{part:{'res': temperatures, 'dm': massflows}}`:
```python
results = my_sim_a.return_stored_data()
```

For plotting of the results, define the plot index:
```python
plot_idx = results['TES']['res'].index
```

Now plot the temperatures and massflows of the valve respectively of part `'pipe_in'`:
```python
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
```
As you can see, the PID tuned with the Ziegler-Nichols method reaches the new SP quite fast and stable with only minor oscillations:
![Valve temperature and massflow](/doc/examples/figures/basic_example_valve.svg)

And finally plot a heatmap of the TES temperature:
```python
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
```
Which yields the temperature of the TES over the time:
![TES temperature](/doc/examples/figures/basic_example_tes.png)


### Instable PID controller

### Bang-bang controller

## Validation

MultiSim has been fully validated following standard ANSI/BPI-2400-S-2015. A stricter set of statistic measures than provided in the standard has been used.

Since MultiSim was part of a PhD thesis, validation results will be added as soon as the thesis has been published (approximately December 2021).

One of the systems simulated to validate MultiSim, was the following heating and PWH (potable water hot) appliance:

![validation appliance scheme](/doc/figures/sim_valild_chp_scheme.svg)

The 2D-KDE prediction-realization plot, created with [multisim.plotting.prediction_realization_2d_kde](multisim/utility/plotting.py) shows that almost all points lie close to the halving diagonal:

![validation appliance kde](/doc/figures/valid_chp_kde_en.svg)
This is also confirmed by the statistical error measures in the plots. The coefficient of determination is, in both cases, `>0.9`. ANSI defines validation bounds of `CV(RMSE) <= 0.3` and `|NME| <= 0.05`. Both bounds are satisfied.

Finally, the heatmap plot of the TES temperature, plotted with [multisim.plotting.heatmap_from_df](multisim/utility/plotting.py), shows only minor differences:

![validation appliance heatmap](/doc/figures/valid_chp_tes_temperature_en.png)

Further validation results of other parts and with more plots will be added to the full documentation as soon as everything is officially published and can be referenced.

## Known limitations and To-do

Even though MultiSim is **fully operational**, many things have to be *refactored*,
*replaced*, *improved* or *deprecated*. Especially the core class `SimEnv` and the
most basic parts like pipes and TES require a general overhaul. Thus
enhancements should start here.

Furthermore current tests used for TDD are based on **proprietary measurement
data**. Thus these tests **cannot be published**. Hence tests included in this
public GitHub repo are merely truncated stumps. Using free data to integrate
extensive tests will be an important step.

The documentation is currently missing but will be added step by step.

Other enhancements could be:

1. Extend the tests using `pytest` with non-proprietary data.

2. Implementing the implicit differential equation solver in numba to speed things up considerably. Implicit solving is currently slowing down the simulation progress.

3. Move the outer explicit solver loop to numba. This should also improve the performance by several percent points.

4. Check if local Nusselt number calculation is implemented in each relevant part.

5. Fully implement parts with compound structures.

6. Refactor type checks during part-adding as much as possible, using `@property` may help. Move away from using to many `kwargs`-based arguments. Include more specific type hints and default args handling.

7. Move to Python 3.8, using type hints and assignment expressions.

8. There is a lot of chaos in `utility_functions`. This needs some heavy refactoring and tidying.

9. Write a documentation.

MultiSim depends mainly on `numpy`, `numba`, `pandas`, `scipy`, and
`matplotlib`. For some parts `scikit-learn` is a dependency.
