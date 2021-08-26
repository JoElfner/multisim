[![PyPi Package](https://badge.fury.io/py/MultiSim.svg)](https://badge.fury.io/py/MultiSim)
[![License](https://img.shields.io/github/license/JoElfner/multisim.svg)](https://github.com/JoElfner/multisim/blob/master/LICENSE)
[![TravisCI Build](https://travis-ci.com/JoElfner/multisim.svg?branch=master)](https://travis-ci.com/JoElfner/multisim)
[![Appveyor Build](https://ci.appveyor.com/api/projects/status/uc42tex22gkcgaxo/branch/master?svg=true)](https://ci.appveyor.com/project/JoElfner/multisim)
[![CodeCoverage](https://codecov.io/gh/JoElfner/multisim/branch/master/graph/badge.svg)](https://codecov.io/gh/JoElfner/multisim)


----------------

# MultiSim - A simulation tool for energy systems

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

### Basic parts
The following basic parts are currently available:
* Pipe
* Thermal storage (TES)
* Heat exchanger

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
There is also a list of [**sensors/meters**](multisim/utility_functions.py#L2443) (file utility_functions.py requires *heavy* refactoring...), which can be "installed" at any (numeric) cell of each part to track the state of this cell or perform calculations like energy flows, cumulated mass and energy flows etc. on the fly, such as:
* Temperature sensor
* Mass flow sensor
* Heat meter (power, mass flow, volume flow, temperature of hot and cold part, cumulated values)

## Examples

coming soon

## Validation

MultiSim has been fully validated following standard ANSI/BPI-2400-S-2015. A stricter set of statistic measures than provided in the standard has been used.

Since MultiSim was part of a PhD thesis, validation results will be added as soon as the thesis has been published (approximately December 2021).


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
