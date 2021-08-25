.. -*- mode: md -*-

|

.. image:: https://badge.fury.io/py/MultiSim.svg
  :target: https://badge.fury.io/py/MultiSim

.. image:: https://img.shields.io/github/license/JoElfner/multisim.svg
  :target: https://github.com/JoElfner/multisim/blob/master/LICENSE

.. image:: https://travis-ci.com/JoElfner/multisim.svg?branch=master
  :target: https://travis-ci.com/JoElfner/multisim

.. image:: https://ci.appveyor.com/api/projects/status/uc42tex22gkcgaxo/branch/master?svg=true
  :target: https://ci.appveyor.com/project/JoElfner/multisim

.. image:: https://codecov.io/gh/JoElfner/multisim/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/JoElfner/multisim

----------------

MultiSim
========

- **Developer/maintainer:** https://www.linkedin.com/in/johannes-elfner

MultiSim is a simulation tool for energy systems consisting of multiple parts like pipes, storages, valves, heat exchangers, etc.
F.i. technical appliances such as space or water heating appliances can be constructed and simulated.

MultiSim was mainly designed to solve the [convection-diffusion-reaction-equation](https://en.wikipedia.org/wiki/Convection_diffusion_equation#General) but can also be utilized to solve other differential equations.
It features an adaptive step size solver, which is capable of yielding a stable solution even when loads fluctuate vigorously.
Furthermore the solver can handle steps in flow variables due to interference of controllers (f.i. PID).

The following standard parts are currently available:
* Pipe
* thermal storage (TES)
* heat exchanger
* three way connector and controlled three way valve
* pump


Parts derived by class inheritance of the standard parts:
* heated pipe
* branched pipe, pipe with valve, pipe with pump


The following controllers are defined in [parts/controllers](parts/controllers.py):
* PID controller
* bang bang controller
* two sensor controller
* model predictive controller (CHP-plant specific)
Preferred tuning method for PID controllers is Ziegler-Nichols, since the parameters `Kp_crit` and `T_crit` can be passed directly to the controller while specifying the aggressiveness of the PID controller with rules like `classic` or `pessen-int` (Pessen integral rule).


Compound parts consisting of multiple other parts and controllers can be found in [parts/part_modules](parts/part_modules).
Part dimensions, such as pipe diameters, and controller coefficients have been to fit a wide range of flow speeds and temperatures, but may be adjusted if controls show instabilities or if the solver requires too many retries to find a stable solution.
* gas boiler
* chp plant, also with flue gas heat exchanger (based on fitting a model to manufacturer specific measurement data)
* three different consumer appliances (space heating, state-of-the-art water heating, low exergy water heating)


New parts and controllers can be added either by defining completely new classes or by inheriting from existing parts.


Short documentation
===================

coming soon

Validation
==========

MultiSim has been fully validated following standard ANSI/BPI-2400-S-2015. A stricter set of statistic measures than provided in the standard has been used.

Since MultiSim was part of a PhD thesis, validation results will be added as soon as the thesis has been published (approximately December 2021).

Examples
========

coming soon

Known limitations and To-do
===========================

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

For building, distribution and installation instructions, see INSTALL.rst_.

.. _INSTALL.rst:   https://github.com/JoElfner/multisim/blob/master/INSTALL.rst
