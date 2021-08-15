.. -*- mode: rst -*-

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

MultiSim is a thermal simulation tool for heating appliances.

- **Developer/maintainer:** https://www.linkedin.com/in/johannes-elfner

Even though MultiSim is **operational**, many things have to be *refactored*,
*replaced*, *improved* etc... Especially at the core class `SimEnv` and the most
basic parts like pipes and TES, a steep learning curve of the Dev can be
witnessed. ;) Thus enhancements should start here.

Furthermore current tests used for TDD are based on **proprietary measurement
data**. Thus these tests **cannot be published**. Hence tests included in this
public GitHub repo are merely truncated stumps. Using free data to integrate
extensive tests will be an important step.

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
`matplotlib`. For some parts `sklearn` is a dependency.

For building, distribution and installation instructions, see INSTALL.rst_.

.. _INSTALL.rst:   https://github.com/JoElfner/multisim/blob/master/INSTALL.rst
