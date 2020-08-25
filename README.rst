MultiSim
========

MultiSim is a thermal simulation tool for heating appliances.

- **Developer/maintainer:** https://www.linkedin.com/in/johannes-elfner

Even though MultiSim is **operational**, many things have to be *refactored*,
*replaced*, *improved* etc... Especially at the core class `Models` and the most
basic parts like pipes and TES, a steep learning curve of the Dev can be
witnessed. ;) Thus enhancements should start here.

Other enhancements could be:

1. Extend the tests using `pytest`.

2. Implementing the implicit differential equation solver in numba to speed things up considerably. Implicit solving is currently slowing down the simulation progress.

3. Move the outer explicit solver loop to numba. This should also improve the performance by several percent points.

4. Check if local Nusselt number calculation is implemented in each relevant part.

5. Fully implement parts with compound structures.

6. Refactor type checks during part-adding as much as possible, using `@property` may help. Move away from using to many `kwargs`-based arguments. Include more specific type hints and default args handling.

7. Move to Python 3.8, using type hints and assignment expressions.

8. There is a lot of chaos in `utility_functions`. This needs some heavy refactoring and tidying.

MultiSim depends mainly on `numpy`, `numba`, `pandas`, `scipy`, and
`matplotlib`. For some parts `sklearn` is a dependency.

For building, distribution and installation instructions, see INSTALL.rst_.

.. _INSTALL.rst:   https://github.com/JoElfner/multisim/blob/master/INSTALL.rst
