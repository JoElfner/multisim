Building and installing MultiSim
++++++++++++++++++++++++++++++++

.. Contents::

.. role:: bash(code)
   :language: bash

INTRODUCTION
============

It is recommended that you use a complete scientific Python distribution like
Anaconda_ with Python >=3.7.7 to utilize MuliSim.

.. _Anaconda: https://www.anaconda.com

This install documentation mostly gives short detail on how to maintain, update
and package MultiSim. Also step-by-step instructions on how to upload the build
to GitHub_ and PyPI_ are included.

.. _GitHub: https://github.com/
.. _PyPI: https://pypi.org/

The direct links to the GitHub repository and to the PyPI project are:

  - MultiSim on GitHub https://github.com/JoElfner/multisim
  - MultiSim on PyPI https://pypi.org/project/MultiSim/

PREREQUISITES
=============

MultiSim requires the following software installed on your platform:

1) Python__ >= 3.7

__ https://www.python.org

2) SciPy__ >= 1.3.2

__ https://www.scipy.org/

3) NumPy__ >= 1.18.1, preferably as part of the SciPy ecosystem

__ https://www.numpy.org/

4) Pandas__ >= 1.0.1, preferably as part of the SciPy ecosystem

__ https://pandas.pydata.org/

5) Numba__ >= 0.50.0

__ http://numba.pydata.org/

6) matplotlib__ >= 3.0, preferably as part of the SciPy ecosystem

__ https://matplotlib.org/

7) scikit-learn__ >= 0.23.0

__ https://scikit-learn.org/stable/index.html

To build and distribute, the following packages are also required:

8) setuptools__

__ https://github.com/pypa/setuptools

9) twine__ >= 2.0.0

__ https://pypi.org/project/twine/

10) If you want to build the documentation: Sphinx__ >= 1.2.1

__ http://www.sphinx-doc.org/


PACKAGING MULTISIM
==================

As long as no collaboration is planned for MultiSim, local packaging should be
chosen over solely working with the GitHub repository to facilitate testing,
checks and keeping things straightforward.

To package MultiSim into an installable package, use your preferred command
line tool, f.i. Anaconda prompt, to navigate to the package`s top-level folder
and build the package. For the current dev`s computer, this is f.i.:

.. code:: bash

  cd /d "path_to_the_package/multisim"
  python setup.py build

To build and directly install the newly built package, do:

.. code:: bash

  cd /d "path_to_the_package/multisim"
  python setup.py build && python setup.py install


Push to GitHub
==============

Forking and committing to this package is highly encouraged. There is so much
refactoring, code cleaning and improving to do...


Troubleshooting
---------------

Everything will be fine... ;)