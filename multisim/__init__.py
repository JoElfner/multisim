#
from .version import version as __version__

from .simenv import SimEnv as Models  # keep backwards compatiblity
from .simenv import SimEnv  # new API
from . import all_parts as ap
from ._precompiled import dimless_no, matprops
from . import precomp_funs as pf  # TODO: deprecate as soon as refactored
from . import simenv as se  # TODO: deprecated?
from . import std_parts as _sp
from . import utility_functions as ut  # TODO: deprecate as soon as refactored
from ._utility import Meters
from ._utility import plotting
# from ._utility import *

__all__ = [
    '__version__',
    'Models',
    'SimEnv',
    'ap',
    'dimless_no',
    'matprops',
    'Meters',
    'plotting',
    'pf',
    'se',
    '_sp',
    'ut',
]
