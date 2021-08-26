#
from .version import version as __version__

from .simenv import SimEnv as Models  # keep backwards compat
from .simenv import SimEnv  # new API
from . import all_parts as ap
from ._precompiled import material_properties as matprops
from ._utility import Meters as Meters
from . import precomp_funs as pf  # TODO: deprecate as soon as refactored
from . import simenv as se  # TODO: deprecated?
from . import std_parts as _sp
from . import utility_functions as ut

__all__ = [
    '__version__',
    'Models',
    'SimEnv',
    'ap',
    'matprops',
    'Meters',
    'pf',
    'se',
    '_sp',
    'ut',
]
