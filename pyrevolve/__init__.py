from __future__ import absolute_import

from pyrevolve.pyrevolve import * # noqa
from pyrevolve.crevolve import * # noqa

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
