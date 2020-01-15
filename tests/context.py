"""
Not committing this as code for now.
Not having imports at top of module is against PEP8.
For a workaround I advise following the Zen of Python and going with this conventional syntax.

import sys
from os.path import abspath, join, dirname, pardir
try:
    import omc3
except ImportError:
    root_path = abspath(join(dirname(__file__), pardir, pardir))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    omc3_path = join(root_path, "omc3")
    if omc3_path not in sys.path:
        sys.path.insert(0, omc3_path)
    import omc3
"""

import sys
from os.path import abspath, dirname, join, pardir

import omc3  # nopep8

root_path = abspath(join(dirname(__file__), pardir))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
omc3_path = join(root_path, "omc3")
if omc3_path not in sys.path:
    sys.path.insert(0, omc3_path)
