import sys
from os.path import abspath, join, dirname, pardir
root_path = abspath(join(dirname(__file__), pardir, pardir))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
omc3_path = join(root_path, "omc3")
if omc3_path not in sys.path:
    sys.path.insert(0, omc3_path)

import omc3