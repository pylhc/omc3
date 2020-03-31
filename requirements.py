"""
Requirement Printer
"""
import argparse
from contextlib import suppress

from setup import ALL_DEPENDENCIES

KEYMAP = {'install': 'install_requires',
          'test': 'tests_require',
          'setup': 'setup_requires'}


def print_requirements():
    """
    Prints the requirements for this package.

    Use either 'install', 'test', 'setup' (if present in setup.py)
    or one of the keys in 'extras_require'.
    If no key is given ALL dependencies are printed.

    """
    keys = _parse_args()
    _check_keys(keys)

    if len(keys) == 0:
        keys = _get_all_dependency_keys()

    dependencies = []
    for k in keys:
        dependencies += _get_dependencies(k)
    _print(dependencies)


# Helper -----------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Print arguments for this package'
    )
    parser.add_argument('keys',
                        type=str, nargs='*', default=[],
                        metavar="KEY",
                        help="The requirements you want.")
    return parser.parse_args().keys


def _get_all_dependency_keys():
    """ Returns a list of all usable dependency keys for this package."""
    deps = [k for k, v in KEYMAP.items() if v in ALL_DEPENDENCIES]
    deps += ALL_DEPENDENCIES.get('extras_require', [])
    return deps


def _get_dependencies(key):
    """ Return the dependencies for that key. """
    with suppress(KeyError):
        return ALL_DEPENDENCIES[KEYMAP[key]]

    with suppress(KeyError):
        return ALL_DEPENDENCIES['extras_require'][key]

    raise KeyError(f"'{key}' does not have requirements in this package.")


def _check_keys(keys):
    """ Check if the keys given are valid. """
    unknown_reqs = [k for k in keys if k not in _get_all_dependency_keys()]
    if unknown_reqs:
        raise KeyError(f"The keys {unknown_reqs} do not have requirements "
                       "in this package.")


def _print(_list):
    """ Print a unique and alphabetically sorted list."""
    print("\n".join(sorted(set(_list))))


# __main__ ---------------------------------------------------------------------


if __name__ == '__main__':
    print_requirements()
