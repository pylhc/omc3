"""
Mock
----

Provides mock functionality for packages necessitating the ``CERN GPN`` and only installable from the
``acc-py`` package index, such as ``pytimber``, ``pjlsa`` etc.

.. code-block:: python

    from omc3.utils.mock import cern_network_import
    pytimber = cern_network_import("pytimber")
    db = pytimber.LoggingDB(source="nxcals")  # will raise if pytimber not installed
"""
import importlib


class CERNNetworkMockPackage:
    """
    Mock class to raise an error if the desired package functionality is called when the package is not
    actually installed. Designed for packages installable only from inside the CERN network,
    that are declared as ``cern`` extra. See module documentation.
    """

    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, item):
        raise ImportError(
            f"The '{self.name}' package does not seem to be installed but is needed for this function. "
            "Install it with the 'cern' extra dependency, which requires to be on the CERN network and to "
            "install from the acc-py package index. Refer to the documentation for more information."
        )


def cern_network_import(package: str):
    """
    Convenience function to try and import packages only available (and installable) on the CERN network.
    If installed, the module is returned, otherwise a mock class is returned, which will raise an
    insightful ``ImportError`` on attempted use.

    Args:
        package (str): name of the package to try and import.
    """
    try:
        return importlib.import_module(package)
    except ImportError:
        return CERNNetworkMockPackage(package)
