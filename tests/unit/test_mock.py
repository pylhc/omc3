import pytest
import tfs

from omc3.utils.mock import TechicalNetworkMockPackage, cern_network_import


class TestCERNNetworkImport:
    def test_absent_package_gives_mock_class(self):
        fake_package = cern_network_import("fake_package")
        assert isinstance(fake_package, TechicalNetworkMockPackage)
        assert fake_package.name == "fake_package"

        with pytest.raises(ImportError):
            fake_package.some_function()

    def test_present_package_works(self):
        tfspandas = cern_network_import("tfs")
        df = tfspandas.TfsDataFrame()
        assert isinstance(df, tfs.TfsDataFrame)
