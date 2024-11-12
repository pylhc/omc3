from generic_parser.entrypoint_parser import ArgumentError
from omc3.utils import parsertools
from generic_parser import EntryPointParameters, EntryPoint
import pytest

@pytest.mark.basic
def test_require_param():
        params = EntryPointParameters()
        params.add_parameter(
            name="input",
            type=str,
            help="the input",
        )
        params.add_parameter(
            name="n",
            type=int,
            help="Some number",
            default=1,
        )
        params.add_parameter(
            name="debug",
            action="store_true",
            help="Debug mode",
        )

        entry = EntryPoint(params)
        (options, _) = entry.parse(["--input", "test", "--n", "10"])
        parsertools.require_param("input", params, options)
        parsertools.require_param("n", params, options)

        with pytest.raises(AttributeError):
            (options, _) = entry.parse(["--n", "10"])
            parsertools.require_param("input", params, options)
