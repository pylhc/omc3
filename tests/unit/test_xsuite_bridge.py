"""Unit tests for the shared xsuite bridge (:mod:`omc3.model.xsuite_bridge`)."""
import inspect

import pytest

from omc3.correction import response_madng, response_xsuite
from omc3.model import xsuite_bridge


@pytest.mark.basic
def test_create_xsuite_json_reexported_from_lhc_creator():
    """The LHC creator must keep exposing create_xsuite_json / XSUITE_JSON (now from the bridge)."""
    from omc3.model.model_creators import lhc_xsuite_model_creator as lhc_xsuite

    assert lhc_xsuite.create_xsuite_json is xsuite_bridge.create_xsuite_json
    assert lhc_xsuite.XSUITE_JSON == xsuite_bridge.XSUITE_JSON


@pytest.mark.basic
@pytest.mark.parametrize("module", [response_xsuite, response_madng])
def test_xsuite_path_does_not_call_madx_wrapper_directly(module):
    """The MAD-X call must live only in the bridge; the xsuite / MAD-NG modules go through it."""
    source = inspect.getsource(module)
    assert "madx_wrapper" not in source, (
        f"{module.__name__} references madx_wrapper directly; it should use xsuite_bridge instead."
    )


@pytest.mark.basic
def test_bridge_is_madx_owner():
    """The bridge is the single module in the xsuite lattice path that imports madx_wrapper."""
    assert "madx_wrapper" in inspect.getsource(xsuite_bridge)


@pytest.mark.basic
def test_parse_correction_file(tmp_path):
    """Correction files (``knob = knob +value;``) parse to (knob, value) increments."""
    corr = tmp_path / "changeparameters.madx"
    corr.write_text(
        "! Values to match model to measurement. \n"
        "kqt.a12 = kqt.a12 +1.200000e-05;\n"
        "kqt.b34 = kqt.b34 -2.500000e-04;\n"
        "\n"
    )
    parsed = xsuite_bridge._parse_correction_file(corr)
    assert parsed == [("kqt.a12", 1.2e-05), ("kqt.b34", -2.5e-04)]


@pytest.mark.basic
def test_apply_correction_files(tmp_path):
    """apply_correction_files increments the matching env vars by the parsed deltas."""
    corr = tmp_path / "changeparameters.madx"
    corr.write_text("kqt.a12 = kqt.a12 +1.0e-05;\nkqt.a12 = kqt.a12 +2.0e-05;\n")

    env = {"kqt.a12": 3.0e-05}
    xsuite_bridge.apply_correction_files(env, [corr])
    assert env["kqt.a12"] == pytest.approx(6.0e-05)
