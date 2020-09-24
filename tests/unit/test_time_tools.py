import pytest
import pytz

from omc3.utils import time_tools as tt

@pytest.mark.basic
def test_tz_check_succeed(now):
    tt.check_tz(now, pytz.utc)

@pytest.mark.basic
def test_tx_check_fail(now):
    with pytest.raises(AssertionError):
        tt.check_tz(now, tt.get_cern_timezone())

@pytest.mark.basic
def test_tz_conversions(now):
    local = tt.utc_to_local(now, tt.get_cern_timezone())
    tt.check_tz(local, tt.get_cern_timezone())
    assert local.timestamp() == now.timestamp()
    assert local.time() != now.time()

    utc = tt.local_to_utc(local, tt.get_cern_timezone())
    assert utc.time() == now.time()

@pytest.mark.basic
def test_strings(now):
    utc_str = now.strftime(tt.get_cern_time_format())
    utc = tt.cern_utc_string_to_utc(utc_str)
    assert now.time() == utc.time()

    local_str = tt.utc_to_local(now, tt.get_cern_timezone()).strftime(tt.get_readable_time_format())
    utc = tt.local_string_to_utc(local_str, tt.get_cern_timezone())
    assert now.time() == utc.time()

@pytest.mark.basic
def test_accelerator_datetime(now):
    lhc = tt.AcceleratorDatetime['lhc'](now)
    ps = tt.AcceleratorDatetime['ps'](now)
    sps = tt.AcceleratorDatetime['sps'](now)
    assert lhc.local.time() == ps.local.time()
    assert lhc.local.time() == sps.local.time()
    assert lhc.local.time != lhc.utc.time()


# Fixtures #####################################################################


@pytest.fixture()
def now():
    return tt.utc_now()