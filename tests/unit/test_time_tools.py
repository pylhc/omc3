import dateutil.tz as tz
import pytest

from datetime import datetime

from omc3.utils import time_tools as tt


@pytest.mark.basic
def test_tz_check_succeed(now):
    tt.check_tz(now, tz.tzutc())


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
    assert lhc.local.time() != lhc.utc.time()

    assert lhc.utc.tzinfo == datetime.now(tz.tzutc()).tzinfo
    assert lhc.local.tzinfo == tt.get_cern_timezone()


@pytest.mark.basic
def test_fold():
    # due to daylight saving time change on the 25th of October 2020
    # 01:00 UTC and 00:00 UTC corresponded to the same local time,
    # which is indicated by the fold attribute (0 == earlier time, 1 == later time)
    # see https://peps.python.org/pep-0495/
    folded = tt.AcceleratorDatetime['lhc'](2020, 10, 25, 1, 0, 0)
    no_fold = tt.AcceleratorDatetime['lhc'](2020, 10, 25, 0, 0, 0)
    
    assert folded.local.hour == no_fold.local.hour
    assert folded.local.fold == 1
    assert no_fold.local.fold == 0

# Fixtures #####################################################################


@pytest.fixture()
def now():
    return tt.utc_now()
