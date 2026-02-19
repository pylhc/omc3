"""
Time Tools
-------------------------

Provides tools to handle times more easily, in particular to switch easily between local time
and UTC time.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

import dateutil.tz as tz
from dateutil.relativedelta import relativedelta

from omc3.definitions.formats import TIME

LOGGER = logging.getLogger(__name__)
MINUS_CHARS: tuple[str, ...] = ("_", "-")


# CLI Time Parsing #############################################################

def parse_time(time: str, timedelta: str | None = None) -> datetime:
    """Parse time from given time-input from command line."""
    t = _parse_time_from_str(time)
    if timedelta:
        t = _add_time_delta(t, timedelta)
    if t.tzinfo is None:
        raise ValueError("Datetime object must be timezone-aware")
    return t.astimezone(tz.UTC)  # ensure returning UTC time always


def _parse_time_from_str(time_str: str) -> datetime:
    """Parse a datetime object from given string.
    A value error is raised, if the string could not be parsed to a
    timezone-aware (!!) datetime object.

    Formats supported:
    - "now" (current time in UTC)
    - ISO-Format string, e.g. "2024-01-30T12:34:56+00:00"
    - Timestamp (float), e.g. "1700000000.83" (will return UTC time)

    Returns:
        datetime: Parsed datetime object, timezone-aware.
    """
    # Now? ---
    if time_str.lower() == "now":
        return datetime.now(tz=tz.UTC)

    # ISOFormat? ---
    try:
        dt = datetime.fromisoformat(time_str)
    except (TypeError, ValueError):
        LOGGER.debug("Could not parse time string as ISO format")
        pass
    else:
        if dt.tzinfo is None:
            raise ValueError(
                "Datetime object must be timezone-aware, e.g. add '+00:00' suffix for UTC"
            )
        return dt

    # Timestamp? ---
    try:
        return datetime.fromtimestamp(float(time_str), tz=tz.UTC)
    except (TypeError, ValueError):
        LOGGER.debug("Could not parse time string as a timestamp")
        pass

    raise ValueError(f"Couldn't read datetime '{time_str}'")


def _add_time_delta(time: datetime, delta_str: str) -> datetime:
    """Parse delta-string and add time-delta to time."""
    sign = -1 if delta_str[0] in MINUS_CHARS else 1
    all_deltas = re.findall(r"(\d+)(\w)", delta_str)  # tuples (value, timeunit-char)
    # mapping char to the time-unit as accepted by relativedelta,
    # following ISO-8601 for time durations
    char2unit = {
        "s": "seconds",
        "m": "minutes",
        "h": "hours",
        "d": "days",
        "w": "weeks",
        "M": "months",
        "Y": "years",
    }
    # add all deltas, which are tuples of (value, timeunit-char)
    time_parts = {char2unit[delta[1]]: sign * int(delta[0]) for delta in all_deltas}
    return time + relativedelta(**time_parts)


# Datetime Conversions #########################################################


def utc_now():
    """Get UTC now as time."""
    return datetime.now(tz.tzutc())


def get_cern_timezone():
    """Get time zone for cern measurement data."""
    return tz.gettz('Europe/Zurich')


def get_cern_time_format():
    """Default time format."""
    return TIME  # TODO maybe only here?


def get_readable_time_format():
    """Human readable time format."""
    return "%Y-%m-%d %H:%M:%S.%f"


def local_to_utc(dt_local, timezone):
    """Convert local datetime object to utc datetime object."""
    check_tz(dt_local, timezone)
    return dt_local.astimezone(tz.tzutc())


def utc_to_local(dt_utc, timezone):
    """Convert UTC datetime object to local datetime object."""
    check_tz(dt_utc, tz.tzutc())
    return dt_utc.astimezone(timezone)


def local_string_to_utc(local_string, timezone):
    """Converts a time string in local time to UTC time."""
    dt = datetime.strptime(local_string, get_readable_time_format())
    dt = dt.replace(tzinfo=timezone)
    return local_to_utc(dt, timezone)


def utc_string_to_utc(utc_string):
    """Convert a time string in utc to a UTC datetime object."""
    dt = datetime.strptime(utc_string, get_readable_time_format())
    return dt.replace(tzinfo=tz.tzutc())


def cern_utc_string_to_utc(utc_string):
    """Convert a time string in cern-utc to a utc datetime object."""
    dt = datetime.strptime(utc_string, get_cern_time_format())
    return dt.replace(tzinfo=tz.tzutc())


def check_tz(localized_dt, timezone):
    """Checks if timezone is correct."""
    if localized_dt.tzinfo is None or localized_dt.tzinfo.utcoffset(localized_dt) is None:
        raise AssertionError("Datetime object needs to be localized!")

    if not localized_dt.tzname() == datetime.now(timezone).tzname():
        raise AssertionError(
            f"Datetime Timezone should be '{timezone}' "
            f"but was '{localized_dt.tzinfo}'"
        )


# AccDatetime Classes ##########################################################


class AccDatetime(datetime):
    """
    Wrapper for a datetime object to easily convert between local and UTC time as well as give
    different presentations.
    """
    _LOCAL_TIMEZONE = None
    nanosecond = 0  # used in by pandas to make indices

    def __new__(cls, *args, **kwargs):
        if cls._LOCAL_TIMEZONE is None:
            raise NotImplementedError("Do not use the AccDatetime class, "
                                      "but one of its children.")
        if len(args) == 1 and isinstance(args[0], datetime):
            dt = args[0]
        else:
            dt = datetime.__new__(cls, *args, **kwargs)

        if dt.tzinfo is None and "tzinfo" not in kwargs and len(args) < 8:  # allows forcing tz to `None`
                dt = dt.replace(tzinfo=tz.tzutc())

        return datetime.__new__(cls, dt.year, dt.month, dt.day,
                                dt.hour, dt.minute, dt.second, dt.microsecond,
                                tzinfo=dt.tzinfo, fold=dt.fold)

    @property
    def datetime(self):
        """
        Return normal datetime object (in case ducktyping does not work. Looking at you, mpl!).
        """
        return datetime(self.year, self.month, self.day,
                        self.hour, self.minute, self.second, self.microsecond,
                        tzinfo=tz.tzutc())

    @property
    def local_timezone(self):
        """Get local timezone."""
        return self._LOCAL_TIMEZONE

    @property
    def utc(self):
        """Get UTC datetime object."""
        return self

    @property
    def local(self):
        """Get local datetime object."""
        return self.astimezone(self._LOCAL_TIMEZONE)

    @property
    def local_string(self):
        """Get local time as string."""
        return self.local.strftime(get_readable_time_format())

    @property
    def utc_string(self):
        """Get utc time as string."""
        return self.strftime(get_readable_time_format())

    def cern_utc_string(self):
        """Get utc time as string (CERN format)."""
        return self.strftime(get_cern_time_format())

    def add(self, **kwargs):
        """Add timedelta and return as new object."""
        return self.__class__(self + timedelta(**kwargs))

    def sub(self, **kwargs):
        """Subtract timedelta and return as new object."""
        return self.__class__(self - timedelta(**kwargs))

    def __add__(self, td):
        """Add timedelta and return as new object."""
        return self.__class__(super().__add__(td))

    def __sub__(self, other):
        """Subtract timedelta or datetime/AccDatetime and return as new object."""
        result = super().__sub__(other)
        if isinstance(result, datetime):
            return self.__class__(result)
        if isinstance(result, timedelta):
                return result
        return NotImplemented

    @classmethod
    def from_local_string(cls, s):
        """Create `AccDatetime` object from datetime in local time string."""
        return cls(local_string_to_utc(s, cls._LOCAL_TIMEZONE))

    @classmethod
    def from_utc_string(cls, s):
        """Create `AccDatetime` object from datetime in utc string."""
        return cls(utc_string_to_utc(s))

    @classmethod
    def from_cern_utc_string(cls, s):
        """Create `AccDatetime` object from datetime in utc string."""
        return cls(cern_utc_string_to_utc(s))

    @classmethod
    def from_local(cls, dt):
        """Create AccDatetime object from datetime in local time."""
        return cls(local_to_utc(dt, cls._LOCAL_TIMEZONE))

    @classmethod
    def from_utc(cls, dt):
        """Create `AccDatetime` object from datetime in utc."""
        return cls(dt)

    @classmethod
    def from_timestamp(cls, ts):
        """Create `AccDatetime` object from timestamp."""
        return cls(datetime.fromtimestamp(ts, tz=tz.UTC))

    @classmethod
    def now(cls):
        """Create `AccDatetime` object at now."""
        return cls(utc_now())


class CERNDatetime(AccDatetime):
    """`AccDatetime` class for all accelerators at CERN."""
    _LOCAL_TIMEZONE = get_cern_timezone()


AcceleratorDatetime = {
    'lhc': CERNDatetime,
    'ps': CERNDatetime,
    'sps': CERNDatetime,
}
"""dict: Accelerator name to AccDatetime mapping. """
