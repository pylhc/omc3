"""
Time Tools
-------------------------

Provides tools to handle times more easily, in particular to switch easily between local time
and UTC time.
"""
from datetime import datetime, timedelta
import dateutil.tz as tz

from omc3.definitions.formats import TIME


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

    dt_utc = dt_local.astimezone(tz.tzutc())
    return dt_utc


def utc_to_local(dt_utc, timezone):
    """Convert UTC datetime object to local datetime object."""
    check_tz(dt_utc, tz.tzutc())

    dt_local = dt_utc.astimezone(timezone)
    return dt_local


def local_string_to_utc(local_string, timezone):
    """Converts a time string in local time to UTC time."""
    dt = datetime.strptime(local_string, get_readable_time_format())
    dt = dt.replace(tzinfo=timezone)
    return local_to_utc(dt, timezone)


def utc_string_to_utc(utc_string):
    """Convert a time string in utc to a UTC datetime object."""
    dt = datetime.strptime(utc_string, get_readable_time_format())
    dt = dt.replace(tzinfo=tz.tzutc())
    return dt


def cern_utc_string_to_utc(utc_string):
    """Convert a time string in cern-utc to a utc datetime object."""
    dt = datetime.strptime(utc_string, get_cern_time_format())
    dt = dt.replace(tzinfo=tz.tzutc())
    return dt


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

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz.tzutc())

        return datetime.__new__(cls, dt.year, dt.month, dt.day,
                                dt.hour, dt.minute, dt.second, dt.microsecond,
                                tzinfo=dt.tzinfo)

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
        return cls(datetime.utcfromtimestamp(ts))

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
