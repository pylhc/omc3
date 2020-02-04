"""
Time Tools
-------------------------

Provides tools to handle times more easily,
in particular to switch easily between local time and utc.

:module: utils.time_tools
:author: jdilly

"""
from datetime import datetime, timedelta

import pytz

from omc3.definitions.formats import TIME


# Datetime Conversions #########################################################


def utc_now():
    """ Get utc now as time """
    return datetime.now(pytz.utc)


def get_cern_timezone():
    """ Get time zone for cern measurement data. """
    return pytz.timezone("Europe/Zurich")


def get_cern_time_format():
    """ Default time format """
    return TIME  # TODO maybe only here?


def get_readable_time_format():
    """ Human readable time format """
    return "%Y-%m-%d %H:%M:%S.%f"


def local_to_utc(dt_local, timezone):
    """ Convert local datetime object to utc datetime object. """
    try:
        return timezone.localize(dt_local).astimezone(pytz.utc)
    except ValueError:
        check_tz(dt_local, timezone)
        return dt_local.astimezone(pytz.utc)


def utc_to_local(dt_utc, timezone):
    """ Convert utc datetime object to local datetime object. """
    try:
        return pytz.utc.localize(dt_utc).astimezone(timezone)
    except ValueError:
        check_tz(dt_utc, pytz.utc)
        return dt_utc.astimezone(timezone)


def local_string_to_utc(local_string, timezone):
    """ Converts a time string in local time to UTC time. """
    return local_to_utc(datetime.strptime(local_string, get_readable_time_format()), timezone)


def utc_string_to_utc(utc_string):
    """ Convert a time string in utc to a utc datetime object """
    return pytz.utc.localize(datetime.strptime(utc_string, get_readable_time_format()))


def cern_utc_string_to_utc(utc_string):
    """ Convert a time string in cern-utc to a utc datetime object """
    return pytz.utc.localize(datetime.strptime(utc_string, get_cern_time_format()))


def check_tz(localized_dt, timezone):
    """ Checks if timezone is correct. """
    if localized_dt.tzinfo is None or localized_dt.tzinfo.utcoffset(localized_dt) is None:
        raise AssertionError("Datetime object needs to be localized!")

    if not str(localized_dt.tzinfo) == str(timezone):
        raise AssertionError(
            f"Datetime Timezone should be '{timezone}' "
            f"but was '{localized_dt.tzinfo}'"
        )


# AccDatetime Classes ##########################################################


class AccDatetime(datetime):
    """ Wrapper for a datetime object to easily convert between local and utc time
    as well as give different presentations."""
    _LOCAL_TIMEZONE = None
    nanosecond = 0  # used in by pandas to make indices

    def __new__(cls, *args, **kwargs):
        if cls._LOCAL_TIMEZONE is None:
            raise NotImplementedError("Do not use the AccDatetime class, "
                                      "but one of its children.")
        if len(args) == 1 and isinstance(args[0], datetime):
            dt = args[0]
        else:
            dt = datetime.__new__(*args, **kwargs)
        dt = utc_to_local(dt, pytz.utc)  # does not convert, but checks tz
        return datetime.__new__(cls,
                                dt.year, dt.month, dt.day,
                                dt.hour, dt.minute, dt.second, dt.microsecond,
                                tzinfo=pytz.utc)

    def get_datetime(self):
        """ Return normal datetime object (in case ducktyping does not work. Looking at you, mpl!). """
        return datetime(self.year, self.month, self.day,
                        self.hour, self.minute, self.second, self.microsecond,
                        tzinfo=pytz.utc)

    def get_local_timezone(self):
        """ Get local timezone. """
        return self._LOCAL_TIMEZONE

    def utc(self):
        """ Get utc datetime object """
        return self

    def local(self):
        """ Get local datetime object """
        return self.astimezone(self._LOCAL_TIMEZONE)

    def local_string(self):
        """ Get local time as string """
        return self.local().strftime(get_readable_time_format())

    def utc_string(self):
        """ Get utc time as string """
        return self.strftime(get_readable_time_format())

    def cern_utc_string(self):
        """ Get utc time as string (cern format)"""
        return self.strftime(get_cern_time_format())

    def add(self, **kwargs):
        """ Add timedelta and return as new object """
        return self.__class__(self + timedelta(**kwargs))

    def sub(self, **kwargs):
        """ Subtract timedelta and return as new object """
        return self.__class__(self - timedelta(**kwargs))

    def __add__(self, td):
        """ Add timedelta and return as new object """
        return self.__class__(super().__add__(td))

    def __sub__(self, other):
        """ Subtract timedelta or datetime/AccDatetime and return as new object """
        result = super().__sub__(other)
        if isinstance(result, datetime):
            return self.__class__(result)
        if isinstance(result, timedelta):
                return result
        return NotImplemented

    @classmethod
    def from_local_string(cls, s):
        """ Create AccDatetime object from datetime in local time string. """
        return cls(local_string_to_utc(s, cls._LOCAL_TIMEZONE))

    @classmethod
    def from_utc_string(cls, s):
        """ Create AccDatetime object from datetime in utc string. """
        return cls(utc_string_to_utc(s))

    @classmethod
    def from_cern_utc_string(cls, s):
        """ Create AccDatetime object from datetime in utc string. """
        return cls(cern_utc_string_to_utc(s))

    @classmethod
    def from_local(cls, dt):
        """ Create AccDatetime object from datetime in local time. """
        return cls(local_to_utc(dt, cls._LOCAL_TIMEZONE))

    @classmethod
    def from_utc(cls, dt):
        """ Create AccDatetime object from datetime in utc. """
        return cls(dt)

    @classmethod
    def from_timestamp(cls, ts):
        """ Create AccDatetime object from timestamp. """
        return cls(datetime.utcfromtimestamp(ts))

    @classmethod
    def now(cls):
        """ Create AccDatetime object at now. """
        return cls(utc_now())


class CERNDatetime(AccDatetime):
    """ AccDatetime class for all accelerators at CERN """
    _LOCAL_TIMEZONE = get_cern_timezone()


AcceleratorDatetime = {
    'lhc': CERNDatetime,
    'ps': CERNDatetime,
    'sps': CERNDatetime,
}
"""dict: Accelerator name to AccDatetime mapping. """