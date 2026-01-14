from collections.abc import Callable
from datetime import datetime

from dateutil import tz

from omc3.utils import logging_tools
from omc3.utils.mock import cern_network_import

jpype: object = cern_network_import("jpype")

LOGGER = logging_tools.get_logger(__name__)


def knob_to_output_name(knob_name: str) -> str:
    """
    Convert a knob name to an output-friendly name for file names.

    Args:
        knob_name (str): The original knob name.

    Returns:
        str: The modified knob name suitable for output.
    """
    return knob_name.replace(":", "_").replace("/", "_").replace("-", "_")


def strip_i_meas(text: str) -> str:
    """
    Remove the I_MEAS suffix from a variable name.

    Args:
        text (str): The variable name possibly ending with ':I_MEAS'.

    Returns:
        str: The variable name without the ':I_MEAS' suffix.
    """
    return text.removesuffix(":I_MEAS")


def try_to_acquire_data(function: Callable, *args, **kwargs):
    """Tries to get data from function multiple times.
    TODO: Move to omc3 as is also used there in BBQ extraction.

     Args:
         function (Callable): function to be called, e.g. db.get
         args, kwargs: arguments passed to ``function``

    Returns:
        Return arguments of ``function``

    """
    retries = kwargs.pop("retries", 3)
    for tries in range(retries + 1):
        try:
            return function(*args, **kwargs)
        except jpype.java.lang.IllegalStateException as e:
            raise OSError("Could not acquire data, user probably has no access to NXCALS") from e
        except jpype.JException as e:  # Might be a case for retries
            if "RetryableException" in str(e) and (tries + 1) < retries:
                LOGGER.warning(f"Could not acquire data! Trial no {tries + 1} / {retries}")
                continue  # will go to the next iteratoin of the loop, so retry
            raise OSError("Could not acquire data!") from e
    raise RuntimeError(f"Could not acquire data after {retries:d} retries.")


def timestamp_to_utciso(timestamp: float) -> str:
    """Convert a timestamp to an ISO format string."""
    return datetime.fromtimestamp(timestamp, tz=tz.UTC).isoformat()
