"""
Timber Extraction
-----------------

Tools to extract data from ``Timber``. It is a bit heavy on the LHC side at the moment.

**Please note**: this module requires the ``pytimber`` package to access ``Timber`` functionality,
both of which are only possible from inside the CERN network.

To install ``pytimber`` along ``omc3``, please do so from inside the CERN network by using the [cern] extra
dependency and installing from the ``acc-py`` package index (by specifying ``--index-url
https://acc-py-repo.cern.ch/repository/vr-py-releases/simple`` and
``--trusted-host acc-py-repo.cern.ch`` to your ``pip`` installation command).
"""
import datetime
import re
from contextlib import suppress
from typing import Dict, List, NewType, Sequence, Tuple, Union

import numpy as np
import tfs

# from jpype import JException, java
from omc3.tune_analysis import constants as const
from omc3.utils import logging_tools
from omc3.utils.mock import cern_network_import
from omc3.utils.time_tools import CERNDatetime

TIME_COL = const.get_time_col()
START_TIME = const.get_tstart_head()
END_TIME = const.get_tend_head()

LOG = logging_tools.get_logger(__name__)
pytimber = cern_network_import("pytimber")
jpype = cern_network_import("jpype")

MAX_RETRIES = 10  # number of retries on retryable exception
AcceptableTimeStamp = NewType("AcceptableTimeStamp", Union[CERNDatetime, int, float])


def lhc_fill_to_tfs(
    fill_number: int, keys: Sequence[str] = None, names: Dict[str, str] = None
) -> tfs.TfsDataFrame:
    """
    Extracts data for keys of fill from ``Timber``.

    Args:
        fill_number (int): Number of the fill to extract from.
        keys (Sequence[str]): the different variables names to extract data for.
        names (Dict[str, str): dict mapping keys to column names.

    Returns:
        The extracted data as a ``TfsDataFrame``.
    """
    db = pytimber.LoggingDB(source="nxcals")
    t_start, t_end = get_fill_times(db, fill_number)
    return extract_between_times(t_start, t_end, keys, names)


def extract_between_times(
    t_start: AcceptableTimeStamp,
    t_end: AcceptableTimeStamp,
    keys: Sequence[str] = None,
    names: Dict[str, str] = None,
) -> tfs.TfsDataFrame:
    """
    Extracts data for keys between ``t_start`` and ``t_end`` from ``Timber``.

    Args:
        t_start (AcceptableTimeStamp): starting time in CERNDateTime or timestamp.
        t_end (AcceptableTimeStamp): end time in local CERNDateTime or timestamp.
        keys (Sequence[str]): the different variables names to extract data for.
        names (Dict[str, str): dict mapping keys to column names.

    Returns:
        Extracted data in a ``TfsDataFrame``.
    """
    with suppress(TypeError):
        t_start: CERNDatetime = CERNDatetime.from_timestamp(t_start)

    with suppress(TypeError):
        t_end: CERNDatetime = CERNDatetime.from_timestamp(t_end)

    db = pytimber.LoggingDB(source="nxcals")
    if keys is None:
        keys = get_tune_and_coupling_variables(db)

    # Attempt getting data from NXCALS, which can sometimes need a few retries (yay NXCALS)
    # If Java gives a feign.RetryableException, retry up to MAX_RETRIES times.
    extract_dict = {}
    for tries in range(MAX_RETRIES + 1):
        try:
            # We use timestamps to avoid any confusion with local time
            extract_dict = db.get(keys, t_start.timestamp(), t_end.timestamp())
        except jpype.java.lang.IllegalStateException as java_state_error:
            raise IOError(
                "Could not get data from Timber, user probably has no access to NXCALS"
            ) from java_state_error
        except jpype.JException as java_exception:  # Might be a case for retries
            if "RetryableException" in str(java_exception) and (tries + 1) < MAX_RETRIES:
                LOG.warning(f"Could not get data from Timber! Trial no {tries + 1} / {MAX_RETRIES}")
                continue  # will go to the next iteratoin of the loop, so retry
            raise IOError("Could not get data from timber!") from java_exception
        else:
            break

    if (not len(extract_dict)  # dict is empty
            or all(not len(v) for v in extract_dict.values())  # values are empty
            or all(len(v) == 2 and not len(v[0]) for v in extract_dict.values())  # arrays are empty (size 2 for time/data)
    ):
        raise IOError(f"Variables {keys} found but no data extracted in time {t_start.utc_string} - {t_end.utc_string} (UTC).\n"
                      f"Possible reasons:\n"
                      f"  - Too small time window.\n"
                      f"  - Old pytimber version.\n"
                      f"  - Variable outdated (i.e. no longer logged).")

    out_df = tfs.TfsDataFrame()
    for key in keys:
        if extract_dict[key][1][0].size > 1:
            raise NotImplementedError("Multidimensional variables are not implemented yet")

        data = np.asarray(extract_dict[key]).transpose()
        column = key if names is None else names.get(key, key)
        key_df = tfs.TfsDataFrame(data, columns=[TIME_COL, column]).set_index(TIME_COL)
        out_df = out_df.merge(key_df, how="outer", left_index=True, right_index=True)

    out_df.index = [CERNDatetime.from_timestamp(i) for i in out_df.index]
    out_df.headers[START_TIME] = t_start.cern_utc_string()
    out_df.headers[END_TIME] = t_end.cern_utc_string()
    return out_df


def get_tune_and_coupling_variables(db) -> List[str]:
    """
    Returns the tune and coupling variable names.

    Args:
        db (pytimber.LoggingDB): pytimber database connexion.

    Returns:
        List of variable names as strings.
    """
    bbq_vars = []
    for search_term in ["%EIGEN%FREQ%", "%COUPL%ABS%"]:
        search_results = db.search(search_term)
        for res in search_results:
            if re.match(r"LHC\.B(OFSU|QBBQ\.CONTINUOUS)", res):
                bbq_vars.append(res)
    return bbq_vars


def get_fill_times(
    db, fill_number: int
) -> Tuple[Union[datetime.datetime, float], Union[datetime.datetime, float]]:
    """
    Returns start and end time of fill with fill number.

    Args:
        db (pytimber.LoggingDB): pytimber database.
        fill_number (int): fill number.

    Returns:
       `Tuple` of start and end time.
    """
    fill = db.getLHCFillData(fill_number)
    return fill["startTime"], fill["endTime"]
