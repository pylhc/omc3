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
import re
from contextlib import suppress

import numpy as np
import tfs
from jpype import java, JException

from omc3.tune_analysis import constants as const
from omc3.utils import logging_tools
from omc3.utils.time_tools import CERNDatetime
from omc3.utils.mock import cern_network_import

TIME_COL = const.get_time_col()
START_TIME = const.get_tstart_head()
END_TIME = const.get_tend_head()

LOG = logging_tools.get_logger(__name__)
pytimber = cern_network_import("pytimber")

RETRIES = 10  # number of retries on retryable exception


def lhc_fill_to_tfs(fill_number, keys=None, names=None) -> tfs.TfsDataFrame:
    """
    Extracts data for keys of fill from ``Timber``.

    Args:
        fill_number: fill number.
        keys: list of data to extract.
        names: dict to map keys to column names.

    Returns: tfs pandas dataframe.
    """
    db = pytimber.LoggingDB(source="nxcals")
    t_start, t_end = get_fill_times(db, fill_number)
    out_df = extract_between_times(t_start, t_end, keys, names)
    return out_df


def extract_between_times(t_start, t_end, keys=None, names=None) -> tfs.TfsDataFrame:
    """
    Extracts data for keys between t_start and t_end from timber.

    Args:
        t_start: starting time in CERNDateTime or timestamp.
        t_end: end time in local CERNDateTime or timestamp.
        keys: list of data to extract.
        names: dict to map keys to column names.

    Returns: tfs pandas dataframe.
    """
    with suppress(TypeError):
        t_start: CERNDatetime = CERNDatetime.from_timestamp(t_start)

    with suppress(TypeError):
        t_end: CERNDatetime = CERNDatetime.from_timestamp(t_end)

    db = pytimber.LoggingDB(source="nxcals")
    if keys is None:
        keys = get_tune_and_coupling_variables(db)

    for ii in range(RETRIES+1):
        # Try getting data from Timber.
        # If there is a feign.RetryableException, retry up to RETRIES times.
        try:
            extract_dict = db.get(keys, t_start.timestamp(), t_end.timestamp())  # use TimeStamps to avoid confusion w/ local time
        except java.lang.IllegalStateException as e:
            raise IOError("Could not get data from timber. Probable cause: User has no access to nxcals!") from e
        except JException as e:
            if "RetryableException" in str(e) and (ii+1) < RETRIES:
                LOG.error(f"Could not get data from timber! Trial no {ii+1}/{RETRIES}.")
                continue
            raise IOError("Could not get data from timber!") from e
        else:
            break

    out_df = tfs.TfsDataFrame()
    for key in keys:
        if extract_dict[key][1][0].size > 1:
            raise NotImplementedError("Multidimensional variables are not implemented yet.")

        data = np.asarray(extract_dict[key]).transpose()
        col = names.get(key, key)

        key_df = tfs.TfsDataFrame(data, columns=[TIME_COL, col]).set_index(TIME_COL)

        out_df = out_df.merge(key_df, how="outer", left_index=True, right_index=True)

    out_df.index = [CERNDatetime.from_timestamp(i) for i in out_df.index]
    out_df.headers[START_TIME] = t_start.cern_utc_string()
    out_df.headers[END_TIME] = t_end.cern_utc_string()
    return out_df


def get_tune_and_coupling_variables(db) -> list:
    """
    Returns the tune and coupling variable names.

    Args:
        db (pytimber.LoggingDB): pytimber database.

    Returns:
        `list` of variable names.
    """
    bbq_vars = []
    for search_term in ['%EIGEN%FREQ%', '%COUPL%ABS%']:
        search_results = db.search(search_term)
        for res in search_results:
            if re.match(r'LHC\.B(OFSU|QBBQ\.CONTINUOUS)', res):
                bbq_vars.append(res)
    return bbq_vars


def get_fill_times(db, fill_number: int) -> tuple:
    """
    Returns start and end time of fill with fill number.

    Args:
        db (pytimber.LoggingDB): pytimber database.
        fill_number (int): fill number.

    Returns:
       `Tuple` of start and end time.
    """
    fill = db.getLHCFillData(fill_number)
    return fill['startTime'], fill['endTime']
