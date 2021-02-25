"""
Timber Extraction
-----------------

Tools to extract data from ``Timber``. It is a bit heavy on the LHC side at the moment.

**Please note**: this module requires the ``pytimber`` package to access ``Timber`` functionality,
both of which are only possible from inside the CERN network.

To install ``pytimber`` along ``omc3``, please do so from inside the CERN network by using the [tech] extra
dependency and installing from the ``acc-py`` package index (by specifying ``--index-url
https://acc-py-repo.cern.ch/repository/vr-py-releases/simple`` and
``--trusted-host acc-py-repo.cern.ch`` to your ``pip`` installation command).
"""
import re

import numpy as np
import pytimber
import tfs

from omc3.tune_analysis import constants as const
from omc3.utils import logging_tools
from omc3.utils.time_tools import CERNDatetime

TIME_COL = const.get_time_col()
START_TIME = const.get_tstart_head()
END_TIME = const.get_tend_head()


LOG = logging_tools.get_logger(__name__)


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
        t_start: starting time in local time or timestamp.
        t_end: end time in local time or timestamp.
        keys: list of data to extract.
        names: dict to map keys to column names.

    Returns: tfs pandas dataframe.
    """
    db = pytimber.LoggingDB(source="nxcals")
    if keys is None:
        keys = get_tune_and_coupling_variables(db)

    extract_dict = db.get(keys, t_start, t_end)

    out_df = tfs.TfsDataFrame()
    for key in keys:
        if extract_dict[key.upper()][1][0].size > 1:
            raise NotImplementedError("Multidimensional variables are not implemented yet.")

        data = np.asarray(extract_dict[key.upper()]).transpose()
        col = names.get(key, key)

        key_df = tfs.TfsDataFrame(data, columns=[TIME_COL, col]).set_index(TIME_COL)

        out_df = out_df.merge(key_df, how="outer", left_index=True, right_index=True)

    out_df.index = [CERNDatetime.from_timestamp(i) for i in out_df.index]
    out_df.headers[START_TIME] = CERNDatetime.from_timestamp(t_start).cern_utc_string()
    out_df.headers[END_TIME] = CERNDatetime.from_timestamp(t_end).cern_utc_string()
    return out_df


def get_tune_and_coupling_variables(db: pytimber.LoggingDB) -> list:
    """
    Returns the tune and coupling variable names.

    Args:
        db: pytimber database.

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


def get_fill_times(db: pytimber.LoggingDB, fill_number: int) -> tuple:
    """
    Returns start and end time of fill with fill number.

    Args:
        db: pytimber database.
        fill_number: fill number.

    Returns:
       `Tuple` of start and end time.
    """
    fill = db.getLHCFillData(fill_number)
    return fill['startTime'], fill['endTime']
