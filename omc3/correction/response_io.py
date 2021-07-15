"""
Response Matrix IO
------------------

Input and output functions for response matrices.
"""
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence, Set

import pandas as pd
import logging


LOG = logging.getLogger(__name__)

COMPLIB = 'blosc'  # zlib is the standard compression
COMPLEVEL = 9  # goes from 0-9, 9 is highest compression, None deactivates if COMPLIB is None


# Fullresponse -----------------------------------------------------------------


def read_fullresponse(path: Path, optics_parameters: Sequence[str] = None) -> Dict[str, pd.DataFrame]:
    """Load the response matrices from disk.
    Beware: As empty DataFrames are skipped on write,
    default for not found entries are empty DataFrames.
    """
    if not path.exists():
        raise IOError(f"Fullresponse file {str(path)} does not exist.")

    LOG.info(f"Loading response matrices from file '{str(path)}'")
    with pd.HDFStore(path, mode='r') as store:
        _check_keys(store, optics_parameters, 'fullresponse')

        fullresponse = defaultdict(pd.DataFrame)
        if optics_parameters is None:
            optics_parameters = _main_store_groups(store)
        for p in optics_parameters:
            fullresponse[p] = store[p]
    return fullresponse


def write_fullresponse(path: Path, fullresponse: Dict[str, pd.DataFrame]):
    """Write the full response matrices to disk.
    Beware: Empty Dataframes are skipped! (HDF creates gigantic files otherwise)
    """
    LOG.info(f"Saving response matrices into file '{str(path)}'")
    if path.exists():
        LOG.warning(f"Fullresponse file {str(path)} already exist and will be overwritten.")
    with pd.HDFStore(path, mode='w', complib=COMPLIB, complevel=COMPLEVEL) as store:
        for param, response_df in fullresponse.items():
            store.put(value=response_df, key=param, format="table")


# Varmap -----------------------------------------------------------------------


def read_varmap(path: Path, k_values: Sequence[str] = None) -> Dict[str, Dict[str, pd.Series]]:
    """Load the variable mapping file from disk.
    Beware: As empty DataFrames are skipped on write,
    default for not found entries are empty Series.
    """
    if not path.exists():
        raise IOError(f"Varmap file {str(path)} does not exist.")

    LOG.info(f"Loading varmap from file '{str(path)}'")
    with pd.HDFStore(path, mode='r') as store:
        _check_keys(store, k_values, 'varmap')

        varmap = defaultdict(lambda: defaultdict(pd.Series))
        for key in store.keys():
            _, param, subparam = key.split('/')
            if k_values is not None and param not in k_values:
                continue
            varmap[param][subparam] = store[key]
    return varmap


def write_varmap(path: Path, varmap: Dict[str, Dict[str, pd.Series]]):
    """Write the  variable mapping file to disk.
    Beware: Empty Dataframes are skipped! (HDF creates gigantic files otherwise)
    """
    LOG.info(f"Saving varmap into file '{str(path)}'")
    with pd.HDFStore(path, mode='w', complib=COMPLIB, complevel=COMPLEVEL) as store:
        for param, sub in varmap.items():
            for subparam, varmap_series in sub.items():
                store.put(value=varmap_series, key=f"{param}/{subparam}", format="table")


# Helper -----------------------------------------------------------------------


def _check_keys(store: pd.HDFStore, keys: Sequence[str], id:str):
    if keys is None:
        return

    groups = _main_store_groups(store)
    not_found = [k for k in keys if k not in groups]
    if len(not_found):
        raise ValueError(f"The following keys could not be found in {id} file:"
                         f" {', '.join(not_found)}")


def _main_store_groups(store: pd.HDFStore) -> Set[str]:
    """Returns sequence of unique main store groups."""
    return {k.split('/')[1] for k in store.keys()}
