"""
Linfile Cleaning
----------------
Performs an automated cleaning of different columns in the lin-file
as a standalone script to allow for manual refinement after harpy is done.

All data is assumed to be gaussian distributed around a "true" value,
and outliers are cleaned by calculating average and standard deviation
of the given data.

The cleaning is done by removing all data-points that are outside of the
1-0.5/n estimated percentile of the data. Where n is the number of
(remaining) data-points in each loop, and the process is repeated until
n stays constant (or 2 or less data-points remain).

Datapoints with a standard deviation smaller than the given limit are not
cleaned. The limit is given in whatever units the data itself is in and
is an absolute value.

Cleaning is done per given file independently
i.e. removed BPMs from one file can be present in the next.
The columns are iterated on the same file, i.e. the cleaning is done
consecutively on the already cleaned data from the last column,
yet the moments of the distribution themselves are evaluated per column.

In the end, the lin-files are overwritten with the cleaned ones.
If the ``backup`` option is activated, a backup of the original file
is created, which can be restored via the ``restore`` option.
In case the restore-flag is given, only the filenames are required.
No cleaning is performed with this option.

**Arguments:**

*--Required--*

- **files** *(PathOrStr)*:

    List of paths to the lin-files, including suffix.


*--Optional--*

- **backup**:

    Create a backup of the files before overwriting. The backups are
    numbered, with the highest number being the latest backup.

    action: ``store_true``


- **columns** *(str)*:

    Columns to clean on.


- **limit** *(float)*:

    Do not clean data-points deviating less than this limit from the
    average.

    default: ``0.0``


- **restore**:

    Restore the latest backup of the files. If this parameter is given, no
    cleaning is performed.

    action: ``store_true``


TODO:
also use isolation forest, BUT this probably needs some modification there
as well, as it only cleans on TUNE, not on NATTUNE.
And it requires an accelerator instance.
"""
import shutil
from numbers import Number
from pathlib import Path
from typing import Sequence, Union

import pandas as pd
import tfs
from generic_parser.entrypoint_parser import (
    entrypoint, EntryPointParameters
)

from omc3.definitions.formats import BACKUP_FILENAME
from omc3.harpy.constants import COL_NAME
from omc3.harpy.handler import _compute_headers
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr, save_config
from omc3.utils.outliers import get_filter_mask

LOG = logging_tools.get_logger(__name__)


def get_params():
    return EntryPointParameters(
        files=dict(
            required=True,
            type=PathOrStr,
            nargs='+',
            help="List of paths to the lin-files, including suffix.",
        ),
        # restore backup
        restore=dict(
            action="store_true",
            help=("Restore the latest backup of the files. "
                  "If this parameter is given, no cleaning is performed."),
        ),
        # for actual cleaning
        columns=dict(
            nargs='+',
            type=str,
            help="Columns to clean on.",
        ),
        limit=dict(
            type=float,
            help="Do not clean data-points deviating less than this limit from the average.",
            default=0.0,
        ),
        backup=dict(
            action="store_true",
            help=("Create a backup of the files before overwriting. "
                  "The backups are numbered, with the highest number being "
                  "the latest backup.")
        ),
    )


@entrypoint(get_params(), strict=True)
def main(opt):
    """Main function, to parse commandline input and separate restoration
    from cleaning."""
    save_config(Path('.'), opt, __file__)
    if opt.restore:
        restore_files(opt.files)
        return

    if opt.columns is None:
        raise ValueError("The option 'columns' is required for cleaning.")
    clean_columns(opt.files, opt.columns, opt.limit, opt.backup)


# Restore ----------------------------------------------------------------------

def restore_files(files: Sequence[Union[Path, str]]):
    """Restore backupped files."""
    failed = []
    for file in files:
        file = Path(file)
        try:
            _restore_file(file)
        except IOError as e:
            failed.append(str(e))

    if len(failed):
        all_errors = '\n'.join(failed)
        if len(failed) == len(files):
            raise IOError(f"Restoration of ALL files has failed\n{all_errors}")
        raise IOError(f"Restoration of some files has failed, "
                      f"but the others were restored:\n{all_errors}")

    LOG.info("Restoration successfully completed.")


def _restore_file(file):
    counter = 1
    backup_file = _get_backup_filepath(file, counter)
    if not backup_file.exists():
        raise IOError(f"No backups found for file {file.name}")

    # get last existing backup file
    while backup_file.exists():
        counter += 1
        backup_file = _get_backup_filepath(file, counter)
    backup_file = _get_backup_filepath(file, counter-1)

    # restore found  file
    LOG.info(f"Restoring last backup file {backup_file.name} to {file.name}.")
    shutil.move(backup_file, file)  # overwrites file


# Clean ------------------------------------------------------------------------

def clean_columns(files: Sequence[Union[Path, str]], columns: Sequence[str],
                  limit: float = 0.0, backup: bool = True):
    """ Clean the columns in the given files."""
    for file in files:
        file = Path(file)
        LOG.info(f"Cleaning {file.name}.")
        df = tfs.read_tfs(file, index=COL_NAME)
        for column in columns:
            df = _filter_by_column(df, column, limit)
        df.headers.update(_compute_headers(df))

        if backup:
            _backup_file(file)

        tfs.write_tfs(file, df, save_index=COL_NAME)


def _filter_by_column(df: pd.DataFrame, column: str, limit: Number) -> pd.DataFrame:
    """Get the dataframe with all rows dropped filtered by the given column."""
    if column not in df.columns:
        LOG.info(f"{column} not in current file. Skipping cleaning.")
        return df

    good_bpms = get_filter_mask(data=df[column], limit=limit)
    n_good, n_total = sum(good_bpms), len(good_bpms)
    LOG.info(f"Cleaned {n_total-n_good:d} of {n_total:d} elements in {column} ({n_good:d} remaining).")
    return df.loc[good_bpms, :]


# Backup ---

def _backup_file(file):
    counter = 1
    backup_file = _get_backup_filepath(file, counter)
    while backup_file.exists():
        counter += 1
        backup_file = _get_backup_filepath(file, counter)

    LOG.info(f"Backing up original file {file.name} to {backup_file.name}.")
    shutil.copy(file, backup_file)


def _get_backup_filepath(file: Path, counter: int):
    return file.with_name(BACKUP_FILENAME.format(basefile=file.name, counter=counter))


# Script Mode ------------------------------------------------------------------

if __name__ == '__main__':
    main()
