"""
Plot Spectrum - Utilities
-------------------------

Common functions and sorting functions for the spectrum plotter.
"""
import os
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path
from typing import Iterable, Sized, Union

import matplotlib
import numpy as np
import pandas as pd
import tfs
from generic_parser import DotDict
from matplotlib import transforms, axes, pyplot as plt
from matplotlib.patches import Rectangle

from omc3.definitions.constants import PLANES
from omc3.harpy.constants import FILE_AMPS_EXT, FILE_FREQS_EXT, FILE_LIN_EXT, COL_NAME
from omc3.plotting.utils.annotations import get_fontsize_as_float
from omc3.plotting.utils.lines import VERTICAL_LINES_ALPHA, plot_vertical_line
from omc3.utils import logging_tools

LOG = logging_tools.getLogger(__name__)

STEM_LINES_ALPHA = 0.5
PATCHES_ALPHA = 0.2
LABEL_Y_SPECTRUM = 'Amplitude in {plane:s} [a.u]'
LABEL_Y_WATERFALL = 'Plane {plane:s}'
LABEL_X = 'Frequency [tune units]'
NCOL_LEGEND = 5  # number of columns in the legend
WATERFALL_FILENAME = "waterfall_spectrum"
STEM_FILENAME = "stem_spectrum"
AMPS = FILE_AMPS_EXT.format(plane='')
FREQS = FILE_FREQS_EXT.format(plane='')
LIN = FILE_LIN_EXT.format(plane='')


# Collector Classes ------------------------------------------------------------


class FigureContainer(object):
    """ Container for attaching additional information to one figure. """
    def __init__(self, path: str) -> None:
        self.fig, self.axes = plt.subplots(nrows=len(PLANES), ncols=1)
        self.data = OrderedDict()  # make sure in plotting to use this order
        self.tunes = {p: [] for p in PLANES}
        self.nattunes = {p: [] for p in PLANES}
        self.path = path
        self.minmax = {p: (1, 0) for p in PLANES}

    def add_data(self, label: str, new_data: dict) -> None:
        self.data[label] = new_data
        for plane in PLANES:
            # Add tunes
            try:
                self.tunes[plane].append(new_data[plane][LIN].loc[f'TUNE{plane.upper()}'])
            except KeyError:
                LOG.warning(f'TUNE{plane.upper()} not found for {label}.')

            try:
                self.nattunes[plane].append(new_data[plane][LIN].loc[f'NATTUNE{plane.upper()}'])
            except KeyError:
                LOG.debug(f'NATTUNE{plane.upper()} not found for {label}.')

            # update min/max
            mmin, mmax = self.minmax[plane]
            self.minmax[plane] = (
                min(mmin, new_data[plane][AMPS].min(skipna=True)),
                max(mmax, new_data[plane][AMPS].max(skipna=True))
            )


class IdData:
    """ Container to keep track of the id-sorting output """
    def __init__(self, id_: str, label: str, path: str) -> None:
        self.id = id_        # id for the figure-container dictionary
        self.label = label  # plot labels
        self.path = path    # figure output path


class FigureCollector:
    """ Class to collect figure containers and manage data adding. """
    def __init__(self) -> None:
        self.fig_dict = {}   # dictionary of matplotlib figures, for output
        self.figs = {}       # dictionary of FigureContainers, used internally

    def add_data_for_id(self, id_data: IdData, data: dict) -> None:
        """ Add the data at the appropriate figure container. """
        try:
            figure_cont = self.figs[id_data.id]
        except KeyError:
            figure_cont = FigureContainer(id_data.path)
            self.figs[id_data.id] = figure_cont
            self.fig_dict[id_data.id] = figure_cont.fig
        figure_cont.add_data(id_data.label, data)


# (Tune-) Line Plotting --------------------------------------------------------


def plot_lines(fig_cont: FigureContainer, lines: DotDict) -> None:
    label_size = get_fontsize_as_float(matplotlib.rcParams['axes.labelsize']) * 0.7
    bottom_qlabel = 1.01

    for idx_plane, plane in enumerate(PLANES):
        ax = fig_cont.axes[idx_plane]
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        bottom_natqlabel = bottom_qlabel + 2 * get_approx_size_in_axes_coordinates(ax, label_size)

        # Tune Lines ---
        for line_params in (("", fig_cont.tunes, lines.tunes, "--", bottom_qlabel),
                            ("NAT", fig_cont.nattunes, lines.nattunes, ":", bottom_natqlabel)):
            _plot_tune_lines(ax, trans, label_size, *line_params)

        # Manual Lines ---
        for mline in lines.manual:
            loc = mline.pop('loc', None)  # needs to be removed in axvline
            text = mline.pop('text', None)
            plot_vertical_line(ax, mline,  text, loc, label_size)
            mline['loc'] = loc  # reset it for later axes/plots
            mline['text'] = text


def _plot_tune_lines(ax, transform, label_size, q_string, tunes, resonances, linestyle, label_y):
    if len(resonances) == 0:
        return

    if all(len(tunes[p]) == 0 for p in PLANES):
        LOG.warning(f"Resonance lines can't be plotted for {q_string}, "
                    "as no tunes were found in files.")
        return

    pref = q_string[0] if len(q_string) else ""
    q_mean = _get_evaluated_tune_array(np.mean, tunes)
    q_min = _get_evaluated_tune_array(np.min, tunes)
    q_max = _get_evaluated_tune_array(np.max, tunes)
    freqs_mean = _get_resonance_frequencies(resonances, q_mean)
    freqs_min = _get_resonance_frequencies(resonances, q_min)
    freqs_max = _get_resonance_frequencies(resonances, q_max)
    for res, f_mean, f_min, f_max in zip(resonances, freqs_mean, freqs_min, freqs_max):
        if not np.isnan(f_mean):
            label, order = f'{pref}({res[0]}, {res[1]})', sum(np.abs(res)) + 1
            color = get_cycled_color(order-2)
            ax.axvline(x=f_mean, label=label,
                       linestyle=linestyle, color=color, marker='None',
                       zorder=-1, alpha=VERTICAL_LINES_ALPHA)
            ax.text(x=f_mean, y=label_y, s=label, transform=transform,
                    color=color,
                    va='bottom', ha='center',
                    fontdict={'size': label_size})
            ax.add_patch(Rectangle(xy=(f_min, 0), width=f_max-f_min, height=1,
                                   transform=transform, color=color, alpha=PATCHES_ALPHA, zorder=-2,))


def _get_resonance_frequencies(resonances, q):
    """ Calculates the frequencies for the resonance lines,
    but also filters lines in case the tune was not found. """
    resonances = np.array(resonances)

    # find zero-tune filter:
    # if tune in plane is not used (i.e. coefficient is zero) we can still plot the line
    use_idx = np.ones(resonances.shape[0], dtype=bool)
    for idx, tune in enumerate(q):
        if tune == 0:
            use_idx &= resonances[:, idx] == 0

    if sum(use_idx) == 0:
        LOG.warning("No usable tunes found to calculate resonance frequencies. "
                    " Maybe you gave natural lines for a free kick?")

    freqs = np.mod(resonances @ q, 1)
    freqs = np.where(freqs > .5, 1 - freqs, freqs)

    freqs.dtype = np.float64  # in case of all zeros, this is int and causes crash with float-nan
    freqs[~use_idx] = np.nan
    return freqs


def _get_evaluated_tune_array(fun, tunes):
    """ Array of tunes per plane that evaluates the tunes by fun,
    returns 0 where no tunes are present.
    """
    return np.array([fun(tunes[p]) if len(tunes[p]) else 0 for p in PLANES])


# ID Finder --------------------------------------------------------------------


def get_stem_id(filename: str, bpm: str, output_dir: str, combine_by: frozenset, filetype: str) -> IdData:
    """ Returns the stem-dictionary id and the path to which the output file should be written.
    By using more or less unique identifiers, this controls the creation of figures in the dictionary."""

    fun_map = {
        _fset("bpms", "files"): _get_id_single_fig_files_and_bpms,
        _fset("files"): _get_id_single_fig_files,
        _fset("bpms"): _get_id_single_fig_bpms,
        _fset(): _get_id_multi_fig,
    }
    return fun_map[combine_by](
        output_dir, STEM_FILENAME, filename, bpm, filetype
    )


def get_waterfall_id(filename: str, bpm: str, output_dir: str, combine_by: frozenset, filetype: str) -> IdData:
    """ Returns the waterfall-dictionary id and the path to which the output file should be written.
    By using identifiers for figures and unique lables per figure,
    this controls the creation of figures in the dictionary."""
    fun_map = {
        _fset("bpms", "files"): _get_id_single_fig_files_and_bpms,
        _fset("files"): _get_id_single_fig_files,
        _fset("bpms"): _get_id_single_fig_bpms,
        _fset(): _get_id_single_fig_bpms,  # same as above as single figure per file AND
    }                                      # bpm does not make sense for waterfall
    return fun_map[combine_by](
        output_dir, WATERFALL_FILENAME, filename, bpm, filetype
    )


def _fset(*args):
    """ Frozen Set shortcut for dict-key readability"""
    return frozenset(args)

# Specific Mappings ---


def _get_id_single_fig_files_and_bpms(output_dir: str, default_name: str, filename: str,
                                      bpm: str, filetype: str) -> IdData:
    """ Same id for all plots. Creates single figure.
    The label of the lines is a combination of filename and bpm.
    """
    return IdData(
        id_=default_name,
        label=f"{filename} {bpm}",
        path=_get_figure_path(output_dir, filename=None,
                              figurename=f"{default_name}.{filetype}")
    )


def _get_id_single_fig_files(output_dir: str, default_name: str, filename: str,
                             bpm: str, filetype: str) -> IdData:
    """ BPM as id for plots.
    Creates len(bpm) figures, with filenames as labels for lines.
    """
    return IdData(
        id_=bpm,
        label=filename,
        path=_get_figure_path(output_dir, filename=None,
                              figurename=f"{default_name}_{bpm}.{filetype}")
    )


def _get_id_single_fig_bpms(output_dir: str, default_name: str, filename: str,
                            bpm: str, filetype: str) -> IdData:
    """ Filename as ID for plots.
    Creates len(files) figures, with bpms as lables for lines.
    """
    return IdData(id_=filename,
                  label=bpm,
                  path=_get_figure_path(output_dir, filename=filename,
                                        figurename=f"{default_name}.{filetype}")
                  )


def _get_id_multi_fig(output_dir: str, default_name: str, filename: str,
                      bpm: str, filetype: str) -> IdData:
    """ Combination of Filename and BPM as ID. Creates len(files)*len(bpms) plots.
    BPM-name is printed as label.
    """
    return IdData(id_=f"{filename}_{bpm}",
                  label=bpm,
                  path=_get_figure_path(output_dir, filename=filename,
                                        figurename=f"{default_name}_{bpm}.{filetype}")
                  )


# Data Sorting -----------------------------------------------------------------


def get_data_for_bpm(data: dict, bpm: str, rescale: bool) -> dict:
    """ Loads data from files and returns a dictionary (over planes) of a
    dictionary over the files containing the bpm data as pandas series. """
    data_series = {p: {} for p in PLANES}
    for plane in PLANES:
        try:
            freqs = data[FREQS][plane].loc[:, bpm]
            amps = data[AMPS][plane].loc[:, bpm]
            lin = data[LIN][plane].loc[bpm, :]
        except KeyError:  # bpm not in this plane
            data_series[plane] = None
        else:
            idxs_data = _get_valid_indices(amps, freqs)
            data_series[plane][LIN] = lin
            data_series[plane][FREQS] = freqs.loc[idxs_data]
            data_series[plane][AMPS] = amps.loc[idxs_data]
            if rescale:
                data_series[plane][AMPS] = rescale_amp(data_series[plane][AMPS])

            if any(data_series[plane][AMPS].isna()):
                raise Exception("NAN FOUND")
    return data_series


def get_unique_filenames(files: Union[Iterable, Sized]):
    """ Way too complicated method to assure unique dictionary names,
        by going backwards through the file-path until the names differ.
    """
    paths = [None] * len(files)
    names = [None] * len(files)
    parts = -1
    for idx, fpath in enumerate(files):
        fpath = Path(fpath)
        fname = _get_partial_filepath(fpath, parts)
        while fname in names:
            parts -= 1
            for idx_old in range(idx):
                names[idx_old] = _get_partial_filepath(paths[idx_old], parts)
            fname = _get_partial_filepath(fpath, parts)
        names[idx] = fname
        paths[idx] = fpath
    return zip(paths, names)


def _get_partial_filepath(path: Path, nparts: int):
    """ Returns the path from nparts until the end"""
    return path.parts[nparts:]


def _get_valid_indices(amps, freqs):
    """ Intersection of filtered AMPS and FREQS indices. """
    return index_filter(amps).intersection(index_filter(freqs))


def index_filter(data: pd.Series):
    """ Only non-NaN and non-Zero data allowed.
    (Amps should not be zero due to _filter_amps() anyway.)"""
    return data[~(data.isna() | (data == 0))].index


def filter_amps(files: dict, limit: float):
    """ Filter amplitudes by limit. """
    for plane in PLANES:
        filter_idx = files[AMPS][plane] <= limit
        files[AMPS][plane][filter_idx] = np.NaN
        files[FREQS][plane][filter_idx] = np.NaN
    return files


def get_bpms(lin_files: dict, given_bpms: Iterable, filename: str, planes: Iterable = PLANES) -> dict:
    """ Return the bpm-names of the given bpms as found in the lin files.
     'file_path' is only used for the error messages."""
    found_bpms = {}
    empty_planes = 0
    for plane in planes:
        found_bpms[plane] = list(lin_files[plane].index)
        if given_bpms is not None:
            found_bpms[plane] = _get_only_given_bpms(found_bpms[plane], given_bpms, plane, filename)

        if len(found_bpms[plane]) == 0:
            LOG.warning(f"(id:{filename}) No BPMs found for plane {plane}!")
            empty_planes += 1

    if empty_planes == len(planes):
        raise IOError(f"(id:{filename}) No BPMs found in any plane!")
    return found_bpms


def _get_only_given_bpms(found_bpms, given_bpms, plane, file_path):
    found_bpms = [bpm for bpm in found_bpms if bpm in given_bpms]
    missing_bpms = [bpm for bpm in given_bpms if bpm not in found_bpms]
    if len(missing_bpms):
        LOG.warning(
            f"({file_path}) The following BPMs are not present or not present in plane {plane}:"
            f" {list2str(missing_bpms)}"
        )
    return found_bpms


def rescale_amp(amp_data: pd.Series) -> pd.Series:
    # return amp_data.divide(amp_data.max(axis=0), axis=1)  # dataframe
    return amp_data.divide(amp_data.max(skipna=True))  # series


# For Output ---


def output_plot(fig_cont: FigureContainer):
    fig = fig_cont.fig
    if fig_cont.path is not None:
        LOG.info(f"Saving Plot '{fig_cont.path}'")
        fig.savefig(fig_cont.path)


def _get_figure_path(out_dir, filename, figurename):
    path = _make_output_dir(out_dir, filename)
    if path is not None and figurename is not None:
        path = os.path.join(path, figurename)
    return path


def _make_output_dir(out_dir, filename):
    if out_dir is not None:
        if filename is not None:
            out_dir = os.path.join(out_dir, os.path.splitext(filename)[0])
        os.makedirs(out_dir, exist_ok=True)
    return out_dir


# Plotting Helper --------------------------------------------------------------


def get_cycled_color(idx: int):
    """ Get the color at (wrapped) idx in the color cycle. The CN-Method only works until 'C9'."""
    cycle = matplotlib.rcParams[u"axes.prop_cycle"].by_key()['color']
    return cycle[idx % len(cycle)]


def get_approx_size_in_axes_coordinates(ax: axes.Axes, label_size: float) -> float:
    transform = ax.transAxes.inverted().transform
    _, label_size_ax = transform((0, label_size)) - transform((0, 0))
    return label_size_ax


def list2str(list_: list):
    return str(list_)[1:-1]


# Spectrum File Loading --------------------------------------------------------


def load_spectrum_data(file_path: Path, bpms: Iterable, planes: Iterable = PLANES):
    """ Load Amps, Freqs and Lin Files into a dictionary, keys are the fileendings without plane,
     with subdicts of the planes. """
    LOG.info("Loading HARPY data.")
    with suppress(FileNotFoundError):
        return _get_harpy_data(file_path, planes)

    LOG.info("Some files not present. Loading SUSSIX data format")
    with suppress(FileNotFoundError):
        return _get_sussix_data(file_path, bpms, planes)

    raise FileNotFoundError(f"Neither harpy nor sussix files found in '{file_path.parent}' "
                            f"matching the name '{file_path.name}'.")


# Harpy Data ---


def _get_harpy_data(file_path, planes):
    return {
        AMPS: _get_planed_files(file_path, ext=FILE_AMPS_EXT, planes=planes),
        FREQS: _get_planed_files(file_path, ext=FILE_FREQS_EXT, planes=planes),
        LIN: _get_planed_files(file_path, ext=FILE_LIN_EXT, planes=planes, index=COL_NAME),
    }


def _get_planed_files(file_path, ext, planes, index=None):
    return {
        plane: tfs.read(
            str(file_path.with_name(file_path.name + ext.format(plane=plane.lower()))),
            index=index)
        for plane in planes
    }


# SUSSIX Data ---


def _get_sussix_data(file_path, bpms, planes):
    bpm_dir = file_path.parent / 'BPM'
    files = {LIN: {}, AMPS: {}, FREQS: {}}
    for plane in planes:
        files[LIN][plane] = tfs.read(
            str(file_path.with_name(f'{file_path.name}_lin{plane.lower()}')),
            index=COL_NAME)
        for id_ in (FREQS, AMPS):
            files[id_][plane] = tfs.TfsDataFrame(columns=bpms)
        for bpm in bpms:
            with suppress(FileNotFoundError):
                df = tfs.read(str(bpm_dir / f'{bpm}.{plane.lower()}'))
                files[FREQS][plane][bpm] = df["FREQ"]
                files[AMPS][plane][bpm] = df["AMP"]
        for id_ in (FREQS, AMPS):
            files[id_][plane] = files[id_][plane].fillna(0)
    return files
