from collections import Iterable
from pathlib import Path

from omc3.plotting.spectrum.utils import IdData, FigureCollector as SpectFigCollector
from matplotlib import pyplot as plt


class FigureContainer:
    """ Container for attaching additional information to one figure. """
    def __init__(self, path: Path, n_planes: int, combine_planes: bool) -> None:
        self.fig, axs = plt.subplots(nrows=1 if combine_planes else n_planes)

        if n_planes == 1:
            self.axes = [axs]
        elif combine_planes:
            self.axes = [axs for _ in range(n_planes)]
        else:
            self.axes = axs

        self.ylabels = [None for _ in range(n_planes)]
        self.data = [{} for _ in range(n_planes)]
        self.path = path


class FigureCollector:
    """ Class to collect figure containers and manage data adding. """
    def __init__(self) -> None:
        self.fig_dict = {}   # dictionary of matplotlib figures, for output
        self.figs = {}       # dictionary of FigureContainers, used internally

    def add_data_for_id(self, id_: str, label: str, data: dict, y_label: str,
                        path: Path = None, plane_idx: int = 0,
                        n_planes: int = 1, combine_planes: bool = False) -> None:
        """ Add the data at the appropriate figure container. """
        try:
            figure_cont = self.figs[id_]
        except KeyError:
            figure_cont = FigureContainer(path, n_planes, combine_planes)

            self.figs[id_] = figure_cont
            self.fig_dict[id_] = figure_cont.fig

        figure_cont.ylabels[plane_idx] = y_label  # always replaced but doesn't matters
        figure_cont.data[plane_idx][label] = data
