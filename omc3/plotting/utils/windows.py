"""
Plot Window
------------

In this module different classes are defined, allowing to put `matplotlib` figures 
manually into GUI windows.
These windows are QT-based, created with `qtpy` which allows to use 
either PySide(2 or 6) or PyQt(5 or 6), depending on which it installed on the system.
As the `qtpy` library is optional, there are some checks to make sure 
the imports do not fail, but then the classes cannot be used.
To check if QtPy is installed, either run :meth:`omc3.plotting.plot_window.is_qtpy_installed`
or try to initialize one of the windows, which will fail with a TypeError, 
as they want to call QApplication which is set to `None`.

An exception to all of this is  :meth:`omc3.plotting.plot_window.create_pyplot_window_from_fig`,
which allows to create a `pyplot` handled window from an already existing figure.
This way, figures can be created without manager (which makes them resource friendlier)
and can either be added to a QT-Window or, if not installed, opened by `pyplot`.
"""
from __future__ import annotations

import sys

import matplotlib
from matplotlib import pyplot as plt, rcParams
from matplotlib.figure import Figure

from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)

# --- optional qtpy import block -----------------------------------------------

try:
    from qtpy.QtWidgets import (
        QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
    )
except ImportError as e:
    LOG.debug(f"Could not import QtPy: {str(e)}")
    QMainWindow, QApplication, QVBoxLayout  = None, None, None
    FigureCanvas, NavigationToolbar = None, None  # for mock in pytest
    QWidget, QTabWidget = object, object
else:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    try:
        matplotlib.use('qtagg')
    except ImportError as e:
        if "headless" not in str(e):
            raise
        LOG.debug("Could not change mpl to use QT, due to headless mode (i.e. no display connected).")

# ------------------------------------------------------------------------------


def is_qtpy_installed():
    """Returns True if QtPy is installed."""
    return QMainWindow is not None


def log_no_qtpy_many_windows():
    """Logs a warning that QtPy is not installed and many windows will be opened."""

    LOG.warning(
        "QtPy is not installed. "
        "Plots will be shown in individual windows. "
        "Install QtPy for a more organized representation. "
    )


class PlotWidget(QWidget):

    def __init__(self, *figures: Figure, title: str):
        """A derived QWidget that contains Matplotlib-Figures,
        stacked vertically, each with its own toolbar.

        Args:
            *figures (Figure): Figures to be contained in the widget
            title (str): Name of the widget
        """
        super().__init__()

        self.title: str = title
        self.figures: tuple[Figure, ...] = figures
        self._canvas_toolbar_handles: list[tuple[FigureCanvas, NavigationToolbar]] = []

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        for figure in self.figures:
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas, self)

            self._layout.addWidget(canvas)
            self._layout.addWidget(toolbar)
            self._canvas_toolbar_handles.append((canvas, toolbar))


class TabWidget(QTabWidget):

    def __init__(self, title: str = None):
        """A simple tab widget, that in addition
        to QTabWidget keeps track of the tabs via dictionary
        and can have a title.
        
        Args:
            title (str): Title of the created Window
        """
        super().__init__()

        self.title: str = title
        self.tabs: dict[str, QWidget] = {}

    def add_tab(self, widget: QWidget):
        self.tabs[widget.title] = widget
        self.addTab(widget, widget.title)


class SimpleTabWindow:

    def __init__(self, title: str = "Simple Tab Window", size: tuple[int, int] = None):
        """A simple GUI window, i.e. a standalone graphical application, 
        which contains a single Tab-Widget, allowing the user to add tabs to it.

        Args:
            title (str): Title of the created Window
            size (Tuple[int, int]): Size of the created window in pixels.
        """
        self.app = QApplication(sys.argv)
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle(title)
        self.current_window = -1
        self.tabs_widget = TabWidget()
        self.main_window.setCentralWidget(self.tabs_widget)
        
        if size is None:
            size_in_inch = rcParams["figure.figsize"]
            size = (int(size_in_inch[0] * 100), int(size_in_inch[1] * 100))  # Approximate
        self.main_window.resize(*size)

        self.main_window.show()

    def add_tab(self, widget: QWidget):
        """ Add a new tab made from the given widget.

        Args:
            widget (QWidget): Widget to add.
        """
        self.tabs_widget.add_tab(widget)

    def show(self):
        self.app.exec()


class VerticalTabWindow(SimpleTabWindow):

    def __init__(self, title: str = "Vertical Tab Window", size: tuple[int, int] = None):
        """A Window in which the tabs are aligned vertically on the left-hand side.
        This window assumes that you may want to have tabs within tabs,
        so the convenience function `add_to_tab` is implemented, which allows
        you to add widgets directly to an already existing tab.

        Args:
            title (str): Title of the created Window.
            size (Tuple[int, int]): Size of the created window in pixels.
        """        
        super().__init__(title, size)
        self.tabs_widget.setTabPosition(QTabWidget.West)

    def add_to_tab(self, widget: QWidget, tab: str):
        """ Add `widget` to the tab with the given
        `tab` identifier.

        Args:
            widget (QWidget): Widget to add.
            tab (str): Title/Name of the tab .
        """
        self.tabs_widget.tabs[tab].add_tab(widget)

    def show(self):
        self.app.exec()


def create_pyplot_window_from_fig(fig: Figure, title: str = None):
    """Create a window from the given figure, which is managed by pyplot. 
    This is similar to how figures behave when created with `pyplot.figure()`,
    but you can crate the figure instance first and the manager later.

    Caveat: Uses private functions of pyplot.

    Args:
        fig (Figure): figure to be managed by pyplot. 
        title (str): Title of the window.
    """
    if fig.canvas.manager is not None:
        raise AttributeError('Figure already has a manager, cannot create a new one')

    allnums = plt.get_fignums()
    next_num = max(allnums) + 1 if allnums else 1
    manager = plt._get_backend_mod().new_figure_manager_given_figure(next_num, fig)
    plt._pylab_helpers.Gcf._set_new_active_manager(manager)
    if title is not None:
        manager.set_window_title(title)
