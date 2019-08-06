import matplotlib.pyplot as plt
from scipy.spatial.qhull import Delaunay
import numpy as np
from utils import logging_tools, outliers
from kmod import kmod_constants

LOG = logging_tools.get_logger(__name__)

PLANES = ['X', 'Y']


# class clicker_class(object):
#     def __init__(self, ax, data, pix_err=1):
#         self.canvas = ax.get_figure().canvas
#         self.cid = None
#         self.data = data
#         self.pt_lst = []
#         self.pt_plot = ax.plot([], [], marker='o', linestyle='-', zorder=5)[0]
#         self.cl_plot = ax.plot([], [], color='r', marker='o', linestyle='', zorder=5)[0]
#         self.tr_plot = ax.plot([], [], color='g', marker='o', linestyle='-', zorder=5)[0]
#         self.pix_err = pix_err
#         self.connect_sf()

#     def set_visible(self, visible):
#         '''sets if the curves are visible '''
#         self.pt_plot.set_visible(visible)

#     def clear(self):
#         '''Clears the points'''
#         self.pt_lst = []
#         x, y = [], []
#         self.pt_plot.set_xdata(x)
#         self.pt_plot.set_ydata(y)
#         self.cl_plot.set_xdata(x)
#         self.cl_plot.set_ydata(y)
#         self.tr_plot.set_xdata(x)
#         self.tr_plot.set_ydata(y)

#         self.canvas.draw()

#     def connect_sf(self):
#         if self.cid is None:
#             self.cid = self.canvas.mpl_connect('button_press_event',
#                                                self.click_event)
#             self.cid = self.canvas.mpl_connect('key_press_event',
#                                                self.key_event)

#     def disconnect_sf(self):
#         if self.cid is not None:
#             self.canvas.mpl_disconnect(self.cid)
#             # print self.data
#             # print self.cleaned_data
#             self.cid = None

#     def key_event(self, event):
#         ''' Extracts locations from the user'''
#         if event.key == 'c':
#             self.cleaned_data = self.data[0]
#             self.disconnect_sf()
#             plt.close()
#             return

#     def click_event(self, event):
#         ''' Extracts locations from the user'''
#         if event.key == 'shift':
#             self.pt_lst = []
#             self.redraw()
#             return
#         if event.xdata is None or event.ydata is None:
#             return
#         if event.button == 1:
#             self.pt_lst.append((event.xdata, event.ydata))
#             if len(self.pt_lst) > 4:
#                 self.disconnect_sf()
#                 plt.close()
#                 return
#         elif event.button == 3:
#             self.clear()
#         self.redraw()
#         if len(self.pt_lst) > 3:
#             self.start_clean()

#     def start_clean(self):
#         self.cleaned_data = clean(self.data, self.pt_lst)
#         self.cl_plot.set_xdata(self.cleaned_data[:, 0])
#         self.cl_plot.set_ydata(self.cleaned_data[:, 1])

#         pt_list = self.pt_lst
#         pt_list.append(pt_list[0])
#         ptdata = zip(*pt_list)

#         self.tr_plot.set_xdata(list(ptdata)[0])
#         self.tr_plot.set_ydata(list(ptdata)[1])
#         self.canvas.draw()

#     def remove_pt(self, loc):
#         if len(self.pt_lst) > 0:
#             self.pt_lst.pop(np.argmin(map(lambda x:
#                                           np.sqrt((x[0] - loc[0]) ** 2 +
#                                                   (x[1] - loc[1]) ** 2),
#                                           self.pt_lst)))

#     def redraw(self):
#         if len(self.pt_lst) > 0:
#             x, y = zip(*self.pt_lst)
#         else:
#             x, y = [], []
#         self.pt_plot.set_xdata(x)
#         self.pt_plot.set_ydata(y)

#         self.canvas.draw()

#     def return_clean_data(self):
#         '''Returns the clicked points in the format the rest of the
#         code expects'''
#         return self.cleaned_data


# def in_hull(p, hull):
#     """
#     Test if points in `p` are in `hull`

#     `p` should be a `NxK` coordinates of `N` points in `K` dimensions
#     `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
#     coordinates of `M` points in `K`dimensions for which Delaunay triangulation
#     will be computed
#     """
#     if not isinstance(hull, Delaunay):
#         hull = Delaunay(hull)

#     return hull.find_simplex(p) >= 0


# def clean(data, trapezium):
#     mask = in_hull([data[0, :, 0:2]], trapezium)
#     cleaned_data = data[mask]
#     return cleaned_data


# def manual_cleaning_data(k, tune_data, tune_data_err):
#     LOG.debug('Manual Tune cleaning')
#     data = np.dstack((k, tune_data, tune_data_err))
#     plt.figure(figsize=(15, 15))
#     plt.xlabel('K')
#     plt.ylabel('Tune')
#     plt.title('Left click: Select corners,  Right click: Cancel selection,  c: Skip')
#     plt.errorbar(k, tune_data, yerr=tune_data_err, fmt='o')
#     ax = plt.gca()
#     cc = clicker_class(ax, data)
#     plt.show()
#     return cc.return_clean_data()


def automatic_cleaning_data(k, tune_data, tune_data_err, limit=1e-5):
    LOG.debug('Automatic Tune cleaning')
    mask = outliers.get_filter_mask(tune_data, x_data=k, limit=limit)
    return mask


def clean_data(kmod_input_params, magnet_df):

    if kmod_input_params.no_autoclean == True:
        LOG.info('Manual cleaning is not yet implemented, no cleaning was performed')
        for plane in PLANES:
            magnet_df[kmod_constants.get_cleaned_col(plane)] = True

    else:
        for plane in PLANES:
            magnet_df[kmod_constants.get_cleaned_col(plane)] = automatic_cleaning_data(magnet_df[kmod_constants.get_k_col()].values, magnet_df[kmod_constants.get_tune_col( plane )].values, magnet_df[kmod_constants.get_tune_err_col( plane )].values )

    return magnet_df
