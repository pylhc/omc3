"""
Helper
------

This module contains helper functionality for ``kmod``.
It provides functions to perform data cleaning, IO loading and plotting.
"""
import datetime

import numpy as np
import tfs
from matplotlib import pyplot as plt

from omc3.definitions import formats
from omc3.definitions.constants import PLANES
from omc3.kmod import analysis
from omc3.kmod.constants import SIDES, ERR, TUNE, EXT, CLEANED, K, AVERAGE, BETA
from omc3.utils import logging_tools, outliers

LOG = logging_tools.get_logger(__name__)


def clean_data(magnet_df, no_autoclean):
    if no_autoclean:
        LOG.info('Manual cleaning is not yet implemented, no cleaning was performed')
        for plane in PLANES:
            magnet_df[f"{CLEANED}{plane}"] = True
    else:
        LOG.debug('Automatic Tune cleaning')
        for plane in PLANES:
            magnet_df[f"{CLEANED}{plane}"] = outliers.get_filter_mask(
                magnet_df[f"{TUNE}{plane}"].to_numpy(), x_data=magnet_df[K].to_numpy(), limit=1e-5)
    return magnet_df


def add_tune_uncertainty(magnet_df, tune_uncertainty):
    LOG.debug(f'adding {tune_uncertainty} units tune measurement uncertainty')
    for plane in PLANES:
        magnet_df[f"{ERR}{TUNE}{plane}"] = np.sqrt(magnet_df[f"{ERR}{TUNE}{plane}"]**2 + tune_uncertainty**2)
    return magnet_df


# ##########################   FILE LOADING    ##########################################


def get_input_data(opt):
    return get_simulation_files(opt.working_directory, opt.beam, opt.magnets) if opt.simulation else merge_data(opt)


def get_simulation_files(working_directory, beam, magnets):
    magnet1_df = tfs.read(working_directory / f"{magnets[0]}.B{beam:d}{EXT}")
    magnet2_df = tfs.read(working_directory / f"{magnets[1]}.B{beam:d}{EXT}")
    magnet1_df.headers['QUADRUPOLE'] = magnet1_df.headers['NAME']
    magnet2_df.headers['QUADRUPOLE'] = magnet2_df.headers['NAME']
    return magnet1_df, magnet2_df


def merge_data(kmod_input_params):
    magnet_df = []
    work_dir = kmod_input_params.working_directory
    ip = kmod_input_params.interaction_point
    beam = kmod_input_params.beam
    for (filepaths, magnet) in zip(
            return_ip_filename(work_dir, ip, beam) if ip is not None
            else return_circuit_filename(work_dir, kmod_input_params.circuits, beam),
            kmod_input_params.magnets):
        LOG.debug(f'Loading tunes from {filepaths[0]} and {filepaths[1]}')
        tune_dfs = dict(X=tfs.read(filepaths[0]), Y=tfs.read(filepaths[1]))
        LOG.debug(f'Loading k from {filepaths[2]}')
        k_df = tfs.read(filepaths[2])
        LOG.debug('Binning data')
        magnet_df.append(bin_tunes_and_k(tune_dfs, k_df, magnet))
    return magnet_df


def return_ip_filename(working_directory, ip, beam):
    LOG.debug('Setting IP trim file names')
    all_filepaths = []
    for side in SIDES:
        path_tunex = working_directory / f"{ip.lower()}b{beam:d}{side}X{EXT}"
        path_tuney = working_directory / f"{ip.lower()}b{beam:d}{side}Y{EXT}"
        path_k = working_directory / f"{ip.lower()}{side}K{EXT}"
        all_filepaths.append([path_tunex, path_tuney, path_k])
    return all_filepaths


def return_circuit_filename(working_directory, circuits_1_and_2, beam):
    LOG.debug('Setting Circuit trim file names')
    all_filepaths = []
    for circuit in circuits_1_and_2:
        path_tunex = working_directory / f"{circuit}_tune_x_b{beam}{EXT}"
        path_tuney = working_directory / f"{circuit}_tune_y_b{beam}{EXT}"
        path_k = working_directory / f"{circuit}_k{EXT}"
        all_filepaths.append([path_tunex, path_tuney, path_k])
    return all_filepaths


def bin_tunes_and_k(tune_dfs, k_df, magnet):
    # create bins, centered around each time step in k with width eq half distance to the next timestep
    bins = np.append((k_df['TIME']-k_df.diff()['TIME']/2.).fillna(value=0).values, k_df['TIME'].iloc[-1])
    magnet_df = k_df.loc[:, ['K']]
    magnet_df['K'] = np.abs(magnet_df['K'].to_numpy())
    for plane in PLANES:
        magnet_df[f"{TUNE}{plane}"], magnet_df[f"{ERR}{TUNE}{plane}"] = return_mean_of_binned_data(bins, tune_dfs[plane])
    return tfs.TfsDataFrame(magnet_df, headers=headers_for_df(magnet, k_df))


def return_mean_of_binned_data(bins, tune_df):
    digitize = np.digitize(tune_df['TIME'], bins)
    mean = [tune_df['TUNE'][digitize == i].mean() for i in range(1, len(bins))]
    std = np.nan_to_num([tune_df['TUNE'][digitize == i].std() for i in range(1, len(bins))])
    return mean, std


def headers_for_df(magnet, k_df):
    LOG.debug('Creating headers for DF')
    head = {}
    head['QUADRUPOLE'] = magnet
    head['DELTA_I'] = np.max(k_df['CURRENT'].rolling(5).mean()) - np.min(k_df['CURRENT'].rolling(5).mean()) / 2
    head['START_TIME'] = datetime.datetime.fromtimestamp(k_df['TIME'].iloc[0] / 1000).strftime(formats.TIME)
    head['END_TIME'] = datetime.datetime.fromtimestamp(k_df['TIME'].iloc[-1] / 1000).strftime(formats.TIME)
    # add starting tunes/tunesplit, number of cycles, ... to header
    return head


# ##############################    PLOTING    #############################################


def plot_cleaned_data(magnet_dfs, plot_name, interactive_plot=False):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i, plane in enumerate(PLANES):
        for j in range(2):
            ax_plot(ax[i, j], magnet_dfs[j], plane)
    ax[1, 1].legend()
    plt.tight_layout()
    plt.savefig(plot_name)
    if interactive_plot:
        plt.show()
    return


def ax_plot(ax, magnet_df, plane):
    ax.set_title(magnet_df.headers['QUADRUPOLE'], fontsize=15)
    ax_errorbar_plot(ax=ax, magnet_df=magnet_df, plane=plane, clean=True,
                     plot_settings={"color": "blue", "marker": "o", "label": "Data", "zorder": 1})
    ax_errorbar_plot(ax=ax, magnet_df=magnet_df, plane=plane, clean=False,
                     plot_settings={"color": "orange", "marker": "o", "label": "Cleaned", "zorder": 2})
    ax.plot((magnet_df.where(magnet_df[f"{CLEANED}{plane}"])[K].dropna() - magnet_df.headers[K]) * 1E3,
            analysis.fit_prec(
                analysis.return_fit_input(magnet_df, plane),
                magnet_df.headers[f"{AVERAGE}{BETA}{plane}"]) + magnet_df.headers[f"{TUNE}{plane}"],
            color='red', label='Fit', zorder=3)
    ax.set_xlabel(r'$ \Delta K $', fontsize=15)
    ax.set_ylabel(r'$ Q_{{{:s}}} $'.format(plane), fontsize=15)
    return


def ax_errorbar_plot(ax, magnet_df, plane, clean, plot_settings):
    new_df = magnet_df.loc[magnet_df.loc[:, f"{CLEANED}{plane}"].to_numpy() == clean, :]
    ax.errorbar((new_df.loc[:, K].dropna() - magnet_df.headers[K]) * 1E3,
                new_df.loc[:, f"{TUNE}{plane}"].dropna(),
                yerr=new_df.loc[:, f"{ERR}{TUNE}{plane}"].dropna(),
                color=plot_settings["color"],
                fmt=plot_settings["marker"],
                label=plot_settings["label"],
                zorder=plot_settings["zorder"]
                )
    return ax
