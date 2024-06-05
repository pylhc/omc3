import tfs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from generic_parser import DotDict
from generic_parser.entrypoint_parser import EntryPointParameters, entrypoint
from omc3.utils import logging_tools
from omc3.utils.iotools import PathOrStr

LOG = logging_tools.get_logger(__name__)


def kmod_average_params():
    """
    A function to create and return EntryPointParameters for Kmod average.
    """
    params = EntryPointParameters()
    params.add_parameter(name="meas_paths",
                         required=True,
                         nargs='+',
                         help="Directories of Kmod results to import")
    params.add_parameter(name="ip",
                         required=True,
                         type=int,
                         help="Specific ip to average over")
    params.add_parameter(name="beta",
                         required=True,
                         type=float,
                         help="Model beta of measurements")
    params.add_parameter(name="output_dir",
                         required=True,
                         type=PathOrStr,
                         help="Path to the directory where to write the output files.")
    return params


@entrypoint(kmod_average_params(), strict=True)
def average_kmod_results_entrypoint(opt: DotDict) -> None:
    opt.meas_paths = [Path(m) for m in opt.meas_paths]
    opt.output_dir = Path(opt.output_dir)
    opt.output_dir.mkdir(exist_ok=True)
    averaged_results = get_average_betastar_results(opt)
    averaged_bpm_results = get_average_bpm_betas_results(opt)
    plot_results(opt, averaged_results)


def get_average_betastar_results(opt):
    """
    Calculate the average betastar results for the given parameters.

    Args:
        opt: The parameters for calculation.

    Returns:
        The final results as a DataFrame.
    """ 
    final_results = []
    for beam in [1, 2]:
        all_dfs = []
        for mpath in opt.meas_paths:
            all_dfs.append(tfs.read(mpath / f'B{beam}' / 'results.tfs').drop(columns=['LABEL', 'TIME']))
        
        panel = np.array(all_dfs)          
        # Calculate mean and std along the new axis (axis=0)
        mean_df = pd.DataFrame(panel.mean(axis=0), index=all_dfs[0].index, columns=all_dfs[0].columns)
        std_df = pd.DataFrame(panel.std(axis=0), index=all_dfs[0].index, columns=all_dfs[0].columns)

        for column in mean_df.columns:
            if not column.startswith('ERR'):
                mean_df[f'ERR{column}'] = std_df[column]

        mean_df['BETSTARMDL'] = opt.beta
        mean_df['BEAM'] = beam
        final_results.append(mean_df)
    final_results = pd.concat(final_results)
    tfs.write(opt.output_dir / f'averaged_ip{opt.ip}_beta{opt.beta}m.tfs', final_results)
    return final_results


def get_average_bpm_betas_results(opt):
    """
    Calculate the average bpm betas results for the given parameters.

    Args:
        opt: The parameters for the calculation.

    Returns:
        final_results: A dictionary containing the average bpm betas results for each beam.
    """
    final_results = {}

    for beam in [1, 2]:
        all_dfs = []
        for mpath in opt.meas_paths:
            all_dfs.append(tfs.read(mpath / f'B{beam}' / 'lsa_results.tfs').set_index('NAME'))
        
        panel = np.array(all_dfs)          
        # Calculate mean and std along the new axis (axis=0)
        mean_df = pd.DataFrame(panel.mean(axis=0), index=all_dfs[0].index, columns=all_dfs[0].columns)
        std_df = pd.DataFrame(panel.std(axis=0), index=all_dfs[0].index, columns=all_dfs[0].columns)
        
        mean_df['ERRBETX'] = std_df['BETX']
        mean_df['ERRBETY'] = std_df['BETY']
        mean_df = mean_df.reset_index()
        final_results[beam] = mean_df
        tfs.write(opt.output_dir / f'averaged_bpm_beam{beam}_ip{opt.ip}_beta{opt.beta}m.tfs', mean_df)
    
    return final_results


def plot_results(opt, results):
    """
    Function to plot the resulting average beta functions.

    Parameters:
    - opt: input options
    - results: the calculated average betas 

    Returns:
    None
    """
    fig, ax = set_square_axes()
    results = results.set_index('BEAM')
    for beam in [1,2]:
        ax.errorbar((results.loc[beam, 'BETSTARX']-results.loc[beam, 'BETSTARMDL'])/results.loc[beam, 'BETSTARMDL']*100,
                    (results.loc[beam, 'BETSTARY']-results.loc[beam, 'BETSTARMDL'])/results.loc[beam, 'BETSTARMDL']*100,
                    xerr=results.loc[beam, 'ERRBETSTARX']/results.loc[beam, 'BETSTARMDL']*100,
                    yerr=results.loc[beam, 'ERRBETSTARY']/results.loc[beam, 'BETSTARMDL']*100,
                    fmt='o:', color=f'C{beam-1}',
                    label=f'B{beam} IP{opt.ip}')

    ax.set_xlabel(r'$\Delta\beta_x/\beta_x$ %', fontsize=14)
    ax.set_ylabel(r'$\Delta\beta_y/\beta_y$ %', fontsize=14)
    all_ticks = np.concatenate([plt.gca().get_xticks() , plt.gca().get_yticks()])
    max_tick = np.max(np.abs(all_ticks))
    ax.set_xlim(-max_tick, max_tick)
    ax.set_ylim(-max_tick, max_tick)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=14)
    fig.savefig(f'{opt.output_dir}/ip{opt.ip}.png', dpi=400)
    plt.show()


def set_square_axes(figsize=(6,6), axes_loc=[0.17, 0.15, 0.8, 0.7]):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax = plt.axes(axes_loc)
    ax.set_aspect('equal')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return fig, ax


if __name__ == "__main__":
    average_kmod_results_entrypoint()
