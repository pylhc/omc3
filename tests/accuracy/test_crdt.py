from os.path import abspath, dirname, isdir, join
from shutil import rmtree
import numpy as np
import pandas as pd
import pytest
import tfs
from omc3.definitions.constants import PLANES
from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.optics_measurements import crdt
from omc3.utils import stats

MEASURE_OPTICS_SETTINGS = dict(
    harpy=False,
    optics=True,
    nonlinear=['rdt', 'crdt'],
    compensation="none",
    accel='lhc',
    ats=False,
    beam=1,
    dpp=0.0,
    driven_excitation=None,
    drv_tunes=None,
    energy=None,
    fullresponse=False,
    model_dir=join(dirname(__file__), "..", "inputs", "models", "inj_beam1"),
    modifiers=None,
    nat_tunes=None,
    xing=False,
    year="2018",
)

ACCURACY_LIMIT = dict(
    Coupling=0.001,
    Sextupole=0.001,
    SkewSextupole=0.001,
    Octupole=0.29,
)

class BasicTests:
    @staticmethod
    def test_joined_planes():
        lin_files = {}
        for plane in PLANES:
            lin_files[plane] = [pd.DataFrame(data={'NAME':['A', 'B'],
                                                   'S':[1, 2],
                                                   f'TUNE{plane}':[0.28, 0.28],
                                                   f'AMP{plane}':[1, 1],
                                                   f'AMP20':[1, 1],
                                                   f'FREQ20':[0.31, 0.31],
                                                   })]
        result_df = crdt.joined_planes(lin_files)
        assert set(result_df[0].columns) == set(['S',
                                                 'TUNEX', 'AMPX',
                                                 'TUNEY', 'AMPY',
                                                 'AMP20_X', 'FREQ20_X',
                                                 'AMP20_Y', 'FREQ20_Y'])


    @staticmethod
    def test_average_results():
        result_dfs = []
        result_dfs.append(pd.DataFrame(index=['A', 'B'],
                                       data={'NAME':['A', 'B'],
                                             'S':[1, 2],
                                             'RES_A':[0.28, 0.28],
                                             'RES_B':[3, 1],
                                             'RES_C':[1, 2]}))
        result_dfs.append(pd.DataFrame(index=['A', 'C'],
                                       data={'NAME':['A', 'C'],
                                             'S':[1, 3],
                                             'RES_A':[0.28, 0.28],
                                             'RES_B':[1, 1],
                                             'RES_C':[np.nan, 1]}))
        result_df = crdt.average_results(result_dfs,
                                         ['NAME', 'S'],
                                         ['RES_A', 'RES_B', 'RES_C'],
                                         [np.nanmean, stats.weighted_nanmean, np.nanmean],
                                         'NAME')
        assert np.all(result_df == pd.DataFrame(index=['A', 'B', 'C'],
                                                data={'S': [1, 2, 3],
                                                      'RES_A': [0.28, 0.28, 0.28],
                                                      'RES_B': [2., 1., 1.],
                                                      'RES_C': [1., 2., 1.],
                                                     }))


class ExtendedTests:
    @staticmethod
    @pytest.mark.parametrize("order", ['Coupling', 'Sextupole', 'SkewSextupole', 'Octupole'])
    def test_crdt(order):
        path_to_lin = join(dirname(__file__), "..", "inputs", "crdt", order)
        optics_opt = MEASURE_OPTICS_SETTINGS.copy()
        optics_opt.update({
            'files': [join(path_to_lin, f'{order}{idx}') for idx in range(1, 4)],
            'outputdir':abspath(join(dirname(__file__), "..", "results", "crdt-test", order)),
        })
        hole_in_one_entrypoint(**optics_opt)
        ptc_crdt = tfs.read(join(path_to_lin, 'ptc_crdt.tfs'), index="NAME")

        for crdt_dict in crdt.CRDTS:
            if order == crdt_dict["order"]:
                print(crdt_dict["term"])
                hio_crdt = tfs.read(join(optics_opt["outputdir"], "crdt", order, f'{crdt_dict["term"]}.tfs'), index="NAME")
                fac = 1
                if order == 'Sextupole':
                    fac = 2
                if order == 'SkewSextupole':
                    fac = 2
                if order == 'Octupole':
                    fac = 4
                assert _max_dev(fac*hio_crdt["AMP"].to_numpy(), ptc_crdt[f"{crdt_dict['term']}_ABS"].to_numpy()) < ACCURACY_LIMIT[order]

        _clean_up(optics_opt["outputdir"])


def _max_dev(a, b):
    a = a[b > 0.01]
    b = b[b > 0.01]
    return np.max(np.abs((a-b)/b))

def _clean_up(path_dir):
    if isdir(path_dir):
        rmtree(path_dir, ignore_errors=True)
