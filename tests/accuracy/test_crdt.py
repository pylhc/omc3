from os.path import abspath, dirname, isdir, join
from shutil import rmtree
import numpy as np
import pandas as pd
import pytest
import tfs
from omc3.definitions.constants import PLANES
from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.optics_measurements import crdt

MEASURE_OPTICS_SETTINGS = dict(
    harpy=False,
    optics=True,
    nonlinear=['rdt', 'crdt'],
    compensation="none",
    accel='lhc',
    ats=True,
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
    Coupling=0.01,
    Sextupole=0.01,
    SkewSextupole=0.01,
    Octupole=0.14,
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
                hio_crdt = tfs.read(join(optics_opt["outputdir"],
                                         "crdt",
                                         order,
                                         f'{crdt_dict["term"]}.tfs'),
                                    index="NAME")
                assert _max_dev(hio_crdt["AMP"].to_numpy(),
                                ptc_crdt[f"{crdt_dict['term']}_ABS"].to_numpy(),
                                1E-2) < ACCURACY_LIMIT[order]

        _clean_up(optics_opt["outputdir"])

def _rel_dev(a, b, limit):
    a = a[b > limit]
    b = b[b > limit]
    return np.abs((a-b)/b)

def _max_dev(a, b, limit):
    return np.max(_rel_dev(a, b, limit))

def _clean_up(path_dir):
    if isdir(path_dir):
        rmtree(path_dir, ignore_errors=True)
