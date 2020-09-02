from os.path import abspath, dirname, isdir, join
from shutil import rmtree
import numpy as np
import pandas as pd
import pytest
import tfs
from omc3.definitions.constants import PLANES
from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.optics_measurements import crdt

# accuracy limits of crdt to ptc, octupole is relaxed as single octupole with a nonideal WP gives weak CRDT
ACCURACY_LIMIT = dict(
    coupling=0.01,
    sextupole=0.01,
    skewsextupole=0.03,
    octupole=0.23,
)


# Levels below which CRDT are not used for test comparison
NOISELEVEL_AMP = dict(
    coupling=1E-2,
    sextupole=1E-2,
    skewsextupole=1E-2,
    octupole=0.5,
)
NOISELEVEL_COMPLEX = dict(
    coupling=1E-3,
    sextupole=1E-3,
    skewsextupole=1E-3,
    octupole=1.5,
)


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

LIN_DIR = join(dirname(__file__), "..", "inputs", "crdt")
RESULTS_PATH = join(dirname(__file__), "..", "results", "crdt-test")

def _create_input(order):
    path_to_lin = join(LIN_DIR, order)
    optics_opt = MEASURE_OPTICS_SETTINGS.copy()
    optics_opt.update({
        'files': [join(path_to_lin, f'{order}{idx}') for idx in range(1, 4)],
        'outputdir': abspath(join(RESULTS_PATH, order)),
    })
    hole_in_one_entrypoint(**optics_opt)
    return (optics_opt, path_to_lin)


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
    @pytest.mark.parametrize("order", ['coupling', 'sextupole', 'skewsextupole', 'octupole'])
    def test_crdt_amp(order, _precreated_input):
        (optics_opt, path_to_lin) = _precreated_input[order]
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
                                NOISELEVEL_AMP[order]) < ACCURACY_LIMIT[order]


    @staticmethod
    @pytest.mark.parametrize("order", ['coupling', 'sextupole', 'skewsextupole']) 
    def test_crdt_complex(order, _precreated_input):
        (optics_opt, path_to_lin) = _precreated_input[order]
        ptc_crdt = tfs.read(join(path_to_lin, 'ptc_crdt.tfs'), index="NAME")

        for crdt_dict in crdt.CRDTS:
            if order == crdt_dict["order"]:
                hio_crdt = tfs.read(join(optics_opt["outputdir"],
                                         "crdt",
                                         order,
                                         f'{crdt_dict["term"]}.tfs'),
                                    index="NAME")

                assert _max_dev(hio_crdt["REAL"].to_numpy(),
                                ptc_crdt[f"{crdt_dict['term']}_REAL"].to_numpy(),
                                NOISELEVEL_COMPLEX[order]) < ACCURACY_LIMIT[order]

                assert _max_dev(hio_crdt["IMAG"].to_numpy(),
                                ptc_crdt[f"{crdt_dict['term']}_IMAG"].to_numpy(),
                                NOISELEVEL_COMPLEX[order]) < ACCURACY_LIMIT[order]


    @classmethod
    def teardown_class(cls):
        _clean_up(RESULTS_PATH)


@pytest.fixture()
def _precreated_input():
    yield {order: _create_input(order) for order in ['coupling', 'sextupole', 'skewsextupole', 'octupole']}


def _rel_dev(a, b, limit):
    a = a[np.abs(b) > limit]
    b = b[np.abs(b) > limit]
    return np.abs((a-b)/b)


def _max_dev(a, b, limit):
    return np.max(_rel_dev(a, b, limit))


def _clean_up(path_dir):
    if isdir(path_dir):
        rmtree(path_dir, ignore_errors=True)
