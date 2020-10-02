import os
from os.path import abspath, dirname, join
import tempfile
import numpy as np
import pytest
import tfs
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
    nonlinear=['crdt'],
    compensation="none",
    accel='lhc',
    ats=True,
    beam=1,
    dpp=0.0,
    model_dir=join(dirname(__file__), os.pardir, "inputs", "models", "inj_beam1"),
    year="2018",
)

LIN_DIR = join(dirname(__file__), os.pardir, "inputs", "crdt")

ORDERS = ['coupling', 'sextupole', 'skewsextupole', 'octupole']


@pytest.fixture(scope='module')
def _create_input():
    omc3_input = {}
    with tempfile.TemporaryDirectory() as cwd:
        for order in ORDERS:
            path_to_lin = join(LIN_DIR, order)
            optics_opt = MEASURE_OPTICS_SETTINGS.copy()
            optics_opt.update({
                'files': [join(path_to_lin, f'{order}{idx}') for idx in range(1, 4)],
                'outputdir': abspath(join(cwd, order)),
                })
            hole_in_one_entrypoint(**optics_opt)
            omc3_input[order] = (optics_opt, path_to_lin)
        yield omc3_input



@pytest.mark.extended
@pytest.mark.parametrize("order", ORDERS)
def test_crdt_amp(order, _create_input):
    omc3_input = _create_input
    (optics_opt, path_to_lin) = omc3_input[order]
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


@pytest.mark.extended
@pytest.mark.parametrize("order", ORDERS[:3])
def test_crdt_complex(order, _create_input):
    omc3_input = _create_input
    (optics_opt, path_to_lin) = omc3_input[order]
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


def _rel_dev(a, b, limit):
    a = a[np.abs(b) > limit]
    b = b[np.abs(b) > limit]
    return np.abs((a-b)/b)


def _max_dev(a, b, limit):
    return np.max(_rel_dev(a, b, limit))
