from pathlib import Path
import numpy as np
import pytest
import tfs
from omc3.hole_in_one import hole_in_one_entrypoint


# the coupling test for real and imaginary are skipped for now as the tests fail
SKIP_REASON = "Coupling is skipped as there, Real and Imag are not aligned with model. Issue to be looked into."

# accuracy limits of rdt to ptc
ACCURACY_LIMIT = dict(
    skew_quadrupole=0.01,
    normal_sextupole=0.12,
)


MEASURE_OPTICS_SETTINGS = dict(
    harpy=False,
    optics=True,
    nonlinear=['rdt'],
    compensation="none",
    accel='lhc',
    ats=True,
    dpp=0.0,
    year="2018",
)


RDTS = (
    {'order': "skew_quadrupole", 'jklm': [1,0,0,1], 'plane': 'X'},
    {'order': "skew_quadrupole", 'jklm': [1,0,1,0], 'plane': 'X'},
    {'order': "skew_quadrupole", 'jklm': [0,1,1,0], 'plane': 'Y'},
    {'order': "skew_quadrupole", 'jklm': [1,0,1,0], 'plane': 'Y'},

    {'order': "normal_sextupole", 'jklm': [3,0,0,0], 'plane': 'X'},
    {'order': "normal_sextupole", 'jklm': [1,2,0,0], 'plane': 'X'},
    {'order': "normal_sextupole", 'jklm': [1,0,2,0], 'plane': 'X'},
    {'order': "normal_sextupole", 'jklm': [1,0,0,2], 'plane': 'X'},
    {'order': "normal_sextupole", 'jklm': [0,1,1,1], 'plane': 'Y'},
    {'order': "normal_sextupole", 'jklm': [1,0,2,0], 'plane': 'Y'},
    {'order': "normal_sextupole", 'jklm': [0,1,2,0], 'plane': 'Y'},
    {'order': "normal_sextupole", 'jklm': [1,0,1,1], 'plane': 'Y'},
)

# BPMs are given for each test allow to avoid edge effects
USE_BPMS = {
    'B1':{
        'skew_quadrupole': ["BPM.29R1.B1", "BPM.30R1.B1", "BPM.31R1.B1", "BPM.32R1.B1", "BPM.33R1.B1", "BPM.34R1.B1", "BPM.33L2.B1", "BPM.32L2.B1", "BPM.31L2.B1"],
        'normal_sextupole': ["BPM.27R1.B1", "BPM.28R1.B1"],
    },
    'B2':{
        'skew_quadrupole': ["BPM.29R1.B2", "BPM.30R1.B2", "BPM.31R1.B2", "BPM.32R1.B2", "BPM.33R1.B2", "BPM.34R1.B2", "BPM.33L2.B2", "BPM.32L2.B2", "BPM.31L2.B2"],
        'normal_sextupole': ["BPM.27L1.B2", "BPM.26L1.B2"],
    }
}

# test data is generated using the scripts in https://github.com/pylhc/MESS/tree/master/LHC/Coupling_RDT_Bump and https://github.com/pylhc/MESS/tree/master/LHC/Sextupole_RDT_Bump
LIN_DIR = Path(__file__).parent.parent / "inputs" / "rdt"
ORDERS = ['skew_quadrupole', 'normal_sextupole']


@pytest.fixture(scope='module')
def _create_input(tmp_path_factory, request):
    omc3_input = {}

    for order in ORDERS:
        path_to_lin = LIN_DIR / order
        optics_opt = MEASURE_OPTICS_SETTINGS.copy()
        optics_opt.update({
            'files': [str(path_to_lin / f'B{request.param}_{order}{idx}') for idx in range(1, 4)],
            'outputdir': tmp_path_factory.mktemp(order).resolve(),
            'beam': request.param,
            'model_dir': Path(__file__).parent.parent / "inputs" / "models" / f"inj_beam{request.param}",
            })
        hole_in_one_entrypoint(**optics_opt)
        omc3_input[order] = (optics_opt, path_to_lin)
    yield omc3_input


@pytest.mark.extended
@pytest.mark.parametrize("_create_input", (1, 2), ids=["Beam1", "Beam2"], indirect=True)
@pytest.mark.parametrize("order", ORDERS)
def test_crdt_amp(order, _create_input):
    omc3_input = _create_input
    (optics_opt, path_to_lin) = omc3_input[order]
    model_rdt = tfs.read(path_to_lin / f'B{optics_opt["beam"]}_model_rdt.tfs', index="NAME")

    for rdt_dict in RDTS:
        if order == rdt_dict["order"]:
            jklm = "".join(map(str, rdt_dict["jklm"]))
            hio_rdt = tfs.read(optics_opt["outputdir"] / "rdt" / order / f'f{jklm}_{rdt_dict["plane"].lower()}.tfs',
                                index="NAME")
            bpms= USE_BPMS[f'B{optics_opt["beam"]}'][order]

            assert _max_dev(hio_rdt["AMP"][bpms].to_numpy(),
                            model_rdt[f"F{jklm}AMP"][bpms].to_numpy(),
                            0.0) < ACCURACY_LIMIT[order]


@pytest.mark.extended
@pytest.mark.parametrize("_create_input", (1, 2), ids=["Beam1", "Beam2"], indirect=True)
@pytest.mark.parametrize("order", 
    [pytest.param(order,
                  marks=pytest.mark.skip(reason=SKIP_REASON) if order=='skew_quadrupole' else pytest.mark.extended
                  ) for order in ORDERS]
                        )
def test_crdt_complex(order, _create_input):
    omc3_input = _create_input
    (optics_opt, path_to_lin) = omc3_input[order]
    model_rdt = tfs.read(path_to_lin / f'B{optics_opt["beam"]}_model_rdt.tfs', index="NAME")

    for rdt_dict in RDTS:
        if order == rdt_dict["order"]:
            jklm = "".join(map(str, rdt_dict["jklm"]))

            hio_rdt = tfs.read(optics_opt["outputdir"] / "rdt" / order / f'f{jklm}_{rdt_dict["plane"].lower()}.tfs',
                                index="NAME")
            
            bpms= USE_BPMS[f'B{optics_opt["beam"]}'][order]

            assert _max_dev(hio_rdt["REAL"][bpms].to_numpy(),
                            model_rdt[f"F{jklm}REAL"][bpms].to_numpy(),
                            0.0) < ACCURACY_LIMIT[order]

            assert _max_dev(hio_rdt["IMAG"][bpms].to_numpy(),
                            model_rdt[f"F{jklm}IMAG"][bpms].to_numpy(),
                            0.0) < ACCURACY_LIMIT[order]


def _rel_dev(a, b, limit):
    a = a[np.abs(b) > limit]
    b = b[np.abs(b) > limit]
    return np.abs((a-b)/b)


def _max_dev(a, b, limit):
    return np.max(_rel_dev(a, b, limit))
