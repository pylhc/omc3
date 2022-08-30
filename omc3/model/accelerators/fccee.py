import logging

import json

from pathlib import Path
from typing import List

from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import AccElementTypes, Accelerator

LOGGER = logging.getLogger(__name__)
CURRENT_DIR = Path(__file__).parent

FCCee_DIR = CURRENT_DIR / 'fccee'

OPERATION_MODES={
    'z':{'energy':45.6},
    'w':{'energy':80.0},
    'h':{'energy':120.0},
    't':{'energy':182.5},
}


class FCCee(Accelerator):
    NAME = "fccee"
    RE_DICT = {AccElementTypes.BPMS: r"BPM*",
               AccElementTypes.MAGNETS: r"Q*",
               AccElementTypes.ARC_BPMS: r"BPM*"}
    BPM_INITIAL = 'B'

    @staticmethod
    def get_parameters():
        params = super(FCCee, FCCee).get_parameters()
        params.add_parameter(name="operation_mode", type=str, choices=list(OPERATION_MODES.keys()), help="FCCee operation mode.")
        params.add_parameter(name="lattice_version", type=str, help="FCCee lattice version, e.g. V22.")
        return params


    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
        self.operation_mode = opt.operation_mode
        self.lattice_version = opt.lattice_version

    def get_variables(self, frm=None, to=None, classes=None):
        if (frm is not None) or (to is not None):
            raise NotImplementedError('From and to have not been implemented')

        correctors_dir = FCCee_DIR / self.lattice_version /"correctors"
        all_corrs = _merge_jsons(
            correctors_dir / "quadrupole_correctors.json",
            correctors_dir / "skew_correctors.json"
        )

        my_classes = classes
        if my_classes is None:
            my_classes = all_corrs.keys()
        vars_by_class = set(
            _flatten_list([all_corrs[corr_cls] for corr_cls in my_classes if corr_cls in all_corrs])
        )

        return list(vars_by_class)


    def get_base_madx_script(self) -> str:
        madx_script = (
            f"! ----- Calling Sequence and Optics -----\n"
            )
        
        madx_script += (
            f"CALL, FILE='{FCCee_DIR/self.lattice_version/f'fccee_{self.operation_mode}.seq'}';\n"
            f"BEAM, PARTICLE=ELECTRON, ENERGY=sqrt(emass^2+{OPERATION_MODES[self.operation_mode]['energy']}^2), RADIATE=FALSE;\n"
            f"VOLTCA1=0;\n"
            f"VOLTCA2=0;\n"
            f"USE, SEQUENCE = FCCEE_P_RING;\n"
        )

        return madx_script

def _merge_jsons(*files) -> dict:
    full_dict = {}
    for json_file in files:
        with open(json_file, "r") as json_data:
            json_dict = json.load(json_data)
            for key in json_dict.keys():
                full_dict[key] = json_dict[key]
    return full_dict

def _flatten_list(my_list: List) -> List:
    return [item for sublist in my_list for item in sublist]
