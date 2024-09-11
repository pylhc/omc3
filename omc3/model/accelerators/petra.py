"""
PETRA
-----

Accelerator-Class for the ``PETRA`` machine.

Model Creation Keyword Args:
    *--Optional--*

    - **dpp** *(float)*:

        Deltap/p to use.

        default: ``0.0``


    - **driven_excitation** *(str)*:

        Denotes driven excitation by `AC-dipole` (acd) or by `ADT` (adt)

        choices: ``('acd', 'adt')``


    - **drv_tunes** *(float)*:

        Driven tunes without integer part.


    - **energy** *(float)*:

        Energy in **Tev**.


    - **model_dir** *(str)*:

        Path to model directory; loads tunes and excitation from model!


    - **modifiers** *(str)*:

        Path to the optics file to use (modifiers file).


    - **nat_tunes** *(float)*:

        Natural tunes without integer part.


    - **xing**:

        If True, x-ing angles will be applied to model

        action: ``store_true``
"""
import json
from pathlib import Path
from generic_parser import EntryPoint
from omc3.model.accelerators.accelerator import Accelerator, AccElementTypes, AcceleratorDefinitionError
from omc3.model.constants import PLANE_TO_HV

CURRENT_DIR = Path(__file__).parent
PETRA_DIR = CURRENT_DIR / "petra"
EXCITER_BPM = "BPM_SOR_13"
MACROS_MADX = "macros.madx"
CORRECTOR_FILE = "correctors.json"
BETA_TO_SEQUENCE = {"low": "p3v24c4l.seq", "high": "p3x_v24.seq"}


class Petra(Accelerator):
    NAME = "petra"
    RE_DICT = {AccElementTypes.BPMS: r"BPM",
               AccElementTypes.MAGNETS: r".*",
               AccElementTypes.ARC_BPMS: r"BPM_(SWR_61|SWR_75|SWR_90|SWR_104|SWR_118|SWR_133|WL_140|WL_126|WL_111|WL_97|WL_82|WR_82|WR_97|WR_111|WR_126|WR_140|NWL_133|NWL_118|NWL_104|NWL_90|NWL_75|NWL_61|NWR_61|NWR_75|NWR_90|NWR_104|NWR_118|NWR_133|NL_140|NL_126|NL_111|NL_97|NL_82|NR_140|NOL_133|NOL_118|NOL_104|NOL_90|NOL_75|NOL_61|OR_140|SOL_133|SOL_118|SOL_104|SOL_90|SOL_75|SOL_61|SOL_54|SOR_61|SOR_75|SOR_90|SOR_104|SOR_118|SOR_133|SL_140|SL_126|SL_111|SL_97|SL_82|SR_82|SR_97|SR_111|SR_126|SR_140|SWL_133|SWL_118|SWL_104|SWL_90|SWL_75|SWL_61|SWL_39)"
               }
    BETAS = ("low", "high")
    # Public Methods ##########################################################

    @classmethod
    def get_parameters(cls):
        params = super(Petra, Petra).get_parameters()
        params.add_parameter(name="beta", type=str, choices=cls.BETAS,
                             help="Chosen optics: 'low' or 'high' beta.")
        return params

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
        self.beta = opt.beta

    def verify_object(self):  # TODO: Maybe more checks?
        pass
        # if self.model_dir is None:  # is the class is used to create full response?
        #     raise AcceleratorDefinitionError("PETRA doesn't have a model creation yet, calling it this "
        #                                      "way is most probably wrong.")

    def get_exciter_bpm(self, plane, commonbpms):
        return [list(commonbpms).index(EXCITER_BPM), EXCITER_BPM], f"KIFB{PLANE_TO_HV[plane]}N"

    def get_base_madx_script(self, best_knowledge: bool = False) -> str:
        madx_script = (
            f"call, file = '{self.model_dir / MACROS_MADX}';\n"
            f"call, file = '{self.model_dir / BETA_TO_SEQUENCE[self.beta]}';\n\n"
            "BEAM, PARTICLE=POSITRON, ENERGY=6.0, bunched, RADIATE, sequence=RING;\n"
            "use, sequence=ring;\n"
        )
        madx_script += (
            f"!  natural tunes\n"
            f"Qx = {self.nat_tunes[0]};\n"
            f"Qy = {self.nat_tunes[1]};\n\n"
            "select, flag=twiss, clear;\n"
            "exec, match_tunes(Qx, Qy);\n"
            )
        return madx_script

    def get_update_correction_script(self, outpath: Path, corr_file: Path) -> str:
        madx_script = self.get_base_madx_script()
        madx_script += (
            f"call, file = '{str(corr_file)}';\n"
            f"exec, do_twiss_elements('{str(outpath)}');\n"
        )
        return madx_script

    def get_variables(self, frm=None, to=None, classes=None):
        with open(PETRA_DIR / CORRECTOR_FILE, "r") as json_data:
            all_corrs = json.load(json_data)
        my_classes = classes
        if my_classes is None:
            my_classes = all_corrs.keys()
        vars_by_class = set(
            _flatten_list([all_corrs[corr_cls] for corr_cls in my_classes if corr_cls in all_corrs])
        )
        return list(vars_by_class)


def _flatten_list(my_list):
    return [item for sublist in my_list for item in sublist]
