import os
from os.path import join
import tfs
from generic_parser import entrypoint, EntryPointParameters
import re
from omc3.utils import logging_tools
from omc3.definitions.constants import PLANES
from omc3.optics_measurements.constants import EXT

KMOD_BETA_NAME = "beta_kmod_"
KMOD_BETA_STAR_NAME = "beta_star_kmod"
KMOD_BETA_NAME_INPUT = "not_sure"
KMOD_BETA_STAR_NAME_INPUT = "not_sure"

LOG = logging_tools.get_logger(__name__)


def loader_params():
    params = EntryPointParameters()
    params.add_parameter(name="kmod_dir", type=str, required=True,
                         help="path to kmod directory with stored KMOD measurement files")
    params.add_parameter(name="res_dir", type=str, required=True, help="Output directory of the optics analysis.")
    params.add_parameter(name="mod_path", type=str, required=True,
                         help="Specify path to current model.")
    params.add_parameter(name="realizations", type=str, choices=("B1", "B2"),
                         help="Number of copies with added noise")
    return params


@entrypoint(loader_params(), strict=True)
def merge_and_copy_kmod_output(opt):
    pattern = re.compile(".*R[0-9]\." + opt.beam)
    model_twiss = tfs.read(opt.mod_path, index="NAME")
    ip_dir_names = [d for _, dirs, _ in os.walk(opt.kmod_dir) for d in dirs if pattern.match(d)]

    # copy beta data
    for plane in PLANES:
        plane = plane.upper()
        new_data = tfs.TfsDataFrame()
        for ip_dir_name in ip_dir_names:
            data = tfs.read(join(opt.kmod_dir, ip_dir_name, f"{KMOD_BETA_NAME_INPUT}{plane}{EXT}"), index="NAME")
            if data is not None:
                in_model = data.index.isin(model_twiss.index)
                new_data = new_data.append(data.loc[in_model, :])

        # Todo: Let Kmod fill these columns in the future
        new_data["S"] = model_twiss.loc[new_data.index, "S"]
        new_data[f"BET{plane}MDL"] = model_twiss.loc[new_data.index, f"BET{plane}"].to_numpy()
        tfs.write(join(opt.res_dir, f"{KMOD_BETA_NAME}{plane}{EXT}"), new_data, save_index="NAME")

    # copy beta* data
    new_data = tfs.TfsDataFrame()
    for ip_dir_name in ip_dir_names:
        data = tfs.read(join(opt.kmod_dir, ip_dir_name, f"{KMOD_BETA_STAR_NAME_INPUT}{EXT}"))
        new_data = new_data.append(data)
    tfs.write_tfs(join(opt.res_dir, f"{KMOD_BETA_STAR_NAME}{EXT}"), new_data)


if __name__ == '__main__':
    merge_and_copy_kmod_output()
