import pathlib
import tfs
from generic_parser import entrypoint, EntryPointParameters
import re
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


def loader_params():
    params = EntryPointParameters()
    params.add_parameter(name="kmod_dirs", type=pathlib.Path,
                         nargs='+', required=True,
                         help="Path to kmod directories with stored KMOD "
                              "measurement files")
    params.add_parameter(name="res_dir", type=pathlib.Path, required=True,
                         help="Output directory where to write the result tfs")
    return params


@entrypoint(loader_params(), strict=True)
def merge_and_copy_kmod_output(opt):
    pattern = re.compile(".*ip[0-9]B[1-2]")

    # Get the directories we need where the tfs are stored
    ip_dir_names = [d for kmod in opt.kmod_dirs
                    for d in kmod.glob('**/*')
                    if pattern.match(d.name) and d.is_dir()]

    # Combine the data into one tfs
    new_data = tfs.TfsDataFrame()
    for ip_dir_name in ip_dir_names:
        path = ip_dir_name / 'results.tfs'
        data = tfs.read_tfs(path)

        new_data = new_data.append(data, ignore_index=True)

    # and write the resulting tfs
    tfs.write(opt.res_dir / 'results.tfs', new_data)


if __name__ == '__main__':
    merge_and_copy_kmod_output()
