"""
Write MAD-X Macros
------------------

Write out madx scripts for the tracking macros.

**Arguments:**

*--Required--*

- **outputdir**:

    Directory where the observation_points.def will be put.

- **twissfile**:

    Path to twissfile with observationspoint in the NAME column.
"""

import tfs
from pathlib import Path
from omc3.model.constants import OBS_POINTS
from generic_parser.entrypoint_parser import entrypoint, EntryPointParameters

def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="twissfile",
        required=True,
        help="Path to twissfile with observationspoint in the NAME column."
        )
    params.add_parameter(
        name="outputdir",
        required=True,
        help=f"Directory where the {OBS_POINTS} will be put."
        )
    return params

@entrypoint(get_params(), strict=True)
def read_twiss_and_return_obs(opt):
    tracking_macros(tfs.read(opt['twissfile'], index='NAME').index.tolist(), Path(opt['outputdir']))

def _call(path_to_call):
    return f"call, file = '{path_to_call}';\n"


def define_observation_points_macros(list_of_bpms):
    macro = ""
    for tracker, prefix in (("ptc", "ptc_"), ("madx", "")):
        macro += f"define_{tracker}_observation_points(): macro = {{\n"
        macro += "".join([f"    {prefix}observe, place='{bpm}';\n" for bpm in list_of_bpms])
        macro += "};\n"
    return macro


def tracking_macros(list_of_bpms, outdir):
    obs_macro_file = outdir / OBS_POINTS
    with open(obs_macro_file, 'w') as obs_script:
        obs_script.write(define_observation_points_macros(list_of_bpms))
    track_macros = _call(obs_macro_file)
    track_macros += """
    /*
    * Performs a single particle tracking of the active sequence.
    * @param hkick: Horizontal kick magnitude in meters.
    * @param vkick: Vertical kick magnitude in meters.
    * @param num_turns: Number of turns to simulate.
    * @param output_path: Path to the output file, will add "one" at the end of the file name.
    */
    ptc_track_single_particle(hkick, vkick, num_turns, output_path): macro = {
    PTC_CREATE_UNIVERSE;
    PTC_CREATE_LAYOUT, model = 3, method = 6, nst = 10;
    PTC_ALIGN;
    EXEC, define_ptc_observation_points();
    PTC_START, x = hkick, y = vkick; ! kick units meters
    PTC_TRACK, deltap = 0.0, icase = 5, turns = num_turns, ELEMENT_BY_ELEMENT, dump, onetable, file = output_path;
    PTC_TRACK_END;
    PTC_END;
    }
    
    /*
    * Performs a single particle tracking of the active sequence using MAD-X track.
    * @param start_x: Particle horizontal start position.
    * @param start_y: Particle vertical start position.
    * @param start_px: Particle horizontal start momentum.
    * @param start_py: Particle vertical start momentum.
    * @param num_turns: Number of turns to simulate.
    * @param output_path: Path to the output file, will add "one" at the end of the file name.
    */
    madx_track_single_particle(start_x, start_y, start_px, start_py, num_turns, output_path): macro = {
    TRACK, DELTAP=0.0, ONETABLE, DUMP, FILE=output_path;
    START, X=start_x, PX=start_px, Y=start_y, PY=start_py;
    EXEC, define_madx_observation_points();
    RUN, TURNS=num_turns;
    ENDTRACK;
    }
    """
    return track_macros


if __name__ == "__main__":
    read_twiss_and_return_obs()
