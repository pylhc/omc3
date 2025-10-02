from pathlib import Path

from omc3 import madx_wrapper
from omc3.optics_measurements.constants import PHASE_ADV

# These paths means that the script can be run from any directory
current_dir = Path(__file__).parent
omc3_dir = Path(__file__).parents[4]

def run_dpp(offset, beam):
    """
    Run a twiss on a 2018 LHC model with a given dpp offset. Then, correct and match before
    writing the final twiss to a file, which only contains select BPMs, and the phase advances and s.
    This is used by the test_lhc_global_correct_dpp to verify that the global correction can
    calculate the dpp offset input in the fake measurement.
    """
    output_file = current_dir / f"twiss_dpp_{offset:.1e}_B{beam}.dat"

    Qx = 62.28001034  # noqa: N806
    Qy = 60.31000965  # noqa: N806
    script = f"""
    option, -echo;
    call, file = '{omc3_dir}/omc3/model/madx_macros/general.macros.madx';
    call, file = '{omc3_dir}/omc3/model/madx_macros/lhc.macros.madx';
    call, file = '{omc3_dir}/omc3/model/accelerators/lhc/2018/main.seq';
    option, echo;
    exec, cycle_sequences();
    exec, define_nominal_beams();

    call, file = '{omc3_dir}/tests/inputs/models/2018_inj_b{beam}_11m/opticsfile.1'; !@modifier

    select, flag = twiss, pattern = 'BPM.*B[12]', column = name, s, {PHASE_ADV}x, {PHASE_ADV}y;
    use, sequence = LHCB{beam};

    ! Match the tunes initially
    match, deltap = {offset};
    vary, name=dQx.b{beam};
    vary, name=dQy.b{beam};
    constraint, range = '#E', mux = {Qx}, muy = {Qy};
    lmdif, tolerance = 1.0e-10;
    endmatch;

    ! Run a twiss with the offset to get orbit
    twiss, deltap = {offset};

    ! Correct the orbit
    correct, mode = svd;

    ! Match the tunes back to normal
    match, deltap = {offset};
    vary, name=dQx.b{beam};
    vary, name=dQy.b{beam};
    constraint, range = '#E', mux = {Qx}, muy = {Qy};
    lmdif, tolerance = 1.0e-10;
    endmatch;

    ! Run the final twiss to get the off-orbit response
    twiss, deltap = {offset}, file = '{output_file}';
    """
    madx_wrapper.run_string(script)

for deltap in [2.5e-4, -1e-4]:
    for beam in [1, 2]:
        run_dpp(deltap, beam)
