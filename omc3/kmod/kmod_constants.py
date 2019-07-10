
import os
new_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir))

SEQUENCES_PATH = os.path.join(new_path, 'omc3', 'kmod', 'sequences')


def get_tune_col(plane):
    return f"TUNE{plane.upper()}"


def get_tune_err_col(plane):
    return f"{get_tune_col(plane)}_ERR"


def get_cleaned_col(plane):
    return f"CLEANED_{plane.upper()}"


def get_k_col():
    return "K"


def get_beta_col(plane):
    return f"BETA{plane.upper()}"


def get_beta_err_col(plane):
    return f"{get_beta_col(plane)}_ERR"


def get_betastar_col(plane):
    return f"BETASTAR{plane.upper()}"


def get_betastar_err_col(plane):
    return f"{get_betastar_col(plane)}_ERR"


def get_phase_adv_col(plane):
    return f"PHASEADV{plane.upper()}"


def get_phase_adv_err_col(plane):
    return f"{get_phase_adv_col(plane)}_ERR"


def get_waist_col(plane):
    return f"WAIST{plane.upper()}"


def get_waist_err_col(plane):
    return f"{get_waist_col(plane)}_ERR"


def get_betawaist_col(plane):
    return f"BETAWAIST{plane.upper()}"


def get_betawaist_err_col(plane):
    return f"{get_betawaist_col(plane)}_ERR"


def get_av_beta_col(plane):
    return f"AVERAGEBETA{plane.upper()}"


def get_av_beta_err_col(plane):
    return f"{get_av_beta_col(plane)}_ERR"


def get_sequence_filename(beam):
    return os.path.join(SEQUENCES_PATH, f"twiss_lhc{beam.lower()}.dat")


def get_working_directory(kmod_input_params):
    return os.path.join(kmod_input_params.working_directory, get_label(kmod_input_params))


def get_label(kmod_input_params):
    if kmod_input_params.ip is not None:
        return f'{kmod_input_params.ip}{kmod_input_params.beam}'

    elif kmod_input_params.circuits is not None:
        return f'{kmod_input_params.magnet1}-{kmod_input_params.magnet2}'
