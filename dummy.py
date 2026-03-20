def func():
    elements, error_method = get_elements_with_errors(meas_input, plane)
    beta_df = _get_filtered_model_df(meas_input, phase, plane)
    bk_model = _get_filtered_model_df(meas_input, phase, plane, best=True)
    tune, mdltune = meas_and_mdl_tunes

    m = int(n_bpms / 2)  # half window: probed BPM has m neighbors on each side
    loc_range = np.arange(-m, m + 1)  # relative indices [-m, ..., 0, ..., m] 0 is the probed BPM
