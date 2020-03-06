


def _set_ylabel(ax, default, y_label, y_plane, chromatic):
    """ Tries to set a mapped y label, otherwise the default """
    try:
        annotations.set_yaxis_label(_map_proper_name(y_label),
                                    y_plane, ax, chromcoup=chromatic)
    except (KeyError, ValueError):
        ax.set_ylabel(default)


def _map_proper_name(name):
    """ Maps to a name understood by plotstyle. """
    return {
        "BET": "beta",
        "BB": "betabeat",
        "D": "dispersion",
        "ND": "norm_dispersion",
        "MU": "phase",
        "X": "co",
        "Y": "co",
        "PHASE": "phase",
        "I": "imag",
        "R": "real",
    }[name.upper()]


def _get_ir_positions(all_data, x_cols):
    """ Check if x is position around the ring and return ir positions if possible """
    ir_pos = None
    x_is_pos = all([xc == "S" for xc in x_cols])
    if x_is_pos:
        ir_pos = _find_ir_pos(all_data)
    return ir_pos, x_is_pos


def _get_auto_scale(y_val, scaling):
    """ Find the y-limits so that scaling% of the points are visible """
    y_sorted = sorted(y_val)
    n_points = len(y_val)
    y_min = y_sorted[int(((1 - scaling/100.) / 2.) * n_points)]
    y_max = y_sorted[int(((1 + scaling/100.) / 2.) * n_points)]
    return y_min, y_max


def _find_ir_pos(all_data):
    """ Return the middle positions of the interaction regions """
    ip_names = ["IP" + str(i) for i in range(1, 9)]
    for data in all_data:
        try:
            ip_pos = data.loc[ip_names, 'S'].values
        except KeyError:
            try:
                # loading failed, use defaults
                return IR_POS_DEFAULT[data.SEQUENCE]
                # return {}
            except AttributeError:
                # continue looking
                pass
        else:
            return dict(zip(ip_names, ip_pos))

    # did not find ips or defaults
    return {}


    if auto_scale:
        current_y_lims = _get_auto_scale(y_val, auto_scale)
        if y_lims is None:
            y_lims = current_y_lims
        else:
            y_lims = [min(y_lims[0], current_y_lims[0]),
                      max(y_lims[1], current_y_lims[1])]
        if last_line:
            ax.set_ylim(*y_lims)

    # things to do only once
    if last_line:
        # setting the y_label
        if y_label is None:
            _set_ylabel(ax, y_col, y_label_from_col, y_plane, chromatic)
        else:
            y_label_from_label = ""
            if y_label:
                y_label_from_label, y_plane, _, _, chromatic = _get_names_and_columns(
                    idx_plot, xy, y_label, "")
            if xy:
                y_label = f"{y_label:s} {y_plane:s}"
            _set_ylabel(ax, y_label, y_label_from_label, y_plane, chromatic)

        # setting x limits
        if x_is_position:
            with suppress(AttributeError, ValueError):
                post_processing.set_xlimits(data.SEQUENCE, ax)

        # setting visibility, ir-markers and label
        if xy and idx_plot == 0:
            ax.axes.get_xaxis().set_visible(False)
            if x_is_position and ir_positions:
                annotations.show_ir(ir_positions, ax, mode='lines')
        else:
            if x_is_position:
                annotations.set_xaxis_label(ax)
                if ir_positions:
                    annotations.show_ir(ir_positions, ax, mode='outside')
