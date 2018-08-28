import time
import numpy as np

import sdds.handler
from sdds import sdds_writer, handler
from tbt import turn_by_turn_reader as tbt_reader


def write_tbt_file(names, matrix, outfile):
    """Writes given turn by turn data and names into oufile in SDDS format.

    Arguments:
        names: Numpy array of BPM names
        matrix: 4D Numpy array [quantity, BPM, particle/bunch No., turn No.]
            quantities in order [x, y]
        outfile: Path to the output file.
    """
    _, _, nbunches, nturns = matrix.shape
    sdds_file = handler.SddsFile()
    for param in _get_all_params(nbunches, nturns):
        sdds_file._parameters[param.name] = param
    for array in _get_all_arrays(names, matrix):
        sdds_file._arrays[array.name] = array
    sdds.handler.write_sdds(sdds_file, outfile)


def _get_all_params(nbunches, nturns):
    stamp = int(time.time()) * 1e9
    return (_get_param(tbt_reader.TIMESTAMP_NAME, "double", stamp),
            _get_param(tbt_reader.NUM_BUNCHES_NAME, "long", nbunches),
            _get_param(tbt_reader.NUM_TURNS_NAME, "long", nturns))


def _get_param(name, type_, value):
    param = handler.SddsParameter(name,
                                  type_, type_,
                                  None, None, None, None, None)
    param.value = value
    return param


def _get_all_arrays(names, matrix):
    _, nbpms, nbunches, nturns = matrix.shape
    mat_x, mat_y = matrix
    samples_x = np.ravel(mat_x)
    samples_y = np.ravel(mat_y)
    bids_x = np.zeros(len(samples_x)).reshape(nbpms, nbunches, nturns)
    bids_y = np.zeros(len(samples_y)).reshape(nbpms, nbunches, nturns)
    for bid in range(nbunches):
        bids_x[:, bid, :] = bid
        bids_y[:, bid, :] = bid
    bids_x = np.ravel(bids_x)
    bids_y = np.ravel(bids_y)
    failed_x = np.zeros(len(samples_x))
    failed_y = np.zeros(len(samples_y))
    return (
        _get_array(tbt_reader.ALL_HOR_POSITIONS_NAME, "float", samples_x),
        _get_array(tbt_reader.ALL_VER_POSITIONS_NAME, "float", samples_y),
        _get_array(tbt_reader.BPM_NAMES_NAME, "string", names),
        _get_array(tbt_reader.HOR_BUNCH_ID_NAME, "long", bids_x),
        _get_array(tbt_reader.HOR_FAILED_BUNCH_ID_NAME, "long", failed_x),
        _get_array(tbt_reader.VER_BUNCH_ID_NAME, "long", bids_y),
        _get_array(tbt_reader.VER_FAILED_BUNCH_ID_NAME, "long", failed_y),
    )


def _get_array(name, type_, values):
    array = handler.SddsArray(name,
                              type_, type_,
                              None, None, None, None, None, None, None)
    if type_ != "string":
        array.values = values.astype(sdds.handler.TYPES[type_])
    else:
        array.values = values
    return array
