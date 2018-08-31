import os
import sys

import tfs
from kmod.gui2kmod import parse_args, returnmagnetname, returncircuitname, run_analysis_simplex
from kmod.timber_output_reader import merge_data

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def _main():
    options = parse_args()

    IP_default_err = {'cminus': 1e-3, 'misalign': 0.006, 'ek': 0.001}
    Circuit_default_err = {'cminus': 1e-3, 'misalign': 0.001, 'ek': 0.001}

    if "cminus" not in options:
        if options.ip is not None:
            cminus = IP_default_err['cminus']
        else:
            cminus = Circuit_default_err['cminus']
    else:
        cminus = options.cminus

    if "ek" not in options:
        if options.ip is not None:
            ek = IP_default_err['ek']
        else:
            ek = Circuit_default_err['ek']
    else:
        ek = options.ek

    if "misalign" not in options:
        if options.ip is not None:
            misalign = IP_default_err['misalign']
        else:
            misalign = Circuit_default_err['misalign']
    else:
        misalign = options.misalign

    working_directory = options.work_dir
    beam = options.beam.upper()

    instruments = options.instruments.split(',')
    instruments = [x.upper() for x in instruments]

    bs = options.betastar
    bs = bs.split(",")

    if len(bs)==2:
        hor_bstar = bs[0]
        vert_bstar = bs[0]
        waist = bs[1]
    if len(bs)==3:
        hor_bstar = bs[0]
        vert_bstar = bs[1]
        waist = bs[2]


    auto_clean = options.a_clean
    command = open(working_directory + '/command.run', 'a')
    command.write(str(' '.join(sys.argv)))
    command.write('\n')
    command.close()

    if beam == 'B1':
        twissfile = os.path.join(CURRENT_PATH, "sequences", "twiss_lhcb1.dat")
    else:
        twissfile = os.path.join(CURRENT_PATH, "sequences", "twiss_lhcb2.dat")
    twiss = tfs.read(twissfile)

    if options.ip is not None:
        if options.ip == 'ip1' or options.ip == 'IP1':
            magnet1, magnet2 = 'MQXA.1L1', 'MQXA.1R1'
        elif options.ip == 'ip5' or options.ip == 'IP5':
            magnet1, magnet2 = 'MQXA.1L5', 'MQXA.1R5'
        elif options.ip == 'ip8' or options.ip == 'IP8':
            magnet1, magnet2 = 'MQXA.1L8', 'MQXA.1R8'
        elif options.ip == 'ip2' or options.ip == 'IP2':
            magnet1, magnet2 = 'MQXA.1L2', 'MQXA.1R2'

    else:
        circuits = options.magnets
        circuits = circuits.split(",")
        circuit1, circuit2 = circuits
        magnet1 = returnmagnetname(circuit1, beam, twiss)
        magnet2 = returnmagnetname(circuit2, beam, twiss)

    path = os.path.join(working_directory, magnet1 + '.' + magnet2 + '.' + beam)

    if not os.path.exists(path):
        os.makedirs(path)
    if options.log:
        logdata = open(path + '/data.log', 'w')

    merge_data(working_directory, magnet1, returncircuitname(magnet1, beam), magnet2, returncircuitname(magnet2, beam),
               beam, options.ip, options.tunemeasuncertainty)

    run_analysis_simplex(path, beam, magnet1, magnet2, hor_bstar, vert_bstar, waist, working_directory, instruments, ek,
                         misalign, cminus, twiss, options.log, logdata, auto_clean)

    logdata.close()


if __name__ == '__main__':
    _main()
