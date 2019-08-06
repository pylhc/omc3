from utils import logging_tools
from kmod import kmod_utils


LOG = logging_tools.get_logger(__name__)
DEFAULTS_IP = {
    "cminus": 1E-3,
    "misalignment": 0.006,
    "errorK": 0.001,
    "errorL": 0.001,
}

DEFAULTS_CIRCUITS = {
    "cminus": 1E-3,
    "misalignment": 0.001,
    "errorK": 0.001,
    "errorL": 0.001,
}

MAGNETS_IP = {
    "IP1": ['MQXA.1L1', 'MQXA.1R1'],
    "IP2": ['MQXA.1L2', 'MQXA.1R2'],
    "IP5": ['MQXA.1L5', 'MQXA.1R5'],
    "IP8": ['MQXA.1L8', 'MQXA.1R8']
}


class KmodInput():
    """
    Class for holding all input variables required for Kmodulation analysis
    """

    def __init__(self):

        self.working_directory = None
        self.beam = None
        self.ip = None
        self.magnet1 = None
        self.magnet2 = None
        self.circuits = None
        self.circuit1 = None
        self.circuit2 = None
        self.betastar_and_waist = None
        self.betastar_x = None
        self.betastar_y = None
        self.waist_x = None
        self.waist_y = None

        self.cminus = None
        self.misalignment = None
        self.errorK = None
        self.tune_uncertainty = None
        self.instruments = None
        self.log = None
        self.simulation = None
        self.no_autoclean = None
        self.no_sigdigit = None
        self.no_plots = None
        self.betastar_required = False
        self.instruments_found = []

    def set_params_from_parser(self, options):

        self.working_directory = options.work_dir
        self.beam = options.beam.upper()
        self.ip = options.ip
        self.circuits = options.circuits
        self.betastar_and_waist = options.betastar

        self.tune_uncertainty = options.tunemeasuncertainty
        self.instruments = list(map(str.upper, options.instruments.split(",")))

        self.log = options.log
        self.simulation = options.simulation
        self.no_autoclean = options.no_a_clean
        self.no_sigdigit = options.no_sig_dig
        self.no_plots = options.no_plots

        self.set_error(options, "cminus")
        self.set_error(options, "errorK")
        self.set_error(options, "errorL")
        self.set_error(options, "misalignment")

        self.set_betastar_and_waist(options)
        self.set_magnets(options)

    def set_betastar_required(self):
        self.betastar_required = True

    def set_instruments_found(self, found):
        self.instruments_found.append(found)

    def set_instrument_position(self, instrument, positions):
        setattr(self, instrument, positions)

    def set_betastar_and_waist(self, options):

        bs = options.betastar
        if len(bs) == 2:
            self.betastar_x, self.betastar_y, self.waist_x, self.waist_y = map(float, (bs[0], bs[0], bs[1], bs[1]))
        elif len(bs) == 3:
            self.betastar_x, self.betastar_y, self.waist_x, self.waist_y = map(float, (bs[0], bs[1], bs[2], bs[2]))
        elif len(bs) == 4:
            self.betastar_x, self.betastar_y, self.waist_x, self.waist_y = map(float, (bs[0], bs[1], bs[2], bs[3]))

    def set_magnets(self, options):

        if options.ip is not None and options.circuits is None:
            LOG.info('IP trim analysis')
            self.magnet1, self.magnet2 = MAGNETS_IP[options.ip.upper()]

        elif options.ip is None and options.circuits is not None:
            LOG.info('Individual magnets analysis')
            self.circuit1, self.circuit2 = options.circuits
            self.magnet1 = kmod_utils.find_magnet(self.beam, self.circuit1)
            self.magnet2 = kmod_utils.find_magnet(self.beam, self.circuit2)
        elif options.ip is None and options.circuits is None:
            raise SystemError('No IP or circuits specfied, stopping analysis')
        else:
            raise SystemError('Both IP and circuits specfied, choose only one, stopping analysis')

    def return_guess(self, plane):

        if plane == 'X':
            return [self.betastar_x, self.waist_x]
        elif plane == 'Y':
            return [self.betastar_y, self.waist_y]

    def set_error(self, options, error):
        if options[error] is None:

            if options.ip is not None:
                setattr(self, error, DEFAULTS_IP[error])

            elif options.circuits is not None:
                setattr(self, error, DEFAULTS_CIRCUITS[error])
        else:
            setattr(self, error, getattr(options, error))


def get_input(opt):

    arguments = KmodInput()

    arguments.set_params_from_parser(opt)

    return arguments
