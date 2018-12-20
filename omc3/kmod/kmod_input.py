import argparse
from utils import logging_tools
from kmod import kmod_utils

LOG = logging_tools.get_logger(__name__)
DEFAULTS_IP = {
    "cminus": 1E-3 ,
    "misalignment": 0.006 ,
    "errorK": 0.001 
}

DEFAULTS_CIRCUITS = {
    "cminus": 1E-3 ,
    "misalignment": 0.001 ,
    "errorK": 0.001 
}

MAGNETS_IP = {
    "IP1": ['MQXA.1L1', 'MQXA.1R1'] ,
    "IP2": ['MQXA.1L2', 'MQXA.1R2'] ,
    "IP5": ['MQXA.1L5', 'MQXA.1R5'] ,
    "IP8": ['MQXA.1L8', 'MQXA.1R8'] 
}

def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--betastar_and_waist',
                        help='Estimated beta star of measurements and waist shift',
                        action='store', type=str, dest='betastar', required=True)
    parser.add_argument('--working_directory',
                        help='path to working directory with stored KMOD measurement files',
                        action='store', type=str, dest='work_dir', required=True)
    parser.add_argument('--beam',
                    help='define beam used: b1 or b2',
                    action='store', type=str, dest='beam', choices=['b1', 'b2', 'B1', 'B2'], required=True)

    parser.add_argument('--cminus',
                        help='C Minus',
                        action='store', type=float, dest='cminus', default=argparse.SUPPRESS)
    parser.add_argument('--misalignment',
                        help='misalignment of the modulated quadrupoles in m',
                        action='store', type=float, dest='misalignment', default=argparse.SUPPRESS)
    parser.add_argument('--errorK',
                        help='error in K of the modulated quadrupoles, unit m^-2',
                        action='store', type=float, dest='errorK', default=argparse.SUPPRESS)
    
    parser.add_argument('--tune_uncertainty',
                        help='tune measurement uncertainty',
                        action='store', type=float, dest='tunemeasuncertainty', default=2.5e-5)
    parser.add_argument('--instruments',
                        help='define instruments (use keywords from twiss) at which beta should be calculated , separated by comma, e.g. MONITOR,RBEND,INSTRUMENT,TKICKER',
                        action='store', type=str, dest='instruments', default='MONITOR,SBEND,TKICKER,INSTRUMENT')

    parser.add_argument('--simulation',
                        help='flag for enabling simulation mode',
                        action='store_true', dest='simulation')    
    parser.add_argument('--log',
                        help='flag for creating a log file',
                        action='store_true', dest='log')
    parser.add_argument('--no_autoclean',
                        help='flag for manually cleaning data',
                        action='store_true', dest='a_clean')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument( '--circuit',
                       help='circuit names of the modulated quadrupoles',
                       action='store', type=str, dest='circuits')
    group.add_argument( '--interaction_point',
                       help='define interaction point',
                       action='store', type=str, dest='ip', choices=['ip1', 'ip2', 'ip5', 'ip8', 'IP1', 'IP2', 'IP5', 'IP8'])

    options = parser.parse_args()

    return options

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
        self.betastar_required=False

    def set_params_from_parser(self, options):

        self.working_directory = options.work_dir
        self.beam = options.beam.upper()
        self.ip = options.ip
        self.circuits = options.circuits
        self.betastar_and_waist=options.betastar

        self.tune_uncertainty = options.tunemeasuncertainty
        self.instruments = list(map( str.upper ,options.instruments.split(",")  ))

        self.log = options.log
        self.simulation = options.simulation
        self.no_autoclean = options.a_clean

        self.set_error(options, "cminus")
        self.set_error(options, "errorK")
        self.set_error(options, "misalignment")

        self.set_betastar_and_waist( options )
        self.set_magnets( options )

    def set_betastar_required(self):
        self.betastar_required=True

    def set_instrument_position( self, instrument, positions ):
        setattr(self, instrument, positions)

    def set_betastar_and_waist(self, options):
        
        bs = options.betastar.split(",")
        if len(bs) == 2:
            self.betastar_x, self.betastar_y, self.waist_x, self.waist_y = map( float, (bs[0], bs[0], bs[1], bs[1]))
        elif len(bs) == 3:
            self.betastar_x, self.betastar_y, self.waist_x, self.waist_y = map( float, (bs[0], bs[1], bs[2], bs[2]))
        elif len(bs) == 4:
            self.betastar_x, self.betastar_y, self.waist_x, self.waist_y = map( float, (bs[0], bs[1], bs[2], bs[3]))

    def set_magnets(self, options):

        if options.ip is not None:
            LOG.info('IP trim analysis')
            self.magnet1, self.magnet2 = MAGNETS_IP[options.ip.upper()]
        # TODO fix this function for IR4 modulation
        else:
            LOG.info('Indiv magnets analysis')
            self.circuit1, self.circuit2 =  options.circuits.split(',')
            self.magnet1 = kmod_utils.find_magnet(self.beam, self.circuit1)
            self.magnet2 = kmod_utils.find_magnet(self.beam, self.circuit2) 
    
    def return_guess(self, plane):

        if plane == 'X':
            return [self.betastar_x, self.waist_x]
        elif plane == 'Y':
            return [self.betastar_y, self.waist_y]

    def set_error( self, options, error ):
        if error not in options:

            if options.ip != None:
                setattr(self, error, DEFAULTS_IP[error])  
            
            elif options.circuits != None:
                setattr(self, error, DEFAULTS_CIRCUITS[error])  
        else:
            setattr(self, error, getattr(options, error))   


def get_input():
    
    arguments = KmodInput()
    options = _parse_args()

    arguments.set_params_from_parser( options )     

    return arguments
