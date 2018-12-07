
import os
new_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir))
# change at later point to point to beta beat src path
SEQUENCES_PATH = os.path.join( new_path,'omc3', 'kmod', 'sequences' )

def get_tune_col(plane):
    return "TUNE{:s}".format( plane.upper() )

def get_tune_err_col(plane):
    return "{:s}_ERR".format( get_tune_col(plane) )

def get_cleaned_col(plane):
    return "CLEANED{:s}".format( plane.upper() )    

def get_k_col():
    return "K" 

def get_betastar_col(plane):
    return "BETASTAR{:s}".format( plane.upper() )    

def get_betastar_err_col(plane):
    return "{:s}_ERR".format( get_betastar_col(plane) )        

def get_waist_col(plane):
    return "WAIST{:s}".format( plane.upper() )    

def get_waist_err_col(plane):
    return "{:s}_ERR".format( get_waist_col(plane) )            

def get_betawaist_col(plane):
    return "BETAWAIST{:s}".format( plane.upper() )    

def get_betawaist_err_col(plane):
    return "{:s}_ERR".format( get_betawaist_col(plane) )

def get_sequence_filename(beam):
    return os.path.join( SEQUENCES_PATH , "twiss_lhc{:s}.dat".format( beam.lower() ))
