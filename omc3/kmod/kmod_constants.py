


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
