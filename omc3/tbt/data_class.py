from datetime import datetime


class TbtData(object):
    """
    Object holding a representation of a Turn-by-Turn Data

    """
    def __init__(self, matrices, date, bunch_ids, nturns):
        self.matrices = matrices  # list per bunch containing dict per plane of DataFrames
        self.date = date if date is not None else datetime.now()
        self.nbunches = len(bunch_ids)
        self.nturns = nturns
        self.bunch_ids = bunch_ids
