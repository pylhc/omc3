"""
Segment by Segment: Constants
-----------------------------

This module provides constants to be used with segment by segment
"""
jobfile = "job.create_segment_{}.madx"
logfile = "job.create_segment_{}.log"
measurement_madx = "measurement_{}.madx"
corrections_madx = "corrections.madx"
TWISS_FORWARD = "twiss_{}_forward.dat"
TWISS_BACKWARD = "twiss_{}_backward.dat"
TWISS_FORWARD_CORRECTED = "twiss_{}_forward_corrected.dat"
TWISS_BACKWARD_CORRECTED = "twiss_{}_backward_corrected.dat"

FORWARD: str = "FWD"
BACKWARD: str = "BWD"
CORRECTION: str = "COR"
EXPECTED: str = "EXP"
