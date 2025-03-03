"""
Segment by Segment: Constants
-----------------------------

This module provides constants to be used with segment by segment
"""
jobfile: str = "job.create_segment_{}.madx"
logfile: str = "job.create_segment_{}.log"
measurement_madx: str = "measurement_{}.madx"
corrections_madx: str = "corrections.madx"
TWISS_FORWARD: str = "twiss_{}_forward.dat"
TWISS_BACKWARD: str = "twiss_{}_backward.dat"
TWISS_FORWARD_CORRECTED: str = "twiss_{}_forward_corrected.dat"
TWISS_BACKWARD_CORRECTED: str = "twiss_{}_backward_corrected.dat"

FORWARD: str = "FWD"
BACKWARD: str = "BWD"
CORRECTION: str = "COR"
EXPECTED: str = "EXP"
