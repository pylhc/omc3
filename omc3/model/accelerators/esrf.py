"""
ESRF
-------------------
"""
from model.accelerators.accelerator import Accelerator


class Esrf(Accelerator):
    NAME = "esrf"
    RE_DICT = {"bpm": r"BPM", "magnet": r".*",
               "arc_bpm": r"BPM\.(\d*[02468]\.[1-5]|\d*[13579]\.[3-7])",
               }  # bpms 1-5 in even cells and bpms 3-7 in odd cells.
