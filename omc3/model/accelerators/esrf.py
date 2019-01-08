import pandas as pd

from model.accelerators.accelerator import Accelerator


class Esrf(Accelerator):

    NAME = "esrf"

    @classmethod
    def get_element_types_mask(cls, list_of_elements, types):
        """
        Return boolean mask for elements in list_of_elements that belong
        to any of the specified types.
        Needs to handle: "bpm", "magnet", "arc_bpm", "amp_bpm"
        TODO: implement "magnet"

        arc_bpms are the ones with high beta, which are:
            bpms 1-5 in even cells.
            bpms 3-7 in odd cells.


        Args:
            list_of_elements: List of elements
            types: Kinds of elements to look for

        Returns:
            Boolean array of elements of specified kinds.

        """
        re_dict = {
            "bpm": r"BPM",
            "arc_bpm": r"BPM\.(\d*[02468]\.[1-5]|\d*[13579]\.[3-7])",
        }

        unknown_elements = [ty for ty in types if ty not in re_dict]
        if len(unknown_elements):
            raise TypeError("Unknown element(s): '{:s}'".format(str(unknown_elements)))

        series = pd.Series(list_of_elements)

        mask = series.str.match(re_dict[types[0]], case=False)
        for ty in types[1:]:
            mask = mask | series.str.match(re_dict[ty], case=False)
        return mask.values

    @classmethod
    def get_class(cls):
        new_class = cls
        return new_class
