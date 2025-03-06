"""
LHC
---

Accelerator-Class for the ``LHC`` collider.

Model Creation Keyword Args:
    *--Required--*

    - **beam** *(int)*:

        Beam to use.

        choices: ``(1, 2)``


    - **year** *(str)*:

        Year of the optics (or hllhc1.3).

        choices: ``('2012', '2015', '2016', '2017', '2018', '2022', 'hllhc1.3')``


    *--Optional--*

    - **ats**:

        Marks ATS optics

        action: ``store_true``


    - **dpp** *(float)*:

        Delta p/p to use.

        default: ``0.0``


    - **driven_excitation** *(str)*:

        Denotes driven excitation by AC-dipole (acd) or by ADT (adt)

        choices: ``('acd', 'adt')``


    - **drv_tunes** *(float)*:

        Driven tunes without integer part.


    - **energy** *(float)*:

        Energy in Tev.


    - **model_dir** *(str)*:

        Path to model directory; loads tunes and excitation from model!


    - **modifiers** *(str)*:

        Path to the optics file to use (modifiers file).


    - **nat_tunes** *(float)*:

        Natural tunes without integer part.


    - **xing**:

        If True, x-ing angles will be applied to model

        action: ``store_true``


"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from generic_parser import EntryPoint

from omc3.model.accelerators.accelerator import (
    AccElementTypes,
    Accelerator,
    AcceleratorDefinitionError,
    AccExcitationMode,
)
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

LOGGER = logging_tools.get_logger(__name__)
CURRENT_DIR = Path(__file__).parent
LHC_DIR = CURRENT_DIR / "lhc"


class Lhc(Accelerator):
    """
    Accelerator class for the Large Hadron Collider.
    """
    NAME: str = "lhc"
    LOCAL_REPO_NAME: str = "acc-models-lhc"
    RE_DICT: dict[str, str] = {
        AccElementTypes.BPMS: r"BPM",
        AccElementTypes.MAGNETS: r"M",
        AccElementTypes.ARC_BPMS: r"BPM.*\.0*(1[5-9]|[2-9]\d|[1-9]\d{2,})[RL]",
    }  # bpms > 14 L or R of IP

    LHC_IPS: tuple[str] = ("1", "2", "5", "8")
    NORMAL_IR_BPMS: str = "BPMSW.1{side}{ip}.B{beam}"
    DOROS_IR_BPMS: str = "LHC.BPM.1{side}{ip}.B{beam}_DOROS"
    DEFAULT_CORRECTORS_DIR: Path = LHC_DIR / "correctors"

    @staticmethod
    def get_parameters():
        params = super(Lhc, Lhc).get_parameters()
        params.add_parameter(
            name="beam",
            type=int,
            choices=(1, 2),
            required=True,
            help="Beam to use."
        )
        params.add_parameter(
            name="year",
            type=str,
            help="Year of the optics (or hllhc1.x version).",
        )
        params.add_parameter(
            name="ats",
            action="store_true",
            help="Force use of ATS macros and knobs for years which are not ATS by default.",
        )
        params.add_parameter(
            name="b2_errors",
            type=str,
            help="The B2 error table to load for the best knowledge model.",
        )
        params.add_parameter(
            name="list_b2_errors",
            action="store_true",
            help="Lists all available b2 error tables",
        )
        return params

    def __init__(self, *args, **kwargs):
        parser = EntryPoint(self.get_parameters(), strict=True)
        opt = parser.parse(*args, **kwargs)
        super().__init__(opt)
        self.year = opt.year
        self.ats = opt.ats
        self.b2_errors = opt.b2_errors
        self.list_b2_errors = opt.list_b2_errors
        self.beam = opt.beam
        beam_to_beam_direction = {1: 1, 2: -1}
        self.beam_direction = beam_to_beam_direction[self.beam]

    def verify_object(self) -> None:  # TODO: Maybe more checks?
        """
        Verifies if everything is defined which should be defined.
        Will Raise an ``AcceleratorDefinitionError`` if one of the checks is invalid.
        """
        LOGGER.debug("Accelerator class verification")

        Accelerator.verify_object(self)
        _ = self.beam

        if self.model_dir is None and self.xing is None:
            raise AcceleratorDefinitionError("Crossing on or off not set.")

        # TODO: write more output prints
        LOGGER.debug(
            "... verification passed. \nSome information about the accelerator:"
        )
        LOGGER.debug(f"Class name       {self.__class__.__name__}")
        LOGGER.debug(f"Beam             {self.beam}")
        LOGGER.debug(f"Beam direction   {self.beam_direction}")
        if self.modifiers:
            LOGGER.debug(
                f"Modifiers        {', '.join([str(m) for m in self.modifiers])}"
            )

    @property
    def beam(self) -> int:
        if self._beam is None:
            raise AcceleratorDefinitionError(
                "The accelerator definition is incomplete, beam "
                "has to be specified (--beam option missing?)."
            )
        return self._beam

    @beam.setter
    def beam(self, value) -> None:
        if value not in (1, 2):
            raise AcceleratorDefinitionError("Beam parameter has to be one of (1, 2)")
        self._beam = value

    @staticmethod
    def get_lhc_error_dir() -> Path:
        return LHC_DIR / "systematic_errors"

    def get_variables(self, frm: float = None, to: float = None, classes: Iterable[str] = None):
        corrector_beam_dir = Path(f"correctors_b{self.beam}")
        all_vars_by_class = _load_jsons(
            *self._get_corrector_files(corrector_beam_dir / "beta_correctors.json"),
            *self._get_corrector_files(corrector_beam_dir / "coupling_correctors.json"),
            *self._get_corrector_files("triplet_correctors.json"),
        )
        if classes is not None:
            if isinstance(classes, str):
                # if not checked, lead to each char being treates as a knob. 
                raise TypeError(f"Classes must be an iterable, not a string: {classes}")  

            known_classes = [c for c in classes if c in all_vars_by_class]
            unknown_classes = [c for c  in classes if c not in all_vars_by_class]

            # names without the prefix '-' are simply added to the list
            add_knobs = set(knob for knob in unknown_classes if knob[0] != "-")
            if add_knobs:
                LOGGER.info("The following names are not found as corrector/variable classes and "
                            f"are assumed to be the variable names directly instead:\n{str(add_knobs)}")

            # if the correction variable name is prepended with '-' it is taken out of the ones we use
            remove_knobs = set(knob[1:] for knob in unknown_classes if knob[0] == "-")
            if remove_knobs:
                LOGGER.info(f"The following names will not be used as correctors, as requested:\n{str(remove_knobs)}")
            
            vars = set(_flatten_list(all_vars_by_class[corr_cls] for corr_cls in known_classes))
            vars = list((vars | add_knobs) - remove_knobs)
        
        else:
            vars = list(set(_flatten_list([vars_ for vars_ in all_vars_by_class.values()])))

        # Sort variables by S (nice for comparing different files)
        return self.sort_variables_by_location(vars, frm, to)

    def sort_variables_by_location(self, variables: Iterable[str], frm: float | None = None, to: str | None = None) -> list[str]:
        """ Sorts the variables by location and filters them between `frm` and `to`.
        If `frm` is larger than `to` it loops back around to the start the accelerator.
        This is a useful function for the LHC that's why it is "public"
        but it is not part of the Accelerator-Class Interface.

        Args:
            variables (Iterable): Names of variables to sort
            frm (float): S-position to filter from
            to (float): S-position to filter to
        """
        elems_matrix = tfs.read(self._get_corrector_elems()).sort_values("S")
        if frm is not None and to is not None:
            if frm > to:
                elems_matrix = elems_matrix[(elems_matrix.S >= frm) | (elems_matrix.S <= to)]
            else:
                elems_matrix = elems_matrix[(elems_matrix.S >= frm) & (elems_matrix.S <= to)]
        elif frm is not None:
            elems_matrix = elems_matrix[elems_matrix.S >= frm]
        elif to is not None:
            elems_matrix = elems_matrix[elems_matrix.S <= to]

        # create single list (some entries might contain multiple variable names, comma separated, e.g. "kqx.l2,ktqx2.l2")
        vars_by_position = _remove_dups_keep_order(
            _flatten_list([raw_vars.split(",") for raw_vars in elems_matrix.loc[:, "VARS"]])
        )
        sorted_vars = _list_intersect_keep_order(vars_by_position, variables)

        # Check if no filtering but only sorting was required
        if (frm is None) and (to is None) and (len(sorted_vars) != len(variables)):
            unknown_vars = list(sorted(var for var in variables if var not in sorted_vars))
            LOGGER.debug(f"The following variables do not have a location: {str(unknown_vars)}")
            sorted_vars = sorted_vars + unknown_vars
        return sorted_vars

    def get_ips(self) -> Iterator[tuple[str]]:
        """
        Returns an iterable with this accelerator's IPs.

        Returns:
            An iterator returning `tuples` with:
                (``ip name``, ``left BPM name``, ``right BPM name``)
        """
        for ip in Lhc.LHC_IPS:
            yield (
                f"IP{ip}",
                Lhc.NORMAL_IR_BPMS.format(side="L", ip=ip, beam=self.beam),
                Lhc.NORMAL_IR_BPMS.format(side="R", ip=ip, beam=self.beam),
            )
            yield (
                f"IP{ip}_DOROS",
                Lhc.DOROS_IR_BPMS.format(side="L", ip=ip, beam=self.beam),
                Lhc.DOROS_IR_BPMS.format(side="R", ip=ip, beam=self.beam),
            )

    def log_status(self) -> None:
        LOGGER.info(f"  model dir = {self.model_dir}")
        LOGGER.info(f"Natural Tune X      [{self.nat_tunes[0]:10.3f}]")
        LOGGER.info(f"Natural Tune Y      [{self.nat_tunes[1]:10.3f}]")
        LOGGER.info(
            f"Best Knowledge Model     "
            f"[{'NO' if self.model_best_knowledge is None else 'OK':>10s}]"
        )

        if self.excitation == AccExcitationMode.FREE:
            LOGGER.info(f"Excitation          [{'NO':>10s}]")
            return
        LOGGER.info(
            f"Excitation          "
            f"[{'ACD' if self.excitation == AccExcitationMode.ACD else 'ADT':>10s}]"
        )
        LOGGER.info(f"> Driven Tune X     [{self.drv_tunes[0]:10.3f}]")
        LOGGER.info(f"> Driven Tune Y     [{self.drv_tunes[1]:10.3f}]")


    def get_exciter_bpm(self, plane: str, commonbpms: list[str]) -> tuple[str, str]:
        """ Returns the name of the BPM closest to the exciter (i.e. ADT or AC-Dipole)
        as well as the name of the exciter element. """
        beam = self.beam
        adt = "H.C" if plane == "X" else "V.B"
        l_r = "L" if ((beam == 1) != (plane == "Y")) else "R"
        a_b = "B" if beam == 1 else "A"
        if self.excitation == AccExcitationMode.ACD:
            excitation_name = "AC-Dipole"
            possible_bpms = [f"BPMY{a_b}.6L4.B{beam}", f"BPM.7L4.B{beam}"]
            element = f"MKQA.6L4.B{beam}"
        elif self.excitation == AccExcitationMode.ADT:
            excitation_name = "ADT-AC-Dipole"
            possible_bpms = [f"BPMWA.B5{l_r}4.B{beam}", f"BPMWA.A5{l_r}4.B{beam}"]
            element = f"ADTK{adt}5{l_r}4.B{beam}"
        else:
            return None

        try:
            return _first_one_in(possible_bpms, commonbpms), element
        except KeyError as e:
            raise KeyError(
                f"{excitation_name} BPM (either {possible_bpms[0]} or {possible_bpms[1]}) "
                "not found in the common BPMs. Maybe cleaned?"
            ) from e

    def important_phase_advances(self) -> list[list[str]]:
        if "hl" in self.year.lower(): 
            # skip if HiLumi, TODO: insert phase advances when they are finalised
            return []

        if self.beam == 2:
            return [["MKD.O5R6.B2", "TCTPH.4R1.B2"], ["MKD.O5R6.B2", "TCTPH.4R5.B2"]]
        if self.beam == 1:
            return [["MKD.O5L6.B1", "TCTPH.4L1.B1"], ["MKD.O5L6.B1", "TCTPH.4L5.B1"]]

    def get_synch_BPMs(self, index):
        # expect passing index.to_numpy()
        if self.beam == 1:
            return [i in index for i in self.model.loc["BPMSW.33L2.B1":].index]
        elif self.beam == 2:
            return [i in index for i in self.model.loc["BPMSW.33R8.B2":].index]
    
    def get_accel_file(self, filename: str | Path) -> Path:
        return LHC_DIR / self.year / filename
    
    
    # Private Methods ##############################################################
    def _get_corrector_elems(self) -> Path:
        """ Return the corrector elements file, either from the instance's specific directory,
        if it exists, or the default directory. """
        return self._get_corrector_files(f"corrector_elems_b{self.beam}.tfs")[-1]
    
    def _get_corrector_files(self, file_name: str | Path) -> list[Path]:
        """ Get the corrector files from the default directory AND 
        the instance's specific directory if it exists AND the model directroy if it exists, 
        in that order. 
        See also discussion in https://github.com/pylhc/omc3/pull/458#discussion_r1764829247 .
        """
        # add file from the default directory (i.e. "model/accelerators/lhc/correctors")
        default_file = Lhc.DEFAULT_CORRECTORS_DIR / file_name
        if not default_file.exists():
            msg = (f"Could not find {file_name} in {Lhc.DEFAULT_CORRECTORS_DIR}."
                  "Something went wrong with the variables getting logic.")
            raise FileNotFoundError(msg)
        
        LOGGER.debug(
            f"Default corrector file {file_name} found in {default_file.parent}."
        )
        corrector_files = [default_file]

        # add file from the accelerator directory (e.g. "model/accelerators/lhc/2024/correctors")
        accel_dir_file = self.get_accel_file(Path(Lhc.DEFAULT_CORRECTORS_DIR.name) / file_name)
        if accel_dir_file.exists():
            LOGGER.debug(
                f"Corrector file {file_name} found in {accel_dir_file.parent}. "
                "Contents will take precedence over defaults."
            )
            corrector_files.append(accel_dir_file)

        # add file from the model directory (without "correctors" and subfolders) - bit of a hidden feature
        if self.model_dir is not None:
            model_dir_file = Path(self.model_dir) / Path(file_name).name
            if model_dir_file.exists():
                LOGGER.info(
                    f"Corrector file {file_name} found in {model_dir_file.parent}. "
                    "Contents will take precedence over omc3-given defaults."
                )
                corrector_files.append(model_dir_file)
        
        return corrector_files


# General functions ##########################################################

def _load_jsons(*files) -> dict:
    full_dict = {}
    for json_file in files:
        with open(json_file, "r") as json_data:
            full_dict.update(json.load(json_data))
    return full_dict


def _flatten_list(my_list: Iterable) -> list:
    return [item for sublist in my_list for item in sublist]


def _remove_dups_keep_order(my_list: list) -> list:
    return list(dict.fromkeys(my_list))


def _list_intersect_keep_order(primary_list: Iterable, secondary_list: Iterable) -> list:
    return [elem for elem in primary_list if elem in secondary_list]


def _first_one_in(bpms_to_find: Sequence[str], bpms: Sequence[str]):
    found_bpms = [bpm for bpm in bpms_to_find if bpm in bpms]
    if len(found_bpms):
        return list(bpms).index(found_bpms[0]), found_bpms[0]
    raise KeyError
