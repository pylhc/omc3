from omc3.optics_measurements.constants import (
    NORM_DISP_NAME, PHASE_NAME, AMP_BETA_NAME, BETA_NAME, DISPERSION_NAME,
    ORBIT_NAME, CHROM_BETA_NAME, SPECIAL_PHASE_NAME, TOTAL_PHASE_NAME,
    KMOD_BETA_NAME, KMOD_BETASTAR_NAME, F1001_NAME, F1010_NAME, EXT
)
from tfs.collection import TfsCollection, Tfs


class OpticsMeasurement(TfsCollection):
    """Class to hold and load the measurements from GetLLM.

    The class will try to load the _free file, then the _free2 and then the
    normal file, if none of them if present an IOError will be raised.

    Arguments:
        directory (Path): The path to the measurement directory, usually an
                          optics-measurement output directory.
    """
    beta = Tfs(BETA_NAME)
    chrom_beta = Tfs(CHROM_BETA_NAME)
    amp_beta = Tfs(AMP_BETA_NAME)
    kmod_beta = Tfs(KMOD_BETA_NAME)
    kmod_betastar = Tfs(KMOD_BETASTAR_NAME, two_planes=False)
    phase = Tfs(PHASE_NAME)
    special_phase = Tfs(SPECIAL_PHASE_NAME)
    phasetot = Tfs(TOTAL_PHASE_NAME)
    disp = Tfs(DISPERSION_NAME)
    orbit = Tfs(ORBIT_NAME)
    f1001 = Tfs(F1001_NAME, two_planes=False)
    f1010 = Tfs(F1010_NAME, two_planes=False)
    norm_disp = Tfs(NORM_DISP_NAME, two_planes=False)

    @staticmethod
    def _get_optics_filename(prefix, plane):
        return f"{prefix}{plane.lower()}{EXT}"

    def get_filename(self, prefix, plane=""):
        filename = self._get_optics_filename(prefix, plane)
        if (self.directory / filename).is_file():
            return filename
        raise IOError(f"No file name found for prefix {prefix} in {self.directory!s}.")

    def write_to(self, value, prefix, plane=""):
        filename = self._get_optics_filename(prefix, plane)
        return filename, value
