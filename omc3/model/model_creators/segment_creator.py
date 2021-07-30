"""
Segment Creator
---------------

This module provides convenience functions for model creation of a ``segment``.
"""
import shutil
import tfs
import numpy

from pathlib import Path

from omc3.model.constants import MACROS_DIR, GENERAL_MACROS, LHC_MACROS
from omc3.utils import logging_tools
from omc3.utils.iotools import create_dirs


LOGGER = logging_tools.get_logger(__name__)

def get_R_terms(betx, bety, alfx, alfy, f1001r, f1001i, f1010r, f1010i):

    ga11 = 1 / numpy.sqrt(betx)
    ga12 = 0
    ga21 = alfx / numpy.sqrt(betx)
    ga22 = numpy.sqrt(betx)
    Ga = numpy.reshape(numpy.array([ga11, ga12, ga21, ga22]), (2, 2))

    gb11 = 1 / numpy.sqrt(bety)
    gb12 = 0
    gb21 = alfy / numpy.sqrt(bety)
    gb22 = numpy.sqrt(bety)
    Gb = numpy.reshape(numpy.array([gb11, gb12, gb21, gb22]), (2, 2))

    J = numpy.reshape(numpy.array([0, 1, -1, 0]), (2, 2))

    absf1001 = numpy.sqrt(f1001r ** 2 + f1001i ** 2)
    absf1010 = numpy.sqrt(f1010r ** 2 + f1010i ** 2)

    gamma2 = 1. / (1. + 4. * (absf1001 ** 2 - absf1010 ** 2))
    c11 = (f1001i + f1010i)
    c22 = (f1001i - f1010i)
    c12 = -(f1010r - f1001r)
    c21 = -(f1010r + f1001r)
    Cbar = numpy.reshape(2 * numpy.sqrt(gamma2) *
                         numpy.array([c11, c12, c21, c22]), (2, 2))

    C = numpy.dot(numpy.linalg.inv(Ga), numpy.dot(Cbar, Gb))
    jCj = numpy.dot(J, numpy.dot(C, -J))
    c = numpy.linalg.det(C)
    r = -c / (c - 1)
    R = numpy.transpose(numpy.sqrt(1 + r) * jCj)
    return numpy.ravel(R)

def create_measurement_file(sbs_path, measurement_dir, betain_name, opt):
    isTest = True
    phase_beta_x = Path('beta_phase_x.tfs')
    phase_beta_y = Path('beta_phase_y.tfs')

    df_betx = tfs.read(measurement_dir / phase_beta_x, index="NAME")
    df_bety = tfs.read(measurement_dir / phase_beta_y, index="NAME")

    betx_start =  df_betx['BETX'].loc[opt.start]
    betx_end   =  df_betx['BETX'].loc[opt.end]

    alfx_start =  df_betx['ALFX'].loc[opt.start]
    alfx_end   =  -df_betx['ALFX'].loc[opt.end]

    bety_start =  df_bety['BETY'].loc[opt.start]
    bety_end   =  df_bety['BETY'].loc[opt.end]

    alfy_start =  df_bety['ALFY'].loc[opt.start]
    alfy_end   =  -df_bety['ALFY'].loc[opt.end]
    '''
    if(isTest):
        f_ini={}
        f_end={}
        f_ini['f1001r'] = 0.001
        f_ini["f1001i"] = 0.002
        f_ini["f1010r"] = 0.0001
        f_ini["f1010i"] = 0.0002

        f_end["f1001r"] = 0.0032
        f_end["f1001i"] = 0.0012
        f_end["f1010r"] = 0.00013
        f_end["f1010i"] = 0.0002

    ini_r11, ini_r12, ini_r21, ini_r22 = get_R_terms(
        betx_start, bety_start, alfx_start, alfy_start,
        f_ini["f1001r"], f_ini["f1001i"],
        f_ini["f1010r"], f_ini["f1010i"]
    )
    end_r11, end_r12, end_r21, end_r22 = get_R_terms(
        betx_end, bety_end, alfx_end, alfy_end,
        f_end["f1001r"], f_end["f1001i"],
        f_end["f1010r"], f_end["f1010i"]
    )
    '''
    measurement_dict = dict(
        betx_ini=betx_start,
        bety_ini=bety_start,
        alfx_ini=alfx_start,
        alfy_ini=alfy_start,
        dx_ini=0,
        dy_ini=0,
        dpx_ini=0,
        dpy_ini=0,
        wx_ini=0,
        phix_ini=0,
        wy_ini=0,
        phiy_ini=0,
        wx_end=0,
        phix_end=0,
        wy_end=0,
        phiy_end=0,
        ini_r11=0,
        ini_r12=0,
        ini_r21=0,
        ini_r22=0,
        end_r11=0,
        end_r12=0,
        end_r21=0,
        end_r22=0,
        betx_end=betx_end,
        bety_end=bety_end,
        alfx_end=alfx_end,
        alfy_end=alfy_end,
        dx_end=0,
        dy_end=0,
        dpx_end=0,
        dpy_end=0,
    )

    betainputfile = f"{sbs_path}/{betain_name}"
    with open(betainputfile, "w") as measurement_file:
        for name, value in measurement_dict.items():
            measurement_file.write(name + " = " + str(value) + ";\n")

def _create_correction_file(sbs_path, label):
    corr_file = Path("corrections_" + label + ".madx")
    corr_file = sbs_path / corr_file
    if(corr_file.exists()==False):
        f = open(corr_file, "w")
        f.write("!Enter the corrections below:")
        f.close()


class SegmentCreator(object):

    @classmethod
    def prepare_run(cls, accel):
        macros_path = accel.model_dir / MACROS_DIR
        create_dirs(macros_path)
        lib_path = Path(__file__).parent.parent/ "madx_macros"
        shutil.copy(lib_path / GENERAL_MACROS, macros_path / GENERAL_MACROS)
        shutil.copy(lib_path / LHC_MACROS, macros_path / LHC_MACROS)

    @classmethod
    def get_madx_script(cls, accel, opt):

        sbs_path = opt.outputdir
        _create_correction_file(sbs_path, opt.label)
        betain_name = Path("measurement_" + opt.label + ".madx")
        create_measurement_file(sbs_path, opt.measuredir, betain_name, opt)

        libs = f"call, file = '{opt.outputdir / MACROS_DIR / GENERAL_MACROS}';\n"
        libs = libs + f"call, file = '{opt.outputdir/ MACROS_DIR / LHC_MACROS}';\n"

        madx_template = accel.get_file("segment.madx").read_text()

        replace_dict = {
            "MAIN_SEQ": accel.load_main_seq_madx(),  # LHC only
            "OPTICS_PATH": accel.modifiers,  # all
            "NUM_BEAM": accel.beam,  # LHC only
            "PATH": accel.model_dir,  # all
            # "OUTPUT": accel.model_dir,  # Booster only
            "LABEL": accel.label,  # all
            "BETAKIND": accel.kind,  # all
            "STARTFROM": accel.start.name,  # all
            "ENDAT": accel.end.name,  # all
            # "RING": accel.ring,  # Booster only
            # "KINETICENERGY": accel.energy,  # PS only
            # "FILES_DIR": accel.get_dir(),  # Booster and PS
            # "NAT_TUNE_X": accel.nat_tunes[0],  # Booster and PS
            # "NAT_TUNE_Y": accel.nat_tunes[1],  # Booster and PS
        }
        return madx_template % replace_dict
