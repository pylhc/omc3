"""
Model Creator
-------------

Entrypoint to run the model creator for LHC, PSBooster and PS models.
"""
import tfs
import pandas as pd
import numpy as numpy
from pathlib import Path

from generic_parser import EntryPointParameters, entrypoint

from omc3.madx_wrapper import run_string
from omc3.model import manager
from omc3.model.constants import JOB_MODEL_MADX
from omc3.model.model_creators.lhc_model_creator import (  # noqa
    LhcBestKnowledgeCreator,
    LhcCouplingCreator,
    LhcModelCreator,
)
from omc3.model.model_creators.ps_model_creator import PsModelCreator
from omc3.model.model_creators.psbooster_model_creator import PsboosterModelCreator
from omc3.model.model_creators.segment_creator import SegmentCreator
from omc3.utils.iotools import create_dirs
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


def _get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="first",
        required=True,
        help="First BPM in segment."
    )
    params.add_parameter(
        name="last",
        required=True,
        help="Last BPM in segment."
    )
    params.add_parameter(
        name="resultdir",
        type=Path,
        required=True,
        help=("Path to the directory containing the measurements.")
    )    
    params.add_parameter(
        name="logfile",
        type=Path,
        help=("Path to the file where to write the MAD-X script output."
              "If not provided it will be written to sys.stdout.")
    )

    return params


# Main functions ###############################################################


@entrypoint(_get_params())
def create_segment(opt,accel_opt):
    isTest = True

    print(opt.first)
    print(opt.last)
    phase_beta_x = Path('beta_phase_x.tfs')
    phase_beta_y = Path('beta_phase_y.tfs')
    
    df_betx = tfs.read(opt.resultdir / phase_beta_x)
    df_bety = tfs.read(opt.resultdir / phase_beta_y)


    df_first_x = df_betx.loc[df_betx['NAME']==opt.first]
    df_first_y = df_bety.loc[df_bety['NAME']==opt.first]

    df_last_x = df_betx.loc[df_betx['NAME']==opt.first]
    df_last_y = df_bety.loc[df_bety['NAME']==opt.first]    
    #print(df_bety.loc[df_bety['NAME']==opt.first]['BETY'])
    #newpd = pd.concat([df_first, tmpy])

    if(isTest):
        f_ini=dict(
        f1001r=0.001)
        f_end={}
        print(f_ini)
        betx_ini=df_first_x["BETX"].values[0]
        bety_ini=df_first_y["BETY"].values[0]
        alfx_ini=df_first_x["ALFX"].values[0]
        alfy_ini=df_first_y["ALFY"].values[0]


        betx_end=df_last_x["BETX"].values[0]
        bety_end=df_last_y["BETY"].values[0]
        alfx_end=df_last_x["ALFX"].values[0]
        alfy_end=df_last_y["ALFY"].values[0]

        f_ini['f1001r'] = 0.001
        f_ini["f1001i"] = 0.002
        f_ini["f1010r"] = 0.0001
        f_ini["f1010i"] = 0.0002

        f_end["f1001r"] = 0.0032
        f_end["f1001i"] = 0.0012
        f_end["f1010r"] = 0.00013
        f_end["f1010i"] = 0.0002

    ini_r11, ini_r12, ini_r21, ini_r22 = _get_R_terms(
        betx_ini, bety_ini, alfx_ini, alfy_ini,
        f_ini["f1001r"], f_ini["f1001i"],
        f_ini["f1010r"], f_ini["f1010i"]
    )
    end_r11, end_r12, end_r21, end_r22 = _get_R_terms(
        betx_end, bety_end, alfx_end, alfy_end,
        f_end["f1001r"], f_end["f1001i"],
        f_end["f1010r"], f_end["f1010i"]
    )

    measurement_dict = dict(
        betx_ini=df_first_x["BETX"].values[0],
        bety_ini=df_first_y["BETY"].values[0],
        alfx_ini=df_first_x["ALFX"].values[0],
        alfy_ini=df_first_y["ALFY"].values[0],
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
        ini_r11=ini_r11,
        ini_r12=ini_r12,
        ini_r21=ini_r21,
        ini_r22=ini_r22,
        end_r11=end_r11,
        end_r12=end_r12,
        end_r21=end_r21,
        end_r22=end_r22,
        betx_end=df_last_x["BETX"].values[0],
        bety_end=df_last_y["BETY"].values[0],
        alfx_end=df_last_x["ALFX"].values[0],
        alfy_end=df_last_y["ALFY"].values[0],
        dx_end=0,
        dy_end=0,
        dpx_end=0,
        dpy_end=0,
    )

    #with open(os.path.join(save_path, "measurement"+betakind+"_" + accel_instance.label + ".madx"), "w") as measurement_file:
    with open("betainput.madx", "w") as measurement_file:
    
        for name, value in measurement_dict.items():
            measurement_file.write(name + " = " + str(value) + ";\n")


def _get_R_terms(betx, bety, alfx, alfy, f1001r, f1001i, f1010r, f1010i):

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

if __name__ == "__main__":
    create_segment()
