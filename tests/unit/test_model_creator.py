from os.path import join, isdir, abspath, dirname, pardir
from shutil import rmtree
from . import context
from utils import iotools
from model_creator import create_instance_and_model
from model.constants import MODIFIERS_MADX

BASE_OUTPUT = abspath(join(dirname(__file__), pardir, "model"))
COMP_MODEL = join(dirname(__file__), pardir, "inputs", "models", "25cm_beam1")


def _create_input():
    iotools.create_dirs(BASE_OUTPUT)
    iotools.copy_item(join(COMP_MODEL, "opticsfile.24_ctpps2"), join(BASE_OUTPUT, "strengths.madx"))
    iotools.write_string_into_new_file(join(BASE_OUTPUT, MODIFIERS_MADX),
                                       f"call, file='{join(BASE_OUTPUT, 'strengths.madx')}';\n")
    iotools.write_string_into_new_file(join(BASE_OUTPUT, "corrections.madx"), "\n")
    iotools.write_string_into_new_file(join(BASE_OUTPUT, "extracted_mqts.str"), "\n")


def test_lhc_creation_nominal():
    _create_input()
    opt_dict = dict(type="nominal", accel="lhc", year="2018", ats=True, beam=1,
                    nat_tunes=[0.31, 0.32], drv_tunes=[0.298, 0.335], driven_excitation="acd",
                    dpp=0.0, energy=6.5, modifiers=join(BASE_OUTPUT, MODIFIERS_MADX),
                    fullresponse=True, outputdir=BASE_OUTPUT,
                    writeto=join(BASE_OUTPUT, "job.twiss.madx"),
                    logfile=join(BASE_OUTPUT, "madx_log.txt"))
    create_instance_and_model(opt_dict)
    _clean_up(BASE_OUTPUT)


def test_lhc_creation_best_knowledge():
    _create_input()
    opt_dict = dict(type="best_knowledge", accel="lhc", year="2018", ats=True, beam=1,
                    nat_tunes=[0.31, 0.32], dpp=0.0, energy=6.5,
                    modifiers=join(BASE_OUTPUT, MODIFIERS_MADX), outputdir=BASE_OUTPUT,
                    writeto=join(BASE_OUTPUT, "job.twiss_best_knowledge.madx"),
                    logfile=join(BASE_OUTPUT, "madx_log_best_knowledge.txt"))
    create_instance_and_model(opt_dict)
    _clean_up(BASE_OUTPUT)


def _clean_up(path_dir):
    if isdir(path_dir):
        rmtree(path_dir, ignore_errors=True)
