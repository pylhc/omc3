import os
import random
import string
import tempfile
import itertools

import numpy as np
import pandas as pd
import pytest
import tfs

from pathlib import Path

import turn_by_turn as tbt
from omc3.definitions.constants import PLANES
from omc3.hole_in_one import hole_in_one_entrypoint

INPUTS = Path(__file__).parent.parent / "inputs"

HARPY_SETTINGS = dict(
    clean=True,
    to_write=['lin',],
    max_peak=[0.02],
    turn_bits=12,
)

# ==== HARPY 
@pytest.mark.basic
def test_default_harpy_resonance(tmp_path):
    '''
    Check that the --resonances flag indeed gives us up to 4th order by default
    '''
    model = INPUTS / "models" / f"2022_inj_b1_acd" / 'twiss.dat'
    input_files = [str(INPUTS / "lhc_200_turns.sdds")]

    # First run the frequency analysis with default resonances value
    clean, to_write, max_peak, turn_bits = HARPY_SETTINGS.values()
    hole_in_one_entrypoint(harpy=True,
                           clean=clean,
                           turn_bits=turn_bits,
                           autotunes="transverse",
                           outputdir=tmp_path,
                           files=input_files,
                           model=model,
                           to_write=to_write,
                           unit="mm")

    files = os.path.join(output_dir, 'lhc_200_turns.sdds')
    lin = dict(X=tfs.read(f"{files}.linx"),
               Y=tfs.read(f"{files}.liny"))

    # Some resonance lines for each order, as multiples of (Qx, Qy)
    # The same line on each plane is given by different RDT, but they're from the same magnet order
    # There is thus no need to differentiate the planes when checking the line
    r_lines = {2: [(0, 2), (-1, 1), (1, 1)],
               3: [(-1, 0), (3, 0), (1, -2), (0, 3)],
               4: [(-3, 0), (0, -3), (0, 3)]
               }
    for order, lines in r_lines.items():
        for line in lines:
            _assert_amp_lin(line, lin, present=True)

    # And now check that we *don't* have some lines
    r_lines = {5: [(-3, -1), (-2, -2), (-3, 1)],
               6: [(-1, -4), (-2, 3), (-5, 0)]
               }
    for order, lines in r_lines.items():
        for line in lines:
            _assert_amp_lin(line, lin, present=False)


@pytest.mark.basic
def test_harpy_bad_resonance_lower(tmp_path):
    '''
    Check that the --resonances minimum order is 2
    '''
    model = INPUTS / "models" / f"2022_inj_b1_acd" / 'twiss.dat'
    input_files = [str(INPUTS / "lhc_200_turns.sdds")]

    # First run the frequency analysis with default resonances value
    clean, to_write, max_peak, turn_bits = HARPY_SETTINGS.values()
    with pytest.raises(AttributeError) as e_info:
        hole_in_one_entrypoint(harpy=True,
                               clean=clean,
                               turn_bits=turn_bits,
                               autotunes="transverse",
                               outputdir=tmp_path,
                               files=input_files,
                               model=model,
                               to_write=to_write,
                               unit="mm",
                               resonances=1)

    assert 'minimum magnet order for resonance lines calculation is 2' in str(e_info)


@pytest.mark.basic
def test_harpy_bad_resonance_upper(tmp_path):
    '''
    Check that the --resonances maximum order is 8
    '''
    model = INPUTS / "models" / f"2022_inj_b1_acd" / 'twiss.dat'
    input_files = [str(INPUTS / "lhc_200_turns.sdds")]

    # First run the frequency analysis with default resonances value
    clean, to_write, max_peak, turn_bits = HARPY_SETTINGS.values()
    with pytest.raises(AttributeError) as e_info:
        hole_in_one_entrypoint(harpy=True,
                               clean=clean,
                               turn_bits=turn_bits,
                               autotunes="transverse",
                               outputdir=tmp_path,
                               files=input_files,
                               model=model,
                               to_write=to_write,
                               unit="mm",
                               resonances=12)

    assert 'maximum magnet order for resonance lines calculation is 8' in str(e_info)


@pytest.mark.extended
def test_harpy_high_order_resonance(tmp_path):
    '''
    Check the --resonances flag  with higher magnet orders: dodecapole (6)
    '''
    model = INPUTS / "models" / f"2022_inj_b1_acd" / 'twiss.dat'
    input_files = [str(INPUTS / "lhc_200_turns.sdds")]

    # First run the frequency analysis with default resonances value
    clean, to_write, max_peak, turn_bits = HARPY_SETTINGS.values()
    hole_in_one_entrypoint(harpy=True,
                           clean=clean,
                           turn_bits=turn_bits,
                           autotunes="transverse",
                           outputdir=tmp_path,
                           files=input_files,
                           model=model,
                           to_write=to_write,
                           unit="mm",
                           resonances=6)
    

    files = os.path.join(output_dir, 'lhc_200_turns.sdds')
    lin = dict(X=tfs.read(f"{files}.linx"),
               Y=tfs.read(f"{files}.liny"))

    r_lines = {5: [(-3, 1), (-1, -3), (3, -1), (0, -4), (0, 4)],
               6: [(-5, 0), (-4, -1), (-1, -4), (0, -5), (0, 5)],
               }
    for order, lines in r_lines.items():
        for line in lines:
            _assert_amp_lin(line, lin, present=True)



# ==== OPTICS
@pytest.mark.extended
def test_optics_default_rdt_order(tmp_path):
    '''
    Check the --rdt_magnet_order default (4)
    '''
    model = INPUTS / "models" / f"2022_inj_b1_acd" / 'twiss.dat'
    input_files = [str(INPUTS / "lhc_200_turns.sdds")]

    # First run the frequency analysis with default resonances value
    clean, to_write, max_peak, turn_bits = HARPY_SETTINGS.values()
    hole_in_one_entrypoint(harpy=True,
                           clean=True,
                           turn_bits=turn_bits,
                           autotunes="transverse",
                           outputdir=tmp_path,
                           files=input_files,
                           model=model,
                           to_write=to_write,
                           unit="mm")

    # And then the optics analysis, with default values as well
    files = [os.path.join(output_dir, 'lhc_200_turns.sdds')]
    hole_in_one_entrypoint(optics=True,
                           outputdir=tmp_path,
                           files=files,
                           model_dir=os.path.dirname(model),
                           accel="lhc",
                           year="2022",
                           beam=1,
                           nonlinear=['rdt'])

    # Now check that we got the wanted directories and nothing more
    rdt_dirs = os.listdir(os.path.join(output_dir, 'rdt'))
    assert len(rdt_dirs) == 6

    magnets = 'quadrupole', 'sextupole', 'octupole'
    prefixes = ('normal', 'skew')
    for magnet in magnets:
        for prefix in prefixes:
            assert f'{prefix}_{magnet}' in rdt_dirs

    # And verify the RDTs are hre
    sample_rdts = {'normal_sextupole': ['f3000_x', 'f0120_y'],
                   'skew_sextupole': ['f2001_x', 'f0030_y'],
                   'normal_octupole': ['f4000_x', 'f0040_y'],
                   'skew_octupole': ['f0310_y', 'f3001_x']
                   }
    for magnet_type, rdts in sample_rdts.items():
        actual_rdts = os.listdir(os.path.join(output_dir, 'rdt', magnet_type))
        for rdt in rdts:
            assert f'{rdt}.tfs' in actual_rdts


@pytest.mark.extended
@pytest.mark.parametrize("order", [1, 9])
def test_optics_wrong_rdt_magnet_order(tmp_path, order):
    '''
    Check that --rdt_magnet_order raises when > 8
    '''
    model = INPUTS / "models" / f"2022_inj_b1_acd" / 'twiss.dat'
    input_files = [str(INPUTS / "lhc_200_turns.sdds")]

    # First run the frequency analysis with default resonances value
    clean, to_write, max_peak, turn_bits = HARPY_SETTINGS.values()
    hole_in_one_entrypoint(harpy=True,
                           clean=True,
                           turn_bits=turn_bits,
                           autotunes="transverse",
                           outputdir=tmp_path,
                           files=input_files,
                           model=model,
                           to_write=to_write,
                           unit="mm")

    # And then the optics analysis, with default values as well
    files = [os.path.join(output_dir, 'lhc_200_turns.sdds')]
    with pytest.raises(AttributeError) as e_info:
        hole_in_one_entrypoint(optics=True,
                               outputdir=tmp_path,
                               files=files,
                               model_dir=os.path.dirname(model),
                               accel="lhc",
                               year="2022",
                               beam=1,
                               nonlinear=['rdt'],
                               rdt_magnet_order=order)
    assert "magnet order for RDT calculation should be between 2 and 8 (inclusive)" in str(e_info)


@pytest.mark.extended
def test_optics_higher_rdt_magnet_order(tmp_path):
    '''
    Check the --rdt_magnet_order with higher magnet orders: dodecapole (6)
    '''
    model = INPUTS / "models" / f"2022_inj_b1_acd" / 'twiss.dat'
    input_files = [str(INPUTS / "lhc_200_turns.sdds")]

    # First run the frequency analysis with default resonances value
    clean, to_write, max_peak, turn_bits = HARPY_SETTINGS.values()
    hole_in_one_entrypoint(harpy=True,
                           clean=True,
                           turn_bits=turn_bits,
                           autotunes="transverse",
                           outputdir=tmp_path,
                           files=input_files,
                           model=model,
                           to_write=to_write,
                           unit="mm",
                           resonances=6)

    # And then the optics analysis, with default values as well
    files = [os.path.join(output_dir, 'lhc_200_turns.sdds')]
    hole_in_one_entrypoint(optics=True,
                           outputdir=tmp_path,
                           files=files,
                           model_dir=os.path.dirname(model),
                           accel="lhc",
                           year="2022",
                           beam=1,
                           nonlinear=['rdt'],
                           rdt_magnet_order=6)

    # Now check that we got the wanted directories and nothing more
    rdt_dirs = os.listdir(os.path.join(output_dir, 'rdt'))
    assert len(rdt_dirs) == 10

    magnets = 'quadrupole', 'sextupole', 'octupole', 'decapole', 'dodecapole'
    prefixes = ('normal', 'skew')
    for magnet in magnets:
        for prefix in prefixes:
            assert f'{prefix}_{magnet}' in rdt_dirs

    # And verify the RDTs are hre
    sample_rdts = {'normal_sextupole': ['f3000_x', 'f0120_y'],
                   'skew_sextupole': ['f2001_x', 'f0030_y'],
                   'normal_octupole': ['f4000_x', 'f0040_y'],
                   'skew_octupole': ['f0310_y', 'f3001_x'],
                   'normal_decapole': ['f5000_x', 'f0320_y'],
                   'skew_decapole': ['f0410_y', 'f1301_x'],
                   'normal_dodecapole': ['f0060_y', 'f6000_x'],
                   'skew_dodecapole': ['f0510_y', 'f1005_x'],
                   }
    for magnet_type, rdts in sample_rdts.items():
        actual_rdts = os.listdir(os.path.join(output_dir, 'rdt', magnet_type))
        for rdt in rdts:
            assert f'{rdt}.tfs' in actual_rdts


def _assert_amp_lin(line, lin_dict, present):
    '''
    Check if an amplitude and its error exist in the lin columns
    e.g. AMP04 and ERRAMP04
    '''
    amp_str = f'AMP{line[0]}{line[1]}'.replace('-', '_')
    erramp_str = f'ERR{amp_str}'
    for plane, lin in lin_dict.items():
        if present:
            assert amp_str in lin.columns
            assert erramp_str in lin.columns
        if not present:
            assert amp_str not in lin.columns
            assert erramp_str not in lin.columns
