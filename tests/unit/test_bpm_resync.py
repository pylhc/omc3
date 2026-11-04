from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tfs
import turn_by_turn as tbt
from generic_parser.dict_parser import ArgumentError

from omc3.definitions.constants import PLANE_TO_NUM, PLANES
from omc3.harpy.constants import COL_MU, FILE_LIN_EXT
from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.optics_measurements.constants import DELTA, ERR, MDL, NAME, PHASE, TUNE
from omc3.optics_measurements.toolbox import ang_diff, ang_interval_check
from omc3.scripts import resync_bpms as resync

INPUTS_DIR = Path(__file__).parent.parent / "inputs" / "bpm_resync"
OPTICS_DIR = Path(__file__).parent.parent / "inputs" / "bpm_resync"

# Simulated measurement: tunes close to the HER's, and how many turns each BPM is off
QX, QY = 0.532, 0.6
SHIFTS = [0, 1, -1, 2, -2]
NTURNS = 20


def simulated_measurement(optics_dir: Path, ring: str) -> tbt.TbtData:
    """Turn by turn data of two bunches where each BPM reads its turn number, off by its `SHIFTS`,
    and the matching phase files. `NOISY` has a phase no offset explains, `MISSING` is not in the
    optics, and the last BPM of `SHIFTS` is only in the Y plane."""
    names = [f"BPM{i}" for i in range(len(SHIFTS))]
    ring_dir = 1 if ring == "HER" else -1
    for plane, tune, bpms in (("X", QX, names[:-1]), ("Y", QY, names)):
        dphase = ang_interval_check(np.array(SHIFTS) * ring_dir * (1 - tune))
        optics = tfs.TfsDataFrame(
            {NAME: [*bpms, "NOISY"], f"{DELTA}{PHASE}{plane}": [*dphase[: len(bpms)], 0.25]},
            headers={f"{TUNE}{PLANE_TO_NUM[plane]}": tune},
        )
        tfs.write(optics_dir / resync.PHASE_FILE.format(plane=plane.lower()), optics)

    turns = pd.DataFrame(
        np.arange(NTURNS) + np.array([[*SHIFTS, 0, 0]]).T, index=[*names, "NOISY", "MISSING"]
    )
    matrices = [
        tbt.TransverseData(X=turns + 1000 * bunch, Y=turns + 1000 * bunch) for bunch in range(2)
    ]
    return tbt.TbtData(matrices=matrices, nturns=NTURNS)


def phase_errors(tbt_data: tbt.TbtData, outputdir: Path) -> np.ndarray:
    """Harpy's Y phase of the test data, a vertical kick, against the model in the optics."""
    hole_in_one_entrypoint(
        harpy=True,
        clean=False,
        autotunes="transverse",
        outputdir=outputdir,
        files=[tbt_data],
        tbt_datatype="tbt_data",
        to_write=["lin"],
        turn_bits=12,
        unit="m",
    )
    lin_file = next(outputdir.glob(f"*{FILE_LIN_EXT.format(plane='y')}"))
    muy = tfs.read(lin_file, index=NAME)[f"{COL_MU}Y"]
    optics = tfs.read(OPTICS_DIR / resync.PHASE_FILE.format(plane="y"), index=NAME)
    bpms = optics.index.intersection(muy.index)
    measured = ang_diff(muy.loc[bpms].to_numpy(), muy.loc[bpms[0]])
    model = ang_diff(optics.loc[bpms, f"{PHASE}Y{MDL}"].to_numpy(), optics.loc[bpms[0], f"{PHASE}Y{MDL}"])
    return ang_diff(measured, model)


def assert_synced(output_file: Path, outputdir: Path, planes: str = "xy"):
    """The output holds the BPMs of the optics in `planes`, shifted so that their phases read the
    model's."""
    output = tbt.read(output_file)
    checked = set().union(
        *(tfs.read(OPTICS_DIR / resync.PHASE_FILE.format(plane=p))[NAME] for p in planes)
    )
    assert set(output.matrices[0].X.index) == checked
    assert np.abs(phase_errors(output, outputdir)).max() < 0.1


@pytest.mark.parametrize(
    ("dphase", "offset", "unexplained"),
    [  # At QX, one turn is 0.468 in phase and two turns -0.064
        (0.01, 0, False),  # in sync
        (0.04, 0, False),  # phase drift
        (0.468, 1, False),
        (-0.468, -1, False),
        (-0.064, 2, False),
        (0.064, -2, False),
        (-0.119, 2, False),  # two turns and a phase error of 0.055
        (0.25, 0, True),  # noise
        (np.nan, 0, True),  # no phase
    ],
)
def test_turn_offsets(dphase, offset, unexplained):
    offsets, mask = resync.turn_offsets(np.array([dphase]), QX)
    assert offsets.tolist() == [offset]
    assert mask.tolist() == [unexplained]


def test_turn_offsets_tolerance():
    offsets, unexplained = resync.turn_offsets(np.array([0.04, 0.47]), QX, tolerance=0.01)
    assert offsets.tolist() == [0, 1]
    assert unexplained.tolist() == [True, False]


@pytest.mark.parametrize("ring", ["HER", "LER"])
def test_sync_tbt(tmp_path, caplog, ring):
    original = simulated_measurement(tmp_path, ring)
    synced = resync.sync_tbt(original, tmp_path, ring, "XY")

    kept_turns = np.arange(2, NTURNS - 2)  # 2 turns lost each way
    assert synced.nturns == len(kept_turns)
    for bunch, matrices in enumerate(synced.matrices):
        for plane in "XY":
            assert matrices[plane].columns.tolist() == list(range(len(kept_turns)))
            assert "MISSING" not in matrices[plane].index
            assert (matrices[plane].to_numpy() == kept_turns + 1000 * bunch).all()
    assert "NOISY" in caplog.text
    assert "MISSING" in caplog.text

    # The input is left untouched
    assert original.nturns == NTURNS
    assert original.matrices[0].X.shape == (len(SHIFTS) + 2, NTURNS)


@pytest.mark.parametrize("kicked", PLANES)
def test_sync_tbt_default_plane(tmp_path, kicked):
    # Only the error bars tell the planes apart
    for plane in PLANES:
        phase_file = resync.PHASE_FILE.format(plane=plane.lower())
        optics = tfs.read(OPTICS_DIR / phase_file)
        optics[f"{ERR}{DELTA}{PHASE}{plane}"] = 0.001 if plane == kicked else 0.1
        tfs.write(tmp_path / phase_file, optics)

    original = tbt.read(INPUTS_DIR / "unsynced.sdds")
    default = resync.sync_tbt(original, tmp_path, "HER").matrices[0]
    explicit = resync.sync_tbt(original, OPTICS_DIR, "HER", kicked).matrices[0]
    for plane in PLANES:
        assert default[plane].equals(explicit[plane])


def test_sync_tbt_tolerance_above_two_turn_step(tmp_path, caplog):
    original = simulated_measurement(tmp_path, "HER")
    resync.sync_tbt(original, tmp_path, "HER", "X", tolerance=0.1)
    assert "above the two-turn step" in caplog.text


def test_bad_arg_input_type(tmp_path):
    with pytest.raises(ArgumentError) as e:
        resync.main(input=42, optics_dir=OPTICS_DIR, output_file=tmp_path / "wat", ring="HER")
    assert "input must be" in str(e.value)


def test_bad_arg_optics_type():
    with pytest.raises(ArgumentError) as e:
        resync.main(input=Path("yeah ok"), optics_dir=42, output_file="yay", ring="HER")
    assert "optics_dir' is not of type Path" in str(e.value)


def test_bad_arg_output_file_type():
    with pytest.raises(ArgumentError) as e:
        resync.main(input=Path("yeah ok"), optics_dir=Path("yay"), output_file=42, ring="HER")
    assert "output_file' is not of type Path" in str(e.value)


def test_bad_arg_ring():
    with pytest.raises(ArgumentError) as e:
        resync.main(
            input="yeah ok", optics_dir=Path("yay"), output_file=Path("wat"), ring="MOON_COLLIDER"
        )
    assert "ring' needs to be one of" in str(e.value)


def test_bad_arg_tbt_datatype():
    with pytest.raises(ArgumentError) as e:
        resync.main(
            input="yeah ok",
            optics_dir=Path("yay"),
            output_file=Path("wat"),
            ring="HER",
            tbt_datatype="quantum_sdds",
        )
    assert "tbt_datatype' needs to be one of" in str(e.value)


def test_resync(tmp_path):
    # The raw data has BPMs turns off, otherwise the check below would pass without the resync
    assert np.abs(phase_errors(tbt.read(INPUTS_DIR / "unsynced.sdds"), tmp_path / "raw")).max() > 0.4

    # Without planes, the kicked plane is found on its own
    resync.main(
        input=INPUTS_DIR / "unsynced.sdds",
        optics_dir=OPTICS_DIR,
        output_file=tmp_path / "output.sdds",
        ring="HER",
    )
    assert_synced(tmp_path / "output.sdds", tmp_path / "synced", planes="y")


def test_resync_tbt_data_input(tmp_path):
    resync.main(
        input=tbt.read(INPUTS_DIR / "unsynced.sdds"),
        optics_dir=OPTICS_DIR,
        output_file=tmp_path / "output.sdds",
        ring="HER",
        planes="YX",
    )
    assert_synced(tmp_path / "output.sdds", tmp_path / "synced")


def test_overwrite_ok(tmp_path):
    # Write an output file to create a conflict
    (tmp_path / "output.sdds").write_text("This file already exists.")

    resync.main(
        input=INPUTS_DIR / "unsynced.sdds",
        optics_dir=OPTICS_DIR,
        output_file=tmp_path / "output.sdds",
        ring="HER",
        planes="YX",
        overwrite=True,
    )
    assert_synced(tmp_path / "output.sdds", tmp_path / "synced")


def test_overwrite_raise(tmp_path):
    # Write an output file to create a conflict
    (tmp_path / "output.sdds").write_text("This file already exists.")

    with pytest.raises(FileExistsError) as e:
        resync.main(
            input=INPUTS_DIR / "unsynced.sdds",
            optics_dir=OPTICS_DIR,
            output_file=tmp_path / "output.sdds",
            ring="HER",
            overwrite=False,
        )

    assert "output.sdds already exists, aborting." in str(e.value)


def test_wrong_plane(tmp_path):
    with pytest.raises(ArgumentError) as e:
        resync.main(
            input=INPUTS_DIR / "unsynced.sdds",
            optics_dir=OPTICS_DIR,
            output_file=tmp_path / "output.sdds",
            ring="HER",
            planes="W",
        )

    assert "planes' needs to be one of '['X', 'Y', 'XY', 'YX']" in str(e.value)


def test_resync_planes(tmp_path):
    # Resync via both planes
    resync.main(
        input=INPUTS_DIR / "unsynced.sdds",
        optics_dir=OPTICS_DIR,
        output_file=tmp_path / "output_X.sdds",
        ring="HER",
        planes="XY",
    )

    resync.main(
        input=INPUTS_DIR / "unsynced.sdds",
        optics_dir=OPTICS_DIR,
        output_file=tmp_path / "output_Y.sdds",
        ring="HER",
        planes="YX",
    )

    output_x_tbt = tbt.read(tmp_path / "output_X.sdds")
    output_y_tbt = tbt.read(tmp_path / "output_Y.sdds")

    # Some BPMs are not going to be resynced the same way depending on which plane we start with
    assert not output_x_tbt.matrices[0].X.equals(output_y_tbt.matrices[0].X)
    assert not output_x_tbt.matrices[0].Y.equals(output_y_tbt.matrices[0].Y)

    # Y holds the signal, X decides only for the BPMs Y lacks
    assert_synced(tmp_path / "output_Y.sdds", tmp_path / "synced")
