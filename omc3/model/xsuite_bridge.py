"""
Xsuite Bridge
-------------

Single owner of "get me an ``xtrack`` lattice for this accelerator / model-creator".

Historically ``omc3`` drives ``MAD-X`` via :mod:`omc3.madx_wrapper` from several places
(model creation, response, MAD-NG sequence build). As part of the migration to
`xsuite <https://xsuite.readthedocs.io>`_ those subsystems instead obtain their lattice
as an ``xtrack`` :class:`~xtrack.Environment` produced here.

The accelerator *sequence* is still built and matched by ``MAD-X`` once (via
:func:`build_madx_sequence`) - this is the single sanctioned ``MAD-X`` bridge, retained
because ``xtrack``'s pure-Python ``MAD-X`` parser cannot handle the acc-models operational
knob toolkit. Everything downstream (twiss, response, corrected models) is pure ``xtrack``.

``xtrack`` is an optional dependency; its import is kept lazy (inside the functions).

:author: OMC Team
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import omc3.madx_wrapper as madx_wrapper
from omc3.utils import logging_tools

if TYPE_CHECKING:
    from pathlib import Path

    from omc3.model.accelerators.accelerator import Accelerator
    from omc3.model.model_creators.abstract_model_creator import ModelCreator

LOGGER = logging_tools.get_logger(__name__)

# Proton rest mass in eV, used to set the xtrack reference particle.
_PROTON_MASS_EV: float = 938.272_088_16e6

# Filename of the xtrack lattice (Environment) json produced from the MAD-X-built sequence.
XSUITE_JSON: str = "xsuite_lattice.json"


def build_madx_sequence(creator: ModelCreator, *, log_file: Path | None = None) -> Path:
    """Build and match the accelerator sequence with ``MAD-X`` and save it to a ``.seq`` file.

    This is the **only** place ``omc3`` invokes the ``MAD-X`` binary for the xsuite / MAD-NG
    lattice. It runs ``get_base_madx_script`` + ``get_save_sequence_script`` (the model-creator
    seam) and returns the path to the saved sequence.

    The caller is expected to have run ``creator.prepare_run()`` beforehand (so macros / symlinks
    are in place), exactly as the model-creation ``full_run`` does.

    Args:
        creator: the model creator, providing the MAD-X script and the model directory / paths.
        log_file: optional explicit path for the MAD-X log; defaults to the creator's ``logfile``.

    Returns:
        Path to the saved ``.seq`` sequence file.
    """
    accel = creator.accel
    LOGGER.info("Building the accelerator sequence with MAD-X.")

    madx_script = creator.get_base_madx_script() + "\n" + creator.get_save_sequence_script()

    run_kwargs = {}
    if accel.model_dir is not None:
        run_kwargs["cwd"] = accel.model_dir
        if log_file is not None:
            run_kwargs["log_file"] = log_file
        elif creator.logfile is not None:
            run_kwargs["log_file"] = accel.model_dir / creator.logfile
        if creator.jobfile is not None:
            run_kwargs["output_file"] = accel.model_dir / creator.jobfile
    madx_wrapper.run_string(madx_script, **run_kwargs)

    seq_file = accel.model_dir / creator.save_sequence_filename
    if not seq_file.exists():
        raise FileNotFoundError(
            f"MAD-X did not produce the expected sequence file {seq_file}. "
            f"Check the MAD-X log{f' at {log_file}' if log_file else ''}."
        )
    return seq_file


def create_xsuite_json(creator: ModelCreator, json_file: Path, *, cache: bool = False) -> Path:
    """Build the accelerator lattice with ``MAD-X`` and save it as an ``xtrack`` Environment json.

    Runs :func:`build_madx_sequence` to produce the ``.seq``, loads it into ``xtrack`` and
    serialises the environment to ``json_file``. This is the single sanctioned ``MAD-X`` bridge.

    Args:
        creator: the model creator (provides the MAD-X script and the model directory / paths).
        json_file: destination path for the xtrack Environment json.
        cache: if ``True`` and ``json_file`` already exists, return it without rebuilding. Defaults
            to ``False`` (always rebuild): cache reuse is only safe when the caller guarantees the
            ``json_file`` matches the current modifiers/energy (e.g. a fresh model dir per run). Pass
            ``cache=True`` explicitly when reusing a lattice across e.g. correction iterations.

    Returns:
        The path to the (possibly newly created) json file.
    """
    import xtrack as xt

    if cache and json_file.exists():
        LOGGER.debug(f"Reusing existing xsuite lattice json {json_file}.")
        return json_file

    seq_file = build_madx_sequence(creator)
    env = xt.load(file=str(seq_file), format="madx")
    env.to_json(str(json_file))
    LOGGER.info(f"Saved xsuite lattice json to {json_file}.")
    return json_file


def load_line(accel: Accelerator, json_file: Path, sequence_name: str):
    """Load an ``xtrack`` line from a lattice json and set its reference particle.

    Centralises reference-particle setup (and, in a later milestone, beam-direction handling)
    so every subsystem obtains its line the same way.

    Args:
        accel: the accelerator instance (for energy / beam).
        json_file: path to the xtrack Environment json.
        sequence_name: the sequence / line name to select from the environment.

    Returns:
        Tuple ``(env, line)``.
    """
    import xtrack as xt

    env = xt.Environment.from_json(str(json_file))
    line = env[sequence_name.lower()]
    line.particle_ref = xt.Particles(p0c=accel.energy * 1e9, mass0=_PROTON_MASS_EV)
    return env, line


def apply_correction_files(env, corr_files) -> None:
    """Apply omc3 correction files (``knob = knob +value;``) as ``xtrack`` env-var increments.

    The correction files written by :func:`omc3.correction.handler.writeparams` contain simple
    ``knob = knob <signed value>;`` statements. Rather than round-tripping through MAD-X, apply
    them directly to the ``xtrack`` environment variables.

    Args:
        env: the xtrack :class:`~xtrack.Environment` (or line) whose vars to update.
        corr_files: iterable of paths to omc3 correction files.
    """
    for corr_file in corr_files:
        for raw in _parse_correction_file(corr_file):
            knob, value = raw
            env[knob] = env[knob] + value


def _parse_correction_file(corr_file: Path) -> list[tuple[str, float]]:
    """Parse ``knob = knob +value;`` lines from an omc3 correction file into ``(knob, value)``."""
    from pathlib import Path

    updates: list[tuple[str, float]] = []
    for line in Path(corr_file).read_text().splitlines():
        line = line.split("!", 1)[0].strip().rstrip(";").strip()
        if not line or "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        knob = lhs.strip()
        # rhs is of the form "<knob> <signed value>" (e.g. "kqt.a12 +1.2e-05")
        tokens = rhs.split()
        value = float(tokens[-1])
        updates.append((knob, value))
    return updates
