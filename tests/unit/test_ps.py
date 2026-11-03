"""
PS Tests
--------

Tests for the PS Accelerator model creation, in particular the handling of
the sequence-file naming change in acc-models-ps starting from 2023
(see issue #548): a single unified 'ps.seq' replaced the two separate
'ps_mu.seq' / 'ps_ss.seq' files used in earlier years.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from omc3.model.constants import JOB_MODEL_MADX_NOMINAL, Fetcher
from omc3.model.model_creators.manager import CreatorType
from omc3.model_creator import create_instance_and_model

if TYPE_CHECKING:
    from pathlib import Path

PS_ACCEL_OPT = {
    "accel": "ps",
    "tune_method": "qf",
    "nat_tunes": [0.21, 0.24],
    "scenario": "bare_machine",
    "cycle_point": "0_proton_injection_energy",
    "str_file": "ps_pro_bare_machine.str",
    "energy": 2.14,
}


class TestModelCreationPS:
    @pytest.mark.basic
    def test_nominal_2023_uses_unified_sequence(self, tmp_path: Path, acc_models_ps_2023: Path):
        """ Regression test for #548: from 2023, acc-models-ps provides a
        single 'ps.seq' file instead of the two separate legacy files. """
        accel_opt = {
            **PS_ACCEL_OPT,
            "year": "2023",
            "fetch": Fetcher.PATH,
            "path": acc_models_ps_2023,
        }
        create_instance_and_model(
            outputdir=tmp_path, type=CreatorType.NOMINAL,
            logfile=tmp_path / "madx_log.txt", **accel_opt
        )
        job_content = (tmp_path / JOB_MODEL_MADX_NOMINAL).read_text()
        assert "ps.seq" in job_content
        assert "ps_mu.seq" not in job_content
        assert "ps_ss.seq" not in job_content

    @pytest.mark.basic
    def test_nominal_2021_uses_legacy_sequence_files(self, tmp_path: Path, acc_models_ps_2021: Path):
        """ Ensures the #548 fix does not break older years, which still
        provide the two separate sequence files. """
        accel_opt = {
            **PS_ACCEL_OPT,
            "year": "2021",
            "fetch": Fetcher.PATH,
            "path": acc_models_ps_2021,
        }
        create_instance_and_model(
            outputdir=tmp_path, type=CreatorType.NOMINAL,
            logfile=tmp_path / "madx_log.txt", **accel_opt
        )
        job_content = (tmp_path / JOB_MODEL_MADX_NOMINAL).read_text()
        assert "ps_mu.seq" in job_content
        assert "ps_ss.seq" in job_content
