""" 
This folder contains LHC Beam 1 files at injection:

 - first three files are at 0Hz
 - second three files are at +50Hz
 - last three files are at -50Hz

see https://be-op-logbook.web.cern.ch/elogbook-server/GET/showEventInLogbook/3982395 .

The files have been truncated to 250 turns (turns 3001 - 3250) to save space.
"""
from collections.abc import Sequence
from pathlib import Path

import turn_by_turn as tbt


def truncate_turns_on_data(tbt_data: tbt.TbtData, start_turn: int = 0, end_turn: int = None):
    if end_turn is None:
        end_turn = tbt_data.nturns-1

    for data in tbt_data.matrices:
        for plane in data.fieldnames():
            setattr(data, plane, getattr(data, plane).loc[:, start_turn:end_turn])
    tbt_data.nturns = len(tbt_data.matrices[0].X.columns)
    return tbt_data

def truncate_turns(files: Sequence[Path], datatype: str, start_turn: int = 0, end_turn: int = None):
    for file in files:
        tbt_data = tbt.read_tbt(file, datatype=datatype)
        tbt_data = truncate_turns_on_data(tbt_data, start_turn, end_turn)

        output_path = file.with_name(f"{file.with_suffix('').name}_{tbt_data.nturns}turns{file.suffix}")
        tbt.write_tbt(output_path, tbt_data,  datatype=datatype)


if __name__ == "__main__":
    truncate_turns(
        files=list(Path(__file__).parent.glob("*.sdds")),
        datatype="lhc",
        start_turn=3001,
        end_turn=3250,
    )
