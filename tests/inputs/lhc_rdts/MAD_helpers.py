from __future__ import annotations

import re
from pathlib import Path

import cpymad.madx as madx
import pandas as pd
import tfs
import turn_by_turn as tbt
from pymadng import MAD
from turn_by_turn import madng

from omc3.model_creator import create_instance_and_model
from tests.inputs.lhc_rdts.rdt_constants import (
    ANALYSIS_DIR,
    DATA_DIR,
    FREQ_OUT_DIR,
    NTURNS,
    ACC_MODELS,
    MODEL_NG_PREFIX,
    MODEL_X_PREFIX,
    MODEL_ANALYTICAL_PREFIX,
) 

from tests.inputs.lhc_rdts.omc3_helpers import (
    get_file_ext,
    get_max_rdt_order,
    get_model_dir,
    get_tbt_name,
)


ANALYSIS_DIR.mkdir(exist_ok=True)
FREQ_OUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

def to_ng_rdts(rdts: list[str]) -> list[str]:
    return list(set([rdt.split("_")[0] for rdt in rdts]))

def create_model_dir(beam: int, order: int, is_skew: bool) -> None:
    model_dir = get_model_dir(beam, order, is_skew)
    create_instance_and_model(
        accel="lhc",
        fetch="afs",
        type="nominal",
        beam=beam,
        year="2024",
        driven_excitation="acd",
        energy=6800.0,
        nat_tunes=[0.28, 0.31],
        drv_tunes=[0.28, 0.31],
        modifiers=[ACC_MODELS / "operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx"],
        outputdir=model_dir,
    )
    with open(model_dir / "job.create_model_nominal.madx", "r") as f:
        lines = f.readlines()
        with madx.Madx(stdout=False) as mad:
            mad.chdir(str(model_dir))
            for i, line in enumerate(lines):
                if i < 32:
                    mad.input(line)
            mad.input(f"""
set, format= "-.16e";
save, sequence=lhcb{beam}, file="lhcb{beam}_saved.seq";
            """)
        update_model_with_ng(beam, order, is_skew)
        if beam == 2:  # Now create the sequence as beam 4
            with madx.Madx(stdout=f) as mad:
                mad.chdir(str(model_dir))
                for i, line in enumerate(lines):
                    if "define_nominal_beams" in line:
                        mad.input(
                            "beam, sequence=LHCB2, particle=proton, energy=450, kbunch=1, npart=1.15E11, bv=1;"
                        )
                        continue
                    if "acc-models-lhc/lhc.seq" in line:
                        line = line.replace(
                            "acc-models-lhc/lhc.seq", "acc-models-lhc/lhcb4.seq"
                        )
                    if i < 32:
                        mad.input(line)
                mad.input(f"""
set, format= "-.16e";
save, sequence=lhcb{beam}, file="lhcb{beam}_saved.seq";
            """)


def update_model_with_ng(beam: int, order: int, is_skew: bool) -> None:
    model_dir = get_model_dir(beam, order, is_skew)
    with MAD() as mad:
        seq_dir = -1 if beam == 2 else 1
        initialise_model(mad, beam, order, is_skew, seq_dir=seq_dir)
        # Create twiss_elements and twiss_ac and twiss data tables in model folder
        mad.send(f"""
hnams = {{
    "name", "type", "title", "origin", "date", "time", "refcol", "direction", 
    "observe", "energy", "deltap", "length", "q1", "q2", "q3", "dq1", "dq2", 
    "dq3", "alfap", "etap", "gammatr"
}}        
cols = {{
    'name', 'kind','s','betx','alfx','bety','alfy', 'mu1' ,'mu2', 'r11', 'r12',
    'r21', 'r22', 'x','y','dx','dpx','dy','dpy'
}}
str_cols = py:recv()
cols = MAD.utility.tblcat(cols, str_cols)
if {beam} == 1 then  -- Cycle the sequence to the correct location 
    MADX.lhcb1:cycle("MSIA.EXIT.B1")
end
! Coupling needs to be true to calculate Edwards-Teng parameters and R matrix
twiss_elements = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true}} 
""").send(strength_cols)
        # MAD.gphys.melmcol(twiss_elements, str_cols)
        add_strengths_to_twiss(mad, "twiss_elements")
        mad.send(
            f"""twiss_elements:write("{model_dir / 'twiss_elements.dat'}", cols, hnams)"""
        )
        observe_BPMs(mad, beam)
        mad.send(f"""
twiss_ac   = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true, observe=1}}
twiss_data = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true, observe=1}}
        """)
        add_strengths_to_twiss(mad, "twiss_ac")
        add_strengths_to_twiss(mad, "twiss_data")
        mad.send(f"""
twiss_ac:write("{model_dir / 'twiss_ac.dat'}", cols, hnams)
twiss_data:write("{model_dir / 'twiss.dat'}", cols, hnams)
print("Replaced twiss data tables")
py:send(1)
""")
        assert mad.receive() == 1, "Error in updating model with new optics"

    # Read the twiss data tables and then convert all the headers to uppercase and column names to uppercase
    export_tfs_to_madx(model_dir / "twiss_ac.dat", beam)
    export_tfs_to_madx(model_dir / "twiss_elements.dat", beam)
    export_tfs_to_madx(model_dir / "twiss.dat", beam)


def observe_BPMs(mad: MAD, beam: int) -> None:
    mad.send(f"""
local observed in MAD.element.flags
MADX.lhcb{beam}:deselect(observed)
MADX.lhcb{beam}:  select(observed, {{pattern="BPM"}})
if {beam} == 1 then  -- Cycle the sequence to the correct location 
    MADX.lhcb1:cycle("MSIA.EXIT.B1")
end
    """)


def export_tfs_to_madx(tfs_file: Path, beam: int) -> None:
    tfs_df = tfs.read(tfs_file)
    tfs_df = convert_tfs_to_madx(tfs_df, beam)
    tfs.write(tfs_file, tfs_df, save_index="NAME")


def convert_tfs_to_madx(tfs_df: tfs.TfsDataFrame, beam: int) -> tfs.TfsDataFrame:
    # First convert all the headers to uppercase and column names to uppercase
    tfs_df.columns = tfs_df.columns.str.upper()
    tfs_df.headers = {key.upper(): value for key, value in tfs_df.headers.items()}

    # Change the columns mu1 and mu2 to mux and muy
    tfs_df.rename(columns={"MU1": "MUX", "MU2": "MUY"}, inplace=True)

    # Change all the drift numbers (the # in DRIFT_#) are consecutive and start from 0
    drifts = tfs_df[tfs_df["KIND"] == "drift"]
    replace_column = [f"DRIFT_{i}" for i in range(len(drifts))]
    tfs_df["NAME"] = tfs_df["NAME"].replace(drifts["NAME"].to_list(), replace_column)

    # Remove all rows that has a "vkicker" or "hkicker" in the KIND column (not seen in MADX)
    tfs_df = tfs_df[~tfs_df["KIND"].str.contains("vkicker|hkicker")]

    tfs_df.set_index("NAME", inplace=True)

    # Remove the rows with "$start" and "$end" in the NAME column
    tfs_df = tfs_df.filter(regex=r"^(?!\$start|\$end).*$", axis="index")
    return tfs_df


strength_cols = ["k1l", "k2l", "k3l", "k4l", "k1sl", "k2sl", "k3sl", "k4sl"]


def add_strengths_to_twiss(mad: MAD, mtable_name: str) -> None:
    mad.send(f"""
strength_cols = py:recv()
MAD.gphys.melmcol({mtable_name}, strength_cols)
    """).send(strength_cols)


def initialise_model(
    mad: MAD,
    beam: int,
    order: int,
    is_skew: bool,
    seq_dir: int = 1,
    kick_amp: float = 1e-3,
) -> None:
    assert beam in [1, 2] and isinstance(beam, int), "Beam must be 1 or 2"
    model_dir = get_model_dir(beam, order, is_skew)
    mad.MADX.load(
        f"'{model_dir/f'lhcb{beam}_saved.seq'}'",
        f"'{model_dir/f'lhcb{beam}_saved.mad'}'",
    )
    mad.send(f"""
lhc_beam = beam {{particle="proton", energy=450}}
MADX.lhcb{beam}.beam = lhc_beam
MADX.lhcb{beam}.dir = {seq_dir}
local kick_amp = py:recv()
print("Initialising model with beam:", {beam}, "dir:", MADX.lhcb{beam}.dir, "kick amplitude:", kick_amp)
X0 = {{x=kick_amp, y=-kick_amp, px=0, py=0, t=0, pt=0}}
    """).send(kick_amp)

    turnoff_sextupoles(mad, beam)
    add_magnet_strengths(mad, beam, order, is_skew)
    match_tunes(mad, beam)


def add_magnet_strengths(mad: MAD, beam: int, order: int, is_skew: bool) -> None:
    assert order in [2, 3], "Order must be 2 or 3"
    if order == 2:
        s_or_o = "s"
        strength = 1e-3
    else:
        s_or_o = "o"
        strength = 1e-2
    skew = "s" if is_skew else ""  # s for skew, "" for normal
    mad.send(f"""
MADX.kc{s_or_o}{skew}x3_r1 = {strength};
MADX.kc{s_or_o}{skew}x3_l5 =-{strength};
""")


def turnoff_sextupoles(mad: MAD, beam: int) -> None:
    mad.send(f"""
for i, element in MADX.lhcb{beam}:iter() do
    if element.name:find("MS%.") then
        element.k2 = 0
        element.k2s = 0
    end
end
    """)


def match_tunes(mad: MAD, beam: int) -> None:
    mad.send(rf"""
local tbl = twiss {{sequence=MADX.lhcb{beam}, mapdef=4}}
match {{
  command := twiss {{sequence=MADX.lhcb{beam}, mapdef=4}},
  variables = {{ rtol=1e-6, -- 1 ppm
    {{ var = 'MADX.dqx_b{beam}_op', name='dQx.b{beam}_op' }},
    {{ var = 'MADX.dqy_b{beam}_op', name='dQy.b{beam}_op' }},
  }},
  equalities = {{
    {{ expr = \t -> math.abs(t.q1)-62.28, name='q1' }},
    {{ expr = \t -> math.abs(t.q2)-60.31, name='q2' }},
  }},
  objective = {{ fmin=1e-7 }},
}}
local tbl = twiss {{sequence=MADX.lhcb{beam}, mapdef=4}}
print("End of matching: ", tbl.q1, tbl.q2, tbl.x[1], tbl.y[1])
py:send("match complete")
""")
    assert mad.receive() == "match complete", "Error in matching tunes"




def run_twiss_rdts(
    beam: int, rdts: list[str], order: int, is_skew: bool
) -> tfs.TfsDataFrame:
    assert beam in [1, 2] and isinstance(
        beam, int
    ), "Beam must be 1 or 2"  # Check beam is an int
    rdt_order = get_max_rdt_order(rdts)
    with MAD() as mad:
        initialise_model(mad, beam, order, is_skew)
        observe_BPMs(mad, beam)
        rdts = mad.quote_strings(rdts)
        mad["twiss_result", "twiss_mflw"] = mad.twiss(
            sequence=f"MADX.lhcb{beam}",
            coupling=True,
            mapdef=rdt_order + 2,
            observe=1,
            trkrdt=rdts,
        )
        df: tfs.TfsDataFrame = mad.twiss_result.to_df()
    return df


def get_twiss_elements(beam: int, order: int, is_skew: bool) -> tfs.TfsDataFrame:
    with MAD() as mad:
        initialise_model(mad, beam, order, is_skew)
        mad.send(f"""
twiss_elements = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true}}
        """)
        add_strengths_to_twiss(mad, "twiss_elements")
        df: tfs.TfsDataFrame = mad.twiss_elements.to_df()
    return df


def read_madng_tfs(file_path: Path, columns: list = None) -> tfs.TfsDataFrame:
    with MAD() as mad:
        mad.send(f"mtbl = mtable:read('{file_path}')")
        df = mad.mtbl.to_df(columns=columns)
    return df


def write_tbt_file(
    beam: int,
    order: int,
    is_skew: bool,
    kick_amp: float = 1e-3,
) -> pd.DataFrame:
    tbt_path = DATA_DIR / get_tbt_name(beam, order, is_skew)
    tfs_path = DATA_DIR / get_tbt_name(beam, order, is_skew, sdds=False)
    print(
        f"Running tracking for beam {beam} over {NTURNS} turns with order {order} and skew {is_skew}"
    )
    with MAD() as mad:
        initialise_model(mad, beam, order, is_skew, kick_amp=kick_amp)
        observe_BPMs(mad, beam)
        mad.send(f"""
local t0 = os.clock()
mtbl = track {{sequence=MADX.lhcb{beam}, nturn={NTURNS}, X0=X0}}
print("NG Runtime: ", os.clock() - t0)
mtbl:write("{tfs_path}")
        """)
        df = mad.mtbl.to_df()
    tbt_data = madng.read_tbt(df)
    tbt.write(tbt_path, tbt_data)

def save_analytical_model(
    df: tfs.TfsDataFrame, beam: int, order: int, is_skew: bool
) -> None:
    save_cpx_model(df, MODEL_ANALYTICAL_PREFIX, beam, order, is_skew, df.columns)


def save_ng_model(df: tfs.TfsDataFrame, beam: int, order: int, is_skew: bool) -> None:
    rdt_columns = [col.upper() for col in df.headers["TRKRDT"]]
    save_cpx_model(df, MODEL_NG_PREFIX, beam, order, is_skew, rdt_columns)


def save_cpx_model(
    df: tfs.TfsDataFrame,
    prefix: str,
    beam: int,
    order: int,
    is_skew: bool,
    rdt_columns: list[str],
) -> None:
    file_ext = get_file_ext(beam, order, is_skew)
    outfile = DATA_DIR / f"{prefix}_{file_ext}.tfs"
    print(
        f"Saving {prefix.replace('_', ' ')}, with {len(rdt_columns)} rdts, to {outfile}"
    )

    for rdt in rdt_columns:
        df[rdt + "REAL"] = df[rdt].apply(lambda x: x.real)
        df[rdt + "IMAG"] = df[rdt].apply(lambda x: x.imag)

    out_columns = [col for rdt in rdt_columns for col in (rdt + "REAL", rdt + "IMAG")]
    tfs.write(outfile, df[out_columns], save_index="NAME")


def save_x_model(
    dfs: dict[str, tfs.TfsDataFrame], beam: int, order: int, is_skew: bool
) -> None:
    file_ext = get_file_ext(beam, order, is_skew)
    outfile = DATA_DIR / f"{MODEL_X_PREFIX}_{file_ext}.tfs"
    rdts = list(dfs.keys())
    print(f"Saving model, with {len(rdts)} rdts, to {outfile}")
    out_dict = {}
    for reim in ["REAL", "IMAG"]:
        for rdt, df in dfs.items():
            out_dict[rdt.upper() + reim] = df[reim]
    out_df = pd.DataFrame(out_dict)
    tfs.write(outfile, out_df, save_index="NAME")
