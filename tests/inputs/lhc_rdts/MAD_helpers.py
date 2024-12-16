from __future__ import annotations

from pathlib import Path

import pandas as pd
import tfs
import turn_by_turn as tbt
from pymadng import MAD
from turn_by_turn import madng
from cpymad.madx import Madx

from omc3.model_creator import create_instance_and_model
from tests.inputs.lhc_rdts.omc3_helpers import (
    get_file_suffix,
    get_max_rdt_order,
    get_model_dir,
    get_tbt_name,
)
from tests.inputs.lhc_rdts.rdt_constants import (
    ACC_MODELS,
    ANALYSIS_DIR,
    DATA_DIR,
    FREQ_OUT_DIR,
    KICK_AMP,
    MODEL_ANALYTICAL_PREFIX,
    MODEL_NG_PREFIX,
    MODEL_X_PREFIX,
    NTURNS,
    OCTUPOLE_STRENGTH,
    SEXTUPOLE_STRENGTH,
)

# Ensure directories exist
ANALYSIS_DIR.mkdir(exist_ok=True)
FREQ_OUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

MADX_FILENAME = "job.create_model_nominal.madx"

def to_ng_rdts(rdts: list[str]) -> list[str]:
    return list(set([rdt.split("_")[0] for rdt in rdts]))


def create_model_dir(beam: int) -> None:
    model_dir = get_model_dir(beam)
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
    with open(model_dir / MADX_FILENAME, "r") as f:
        lines = f.readlines()
    
    # Make the sequence as beam 1 or 2
    make_madx_seq(model_dir, lines, beam)

    # Update the model by using beam 1 or 2.
    update_model_with_ng(beam)

    # If beam 2, now we need to make the sequence as beam 4 for tracking
    if beam == 2:
        make_madx_seq(model_dir, lines, beam, beam4=True)


def make_madx_seq(
    model_dir: Path, lines: list[str], beam: int, beam4: bool = False
) -> None:
    with Madx(stdout=False) as madx:
        madx.chdir(str(model_dir))
        for i, line in enumerate(lines):
            if beam4:
                if "define_nominal_beams" in line:
                    madx.input(
                        "beam, sequence=LHCB2, particle=proton, energy=450, kbunch=1, npart=1.15E11, bv=1;\n"
                    )
                    continue
                elif "acc-models-lhc/lhc.seq" in line:
                    line = line.replace(
                            "acc-models-lhc/lhc.seq", "acc-models-lhc/lhcb4.seq"
                        )
            if i < 32:
                madx.input(line)
        madx.input(
            f"""
set, format= "-.16e";
save, sequence=lhcb{beam}, file="lhcb{beam}_saved.seq", noexpr=false;
        """)

def update_model_with_ng(beam: int) -> None:
    model_dir = get_model_dir(beam)
    with MAD() as mad:
        seq_dir = -1 if beam == 2 else 1
        initialise_model(mad, beam, seq_dir=seq_dir)
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
-- Calculate the twiss parameters with coupling and observe the BPMs
! Coupling needs to be true to calculate Edwards-Teng parameters and R matrix
twiss_elements = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true}} 
-- Select everything
twiss_elements:select(nil, \ -> true)
-- Deselect the drifts
twiss_elements:deselect{{pattern="drift"}}
""").send(strength_cols)
        # MAD.gphys.melmcol(twiss_elements, str_cols)
        add_strengths_to_twiss(mad, "twiss_elements")
        mad.send(
            # True below is to make sure only selected rows are written
            f"""twiss_elements:write("{model_dir / 'twiss_elements.dat'}", cols, hnams, true)"""
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
    export_tfs_to_madx(model_dir / "twiss_ac.dat")
    export_tfs_to_madx(model_dir / "twiss_elements.dat")
    export_tfs_to_madx(model_dir / "twiss.dat")


def observe_BPMs(mad: MAD, beam: int) -> None:
    mad.send(f"""
local observed in MAD.element.flags
MADX.lhcb{beam}:deselect(observed)
MADX.lhcb{beam}:  select(observed, {{pattern="BPM"}})
    """)


def export_tfs_to_madx(tfs_file: Path) -> None:
    tfs_df = tfs.read(tfs_file)
    tfs_df = convert_tfs_to_madx(tfs_df)
    tfs.write(tfs_file, tfs_df, save_index="NAME")


def convert_tfs_to_madx(tfs_df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
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


strength_cols = ["k1l", "k2l", "k3l", "k4l", "k5l", "k1sl", "k2sl", "k3sl", "k4sl", "k5sl"]


def add_strengths_to_twiss(mad: MAD, mtable_name: str) -> None:
    mad.send(f"""
strength_cols = py:recv()
MAD.gphys.melmcol({mtable_name}, strength_cols)
    """).send(strength_cols)


def initialise_model(
    mad: MAD,
    beam: int,
    seq_dir: int = 1,
) -> None:
    assert beam in [1, 2] and isinstance(beam, int), "Beam must be 1 or 2"
    model_dir = get_model_dir(beam)
    mad.MADX.load(
        f"'{model_dir/f'lhcb{beam}_saved.seq'}'",
        f"'{model_dir/f'lhcb{beam}_saved.mad'}'",
    )
    mad.send(f"""
lhc_beam = beam {{particle="proton", energy=450}}
MADX.lhcb{beam}.beam = lhc_beam
MADX.lhcb{beam}.dir = {seq_dir}
print("Initialising model with beam:", {beam}, "dir:", MADX.lhcb{beam}.dir)
    """)
    if seq_dir == 1 and beam == 1:
        mad.send("""MADX.lhcb1:cycle("MSIA.EXIT.B1")""")

    ensure_bends_are_on(mad, beam)
    deactivate_sextupoles(mad, beam)
    add_magnet_strengths(mad)
    match_tunes(mad, beam)

MAGNET_STRENGTHS = {
    "s" : SEXTUPOLE_STRENGTH,
    "o" : OCTUPOLE_STRENGTH
}
def add_magnet_strengths(mad: MAD) -> None:
    for s_or_o, strength in MAGNET_STRENGTHS.items():
        for skew in ["", "s"]:
            mad.send(f"""
        MADX.kc{s_or_o}{skew}x3_r1 = {strength:+.16e};
        MADX.kc{s_or_o}{skew}x3_l5 = {-strength:+.16e};
        """)


def deactivate_sextupoles(mad: MAD, beam: int) -> None:
    mad.send(f"""
for i, element in MADX.lhcb{beam}:iter() do
    if element.kind == "sextupole" then
        element.k2 = 0
        element.k2s = 0
    end
end
    """)


def ensure_bends_are_on(mad: MAD, beam: int) -> None:
    mad.send(f"""
for i, element in MADX.lhcb{beam}:iter() do
    if (element.kind == "sbend" or element.kind == "rbend") and (element.angle ~= 0 and element.k0 == 0) then
        element.k0 = \s->s.angle/s.l -- restore deferred expression
    end
end
    """)


def match_tunes(mad: MAD, beam: int) -> None:
    mad.send(rf"""
local tbl = twiss {{sequence=MADX.lhcb{beam}, mapdef=4}}
print("Initial tunes: ", tbl.q1, tbl.q2, tbl.x[1], tbl.y[1])
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
print("Final tunes: ", tbl.q1, tbl.q2, tbl.x[1], tbl.y[1])
tbl:write("twiss_match.dat")
py:send("match complete")
""")
    assert mad.receive() == "match complete", "Error in matching tunes"


def run_twiss_rdts(beam: int, rdts: list[str]) -> tfs.TfsDataFrame:
    rdt_order = get_max_rdt_order(rdts)
    with MAD() as mad:
        initialise_model(mad, beam)
        observe_BPMs(mad, beam)
        mad["twiss_result", "twiss_mflw"] = mad.twiss(
            sequence=f"MADX.lhcb{beam}",
            coupling=True,
            mapdef=rdt_order + 2,
            observe=1,
            trkrdt=mad.quote_strings(rdts),
        )
        df: tfs.TfsDataFrame = mad.twiss_result.to_df()
    return df


def get_twiss_elements(beam: int) -> tfs.TfsDataFrame:
    with MAD() as mad:
        initialise_model(mad, beam)
        mad.send(f"""
twiss_elements = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true}}
        """)
        add_strengths_to_twiss(mad, "twiss_elements")
        df = mad.twiss_elements.to_df()
    return df

def write_tbt_file(beam: int) -> pd.DataFrame:
    tbt_path = DATA_DIR / get_tbt_name(beam,)
    tfs_path = DATA_DIR / get_tbt_name(beam, sdds=False)
    with MAD() as mad:
        initialise_model(mad, beam)
        observe_BPMs(mad, beam)
        # Octupolar resonances are harder to observe with only 1000 turns 
        # so we need to increase the kick amplitude for order 3
        mad.send(f"""
local t0 = os.clock()
local kick_amp = py:recv()
local X0 = {{x=kick_amp, y=-kick_amp, px=0, py=0, t=0, pt=0}}
print("Running MAD-NG track with kick amplitude: ", kick_amp)

mtbl = track {{sequence=MADX.lhcb{beam}, nturn={NTURNS}, X0=X0}}
print("NG Runtime: ", os.clock() - t0)
        """).send(KICK_AMP)
        df = mad.mtbl.to_df(columns=["name", "x", "y", "eidx", "turn", "id"])
    tfs.write(tfs_path, df)
    tbt_data = madng.read_tbt(tfs_path)
    tbt.write(tbt_path, tbt_data)

def save_analytical_model(df: tfs.TfsDataFrame, beam: int) -> None:
    save_cpx_model(df, MODEL_ANALYTICAL_PREFIX, beam, df.columns)


def save_ng_model(df: tfs.TfsDataFrame, beam: int) -> None:
    rdt_columns = [col.upper() for col in df.headers["TRKRDT"]]
    save_cpx_model(df, MODEL_NG_PREFIX, beam, rdt_columns)


def save_cpx_model(
    df: tfs.TfsDataFrame,
    prefix: str,
    beam: int,
    rdt_columns: list[str],
) -> None:
    assert len(rdt_columns) > 0, "No RDT columns to save"
    assert beam in [1, 2], "Beam must be 1 or 2"

    file_ext = get_file_suffix(beam)
    outfile = DATA_DIR / f"{prefix}_{file_ext}.tfs"
    print(
        f"Saving {prefix.replace('_', ' ')}, with {len(rdt_columns)} rdts, to {outfile}"
    )

    new_df_dict = {
        "NAME": df.index,
    }
    for rdt in rdt_columns:
        new_df_dict[rdt + "REAL"] = df[rdt].apply(lambda x: x.real)
        new_df_dict[rdt + "IMAG"] = df[rdt].apply(lambda x: x.imag)

    tfs.write(outfile, tfs.TfsDataFrame(new_df_dict))


def save_x_model(dfs: dict[str, tfs.TfsDataFrame], beam: int) -> None:
    file_ext = get_file_suffix(beam)
    outfile = DATA_DIR / f"{MODEL_X_PREFIX}_{file_ext}.tfs"
    rdts = list(dfs.keys())
    print(f"Saving model, with {len(rdts)} rdts, to {outfile}")
    out_dict = {}
    for column_type in ["REAL", "IMAG", "AMP", "PHASE"]:
        for rdt, df in dfs.items():
            out_dict[rdt.upper() + column_type] = df[column_type]
    out_df = tfs.TfsDataFrame(out_dict)
    tfs.write(outfile, out_df, save_index="NAME")
