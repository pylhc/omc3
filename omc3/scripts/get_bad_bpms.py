"""
Scans all measurements in a list of given GUI output folders (`DATES`) and compiles a list of bad BPMs with
their given number of appearances after SVD and isolation forest.

Output will be written to `"bad_bpms.txt"`

Usage:
    1. Make sure that the measurements have the desired cleaning method applied.
    If needed rerun the measurements with the GUI.

    2. Adapt the `DATES` list at the beginning of this script accordingly

    3. a) Run this script in `OP_DATA` or
       b) `ln -s OP_DATA/Betabeat` into your working dir or
       c) change `ROOT` to a parent dir of your GUI output

"""
import pathlib
import tfs

ROOT = pathlib.Path("./Betabeat/")

DATES = [
#        "2023-08-28/",
#        "2023-08-30/",
#        "2023-08-31/",
#        "2023-09-01/",
#        "2023-09-02/",
#        "2023-09-03/",
#        "2023-09-04/",
        "2023-09-06"
]

OUTFILE = None

def main():
    global OUTFILE
    OUTFILE = open("bad_bpms.txt", "w")

    get_the_bad_bpms()

    OUTFILE.close()

def bad_bpms_per_date(date: pathlib.Path, plane: str, beam: str):
    meas = (date / f"LHCB{beam}/Measurements")
    #print(f"M: {meas}")

    measurements = [p for p in meas.iterdir() if p.is_dir()]

    def get_bad_bpms(path: pathlib.Path):
        for m in path.iterdir():
            if m.suffix == f".bad_bpms_{plane}":
                return m
        return None

    def get_iforest_bpms(path: pathlib.Path):
        for m in path.iterdir():
            if m.name == f"bad_bpms_iforest_{plane}.tfs":
                return m
        return None

    measurements_svd = [get_bad_bpms(p) for p in measurements]
    measurements_svd = [p for p in measurements_svd if p is not None]

    measurements_iforest = [get_iforest_bpms(p) for p in measurements]
    measurements_iforest = [p for p in measurements_iforest if p is not None]
    return (measurements_svd, measurements_iforest)

def get_bad_bpms_for_beam_and_plane_svd(date: str, plane: str, beam: str, bad_bpms_list):
    (measurements, measurements_iforest) = bad_bpms_per_date(ROOT / date , plane, beam)
    for m in measurements:
        appeared = []
        #print(open(m).read())
        for bpm in open(m):
            words = bpm.split()
            if words[0] in appeared:
                continue
            if "model" in words and "not" in words:
                continue
            appeared.append(words[0])
            if words[0] in bad_bpms_list:
                bad_bpms_list[words[0]] = bad_bpms_list[words[0]] + 1
            else:
                bad_bpms_list[words[0]] = 1
    return len(measurements)

def get_bad_bpms_for_beam_and_plane_iforest(date: str, plane: str, beam: str, bad_bpms_list):
    (measurements, measurements_iforest) = bad_bpms_per_date(ROOT / date , plane, beam)
    for m in measurements_iforest:
        appeared = []
        #print(open(m).read())
        bpm_tfs = tfs.read_tfs(m, index="NAME")
        for bpm in bpm_tfs.index:
            if bpm in appeared:
                continue
            appeared.append(bpm)
            if bpm in bad_bpms_list:
                bad_bpms_list[bpm] = bad_bpms_list[bpm] + 1
            else:
                bad_bpms_list[bpm] = 1
    return len(measurements_iforest)


def do_print(msg: str):
    global OUTFILE
    print(msg)
    OUTFILE.write(f"{msg}\n")

def get_the_bad_bpms():
    for plane in ["x", "y"]:
        for beam in ["1", "2"]:
            
            pl = "H" if plane == "x" else "V"
            do_print("")
            do_print(f"BEAM {beam} {pl}")
            do_print("{:20} | {} | {} | {:8}  | {:7}".format("NAME", "PLANE", "BEAM", "FREQUENCY", "FROM"))
            do_print("---------------------|-------|------|------------|-----------")
            bad_bpms_list_iforest = {}
            bad_bpms_list_svd = {}

            n_svd = 0
            n_iforest = 0
            for date in DATES:
                print(f"date {date}")
                n_svd += get_bad_bpms_for_beam_and_plane_svd(date, plane, beam, bad_bpms_list_svd)
                n_iforest += get_bad_bpms_for_beam_and_plane_iforest(date, plane, beam, bad_bpms_list_iforest)

            print(f"SVD [{n_svd} files]")
            for (k,v) in bad_bpms_list_svd.items():
                v_perc = v / n_svd * 100
                if v_perc > 75:
                    do_print(f"{k:20} | {plane:5} | {beam:4} | {v_perc:8.2f} % | SVD")
                elif v_perc > 50:
                    do_print(f"{k:20} | {plane:5} | {beam:4} | {v_perc:8.2f} % | SVD")
#                elif v_perc > 25:
#                    do_print(f"{k:20} | {plane:5} | {beam:4} | {v_perc:8.2f} % | SVD")


            print(f"IFOREST [{n_iforest} files]")
            for (k,v) in bad_bpms_list_iforest.items():
                v_perc = v / n_iforest * 100
                if v_perc > 75:
                    do_print(f"{k:20} | {plane:5} | {beam:4} | {v_perc:8.2f} % | IFOREST")
                elif v_perc > 50:
                    do_print(f"{k:20} | {plane:5} | {beam:4} | {v_perc:8.2f} % | IFOREST")
#                elif v_perc > 25:
#                    do_print(f"{k:20} | {plane:5} | {beam:4} | {v_perc:8.2f} % | IFOREST")

if __name__ == "__main__":
    main()
