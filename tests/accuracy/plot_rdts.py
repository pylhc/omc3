import matplotlib.pyplot as plt
import numpy as np
import tfs

from tests.inputs.lhc_rdts.rdt_functions import get_rdts, to_ng_rdts, get_file_ext
from tests.inputs.lhc_rdts.rdt_constants import DATA_DIR, MODEL_NG_PREFIX, MODEL_X_PREFIX, MODEL_ANALYTICAL_PREFIX

order = 3   
is_skew = True
beam = 2

# Loop through the all the rdts
rdts = [rdt.upper() for rdt in get_rdts(order, is_skew)]
ng_rdts = to_ng_rdts(rdts)

file_ext = get_file_ext(beam, order, is_skew)
ng_df = tfs.read(DATA_DIR / f"{MODEL_NG_PREFIX}_{file_ext}.tfs", index="NAME")
x_df = tfs.read(DATA_DIR / f"{MODEL_X_PREFIX}_{file_ext}.tfs", index="NAME")
ana_df = tfs.read(DATA_DIR / f"{MODEL_ANALYTICAL_PREFIX}_{file_ext}.tfs", index="NAME")

for rdt in rdts:
    ng_rdt = rdt.split("_")[0]
    
    ng_cpx = ng_df[f"{ng_rdt}REAL"] + 1j * ng_df[f"{ng_rdt}IMAG"]
    x_cpx = x_df[f"{rdt}REAL"] + 1j * x_df[f"{rdt}IMAG"]
    ana_cpx = ana_df[f"{ng_rdt}REAL"] + 1j * ana_df[f"{ng_rdt}IMAG"]

    ng_amp = np.abs(ng_cpx)
    x_amp = np.abs(x_cpx)
    ana_amp = np.abs(ana_cpx)
    diff = ng_amp - x_amp
    diff[ng_amp.abs() > 1] = diff[ng_amp.abs() > 1] / ng_amp[ng_amp.abs() > 1]
    diff_2 = ana_amp - x_amp
    diff_2[x_amp.abs() > 1] = diff_2[x_amp.abs() > 1] / x_amp[x_amp.abs() > 1]
    print(f"{rdt} max diff with NG: {diff.abs().max()}")
    print(f"{rdt} max diff with Analytical: {diff_2.abs().max()}")

    ng_phase = np.angle(ng_cpx)
    x_phase = np.angle(x_cpx)
    ana_phase = np.angle(ana_cpx)
    diff_phase = np.angle(ng_cpx / x_cpx)
    diff_phase_2 = np.angle(ana_cpx / x_cpx)
    


    print(f"{rdt} max phase diff with NG: {np.max(np.abs(diff_phase))}")
    print(f"{rdt} max phase diff with Analytical: {np.max(np.abs(diff_phase_2))}")

    if np.max(np.abs(diff_phase_2)) > 5e-2:
        plt.figure()
        plt.title(f"{rdt} Amplitude")
        plt.plot(ng_amp.to_numpy(), label="NG")
        plt.plot(x_amp.to_numpy(), label="X")
        plt.plot(ana_amp.to_numpy(), label="Analytical")
        plt.legend()
        plt.figure()
        plt.title(f"{rdt} Phase")
        plt.plot(ng_phase, label="NG")
        plt.plot(x_phase, label="X")
        plt.plot(ana_phase, label="Analytical")
        plt.legend()
        plt.figure()
        plt.title(f"{rdt} Diff")
        plt.plot(diff.to_numpy(), label="Diff NG")
        plt.plot(diff_2.to_numpy(), label="Diff Analytical")
        plt.legend()
        plt.figure()
        plt.title(f"{rdt} Phase Diff")
        plt.plot(diff_phase, label="Diff Phase NG")
        plt.plot(diff_phase_2, label="Diff Phase Analytical")
        plt.legend()

        for reim in ["REAL", "IMAG"]:
            assert f"{ng_rdt}{reim}" in ng_df.columns
            assert f"{rdt}{reim}" in x_df.columns

            ng_reim  = ng_df[f"{ng_rdt}{reim}"]
            x_reim   = x_df[f"{rdt}{reim}"]
            ana_reim = ana_df[f"{ng_rdt}{reim}"]
            diff = x_reim - ng_reim
            greater = ng_reim.abs() > 1
            diff[greater] = diff[greater] / ng_reim[greater]
            plt.figure()
            plt.title(f"{rdt} {reim}")
            plt.plot(ng_reim.to_numpy(), label="NG")
            plt.plot(x_reim.to_numpy(), label="X")
            plt.plot(ana_reim.to_numpy(), label="Analytical")
            plt.legend()
            plt.figure()    
            plt.title(f"{rdt} {reim} Diff")
            plt.plot(diff[greater].to_numpy(), label="Diff")
        plt.show()
            

""" F1200_XREAL max diff: 0.07985280236799996
F1200_XIMAG max diff: 0.0784654353384
F3000_XREAL max diff: 0.04841745620800009
F3000_XIMAG max diff: 0.052798811684000024
F1002_XREAL max diff: 0.042410175347000034
F1002_XIMAG max diff: 0.04475298336200001
F1020_XREAL max diff: 0.13928339189
F1020_XIMAG max diff: 0.1534329870075938
F0111_YREAL max diff: 0.025894905688000014
F0111_YIMAG max diff: 0.029416003747999964
F0120_YREAL max diff: 0.01061955100949572
F0120_YIMAG max diff: 0.013071424369999973
F1011_YREAL max diff: 0.023007179801999955
F1011_YIMAG max diff: 0.03066729492300002
F1020_YREAL max diff: 0.06406624055198544
F1020_YIMAG max diff: 0.07172148195418034 """
