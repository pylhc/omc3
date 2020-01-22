import numpy as np
#from utils import logging_tools

#LOG = logging_tools.get_logger(__name__)


def coupling_from_r_matrix(model):
    """ Returns the coupling Resonance driving terms F1001 and F1010.

    Warnings:
        Changes sign of the real part of the RDTs compared to [#FranchiAnalyticformulasrapid2017]_
        to be consistent with the RDT calculations from [#CalagaBetatroncouplingMerging2005]_.

    Args:
        model:  Model to be used by Coupling calculation from C-matrix
    """
    res = model.loc[:, ["S"]]
    denom = 4 * model.loc[:, "BETX"].values * model.loc[:, "BETY"].values
    nbpms = model.index.size
    gas, rs, igbs = np.zeros((nbpms, 2, 2)), np.zeros((nbpms, 2, 2)), np.zeros((nbpms, 2, 2))

    gas[:, 0, 0] = 1
    gas[:, 1, 0] = model.loc[:, "ALFX"]
    gas[:, 1, 1] = model.loc[:, "BETX"]

    rs[:, 0, 0] = model.loc[:, "R22"]
    rs[:, 0, 1] = -model.loc[:, "R21"]
    rs[:, 1, 0] = -model.loc[:, "R12"]
    rs[:, 1, 1] = model.loc[:, "R11"]

    igbs[:, 0, 0] = model.loc[:, "BETY"]
    igbs[:, 1, 0] = -model.loc[:, "ALFY"]
    igbs[:, 1, 1] = 1

    cs = np.einsum("kij,kjl,kln->kin", gas, rs, igbs)
    cs2 = np.einsum("ki,kij->kj", np.ones((nbpms, 2)) * np.array([1j, -1]), cs)
    res["F1001"] = (cs2[:, 0] - 1j * cs2[:, 1]) / denom
    res["F1010"] = (cs2[:, 0] + 1j * cs2[:, 1]) / denom
    print(f"  Average coupling amplitude |F1001|: {np.mean(np.abs(res.loc[:, 'F1001']))}")
    print(f"  Average coupling amplitude |F1010|: {np.mean(np.abs(res.loc[:, 'F1010']))}")
    return res


def get_coupling(tw):
    """ Returns the coupling term.

            .. warning::
                This function changes sign of the real part of the RDTs compared to
                [#FranchiAnalyticformulasrapid2017]_ to be consistent with the RDT
                calculations from [#CalagaBetatroncouplingMerging2005]_ .


                Calculates C matrix and Coupling and Gamma from it.
        See [#CalagaBetatroncouplingMerging2005]_

            Args:
                model:  Model to be used by cmatrix calculation
            """

    res=tw.loc[:, ["S"]]

    j = np.array([[0., 1.],
                  [-1., 0.]])
    rs = np.reshape(tw.as_matrix(columns=["R11", "R12", "R21", "R22"]), (len(tw), 2, 2))
    cs = np.einsum("ij,kjn,no->kio",
                   -j, np.transpose(rs, axes=(0, 2, 1)), j)
    cs = np.einsum("k,kij->kij", (1 / np.sqrt(1 + np.linalg.det(rs))), cs)

    g11a = 1 / np.sqrt(tw.loc[:, "BETX"])
    g12a = np.zeros(len(tw))
    g21a = tw.loc[:, "ALFX"] / np.sqrt(tw.loc[:, "BETX"])
    g22a = np.sqrt(tw.loc[:, "BETX"])
    gas = np.reshape(np.array([g11a, g12a, g21a, g22a]).T, (len(tw), 2, 2))

    ig11b = np.sqrt(tw.loc[:, "BETY"])
    ig12b = np.zeros(len(tw))
    ig21b = -tw.loc[:, "ALFY"] / np.sqrt(tw.loc[:, "BETY"])
    ig22b = 1. / np.sqrt(tw.loc[:, "BETY"])
    igbs = np.reshape(np.array([ig11b, ig12b, ig21b, ig22b]).T, (len(tw), 2, 2))
    cs = np.einsum("kij,kjl,kln->kin", gas, cs, igbs)
    gammas = np.sqrt(1 - np.linalg.det(cs))

    res.loc[:, "GAMMA_C"] = gammas
    res.loc[:, "F1001_C"] = ((cs[:, 0, 0] + cs[:, 1, 1]) * 1j +
                             (cs[:, 0, 1] - cs[:, 1, 0])) / 4 / gammas
    res.loc[:, "F1010_C"] = ((cs[:, 0, 0] - cs[:, 1, 1]) * 1j +
                             (-cs[:, 0, 1]) - cs[:, 1, 0]) / 4 / gammas

    print(f"  Average coupling amplitude |F1001|: {np.mean(np.abs(res.loc[:, 'F1001_C']))}")
    print(f"  Average coupling amplitude |F1010|: {np.mean(np.abs(res.loc[:, 'F1010_C']))}")
    print(f"  Average gamma: {np.mean(np.abs(res.loc[:, 'GAMMA_C']))}")

    res_df = res.loc[:, ['S', 'F1001_C', 'F1010_C']]
    return res_df.rename(columns=lambda x: x.replace("_C", ""))



# if __name__ == '__main__':
#     import tfs
#     coupling_from_r_matrix(tfs.read("/afs/cern.ch/work/l/lmalina/2019-09-16/LHCB1/Results/Getllm_for_coupling/twiss_cor.dat"))