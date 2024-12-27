# Converted from MATLAB
# qM, dM, QDplusInd are optional, they can be pre-computed by preCompQDMatrix(TM)

#=========================================
# this function precompute some results that are common for each time point
# w.r.t. the same transition matrix.
# 05-Dec-2017 author: Yang Cao 
#-----------------inputs-----------------
# TM: transition matrix
# e: privacy  budget at each time
# qM: n(n-1)*n matrix, matrix version of "qArr", contains q w.r.t. the corresponding transition point 
# dM: n(n-1)*n matrix, matrix version of "dArr", contains d w.r.t. the corresponding transition point
# QDplusInd: n(n-1)*n matrix, contains 0 or 1 means whether the position exist a transition points
#-----------------outputs-----------------
# maxSup:  supremum of privacy leakage (BPL or FPL)
# q_sup : the value of q w.r.t. such privacy leakage
# d_sup : the value of d w.r.t. such privacy leakage
#=========================================


import numpy as np
from itertools import combinations

def find_sup(TM, e, qM=None, dM=None, QDplusInd=None):
    if qM is None or dM is None or QDplusInd is None:
        n = TM.shape[0]
        pairs = np.array(list(combinations(range(1, n + 1), 2)))
        pairs = np.vstack((pairs, pairs[:, ::-1]))

        QM = TM[pairs[:, 0] - 1, :]
        DM = TM[pairs[:, 1] - 1, :]
        QMs, DMs = sort_ratio_m1_m2_des(QM, DM)

        QDplusInd = QMs > DMs
        qM = np.cumsum(QMs, axis=1)
        dM = np.cumsum(DMs, axis=1)

    qs = qM[QDplusInd]
    ds = dM[QDplusInd]

    rNums = qs.size

    if rNums == 0:
        maxSup = e
        q_sup = np.nan
        d_sup = np.nan
    else:
        maxSup = -1
        for i in range(rNums):
            sup = theorem10(qs[i], ds[i], e)
            if maxSup < sup:
                maxSup = sup
                q_sup = qs[i]
                d_sup = ds[i]

    return maxSup, q_sup, d_sup

