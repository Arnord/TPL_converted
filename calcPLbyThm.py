# Converted from MATLAB
#=========================================
# This function calculate the incremental privacy leakag, i.e. L(a), by Theorem 4 and Corollary 2 (in our TKDE paper). 
# note1: this function only calc L(a), while BPL or FPL = L(a)+ eps_t.
# note2: some precision problem  could happen when a is large (e.g. a>30,
# because exp(a)),but calcBPLbyPreComp.m and calcBPLbyFunc.m is robust to precision problem
# 04-Dec-2017 author: Yang Cao 
#-----------------inputs-----------------
# TM: transition matrix
# a: previouts BPL or the next FPL
#-----------------outputs-----------------
# maxPL: maximum incremental privacy leakage L(a)
# q,d: two scalars that satisfy Theorem 4 in our TKDE paper
#=========================================


import numpy as np
from itertools import combinations

def calcPLbyThm(TM, a):
    # transition matrix M, previous BPL a
    n = TM.shape[0]
    pairs = np.array(list(combinations(range(1, n + 1), 2)))
    pairs = np.vstack((pairs, pairs[:, ::-1]))

    QM = TM[pairs[:, 0] - 1, :]
    DM = TM[pairs[:, 1] - 1, :]

    QDplusInd = QM > DM
    QM = QM * QDplusInd
    DM = DM * QDplusInd

    update = True
    while update:
        sizeRemain = np.sum(QDplusInd)
        valArr = (np.sum(QM, axis=1) * (np.exp(a) - 1) + 1) / (np.sum(DM, axis=1) * (np.exp(a) - 1) + 1)
        QDplusIndNew = QM / DM > valArr[:, np.newaxis]
        if sizeRemain == np.sum(QDplusIndNew):
            update = False
        else:
            idx = np.where(QDplusIndNew != QDplusInd)
            QM[idx] = 0
            DM[idx] = 0
            QDplusInd = QDplusIndNew

    maxVal = np.max(valArr)
    I = np.argmax(valArr)
    maxPL = np.log(maxVal)

    qArr = np.sum(QM, axis=1)
    q = qArr[I]

    dArr = np.sum(DM, axis=1)
    d = dArr[I]

    return maxPL, q, d

