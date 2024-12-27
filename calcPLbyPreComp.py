# Converted from MATLAB
#=========================================
# this function calculate the incremental privacy leakag, i.e. L(a), by
# pre-computating the common results
# 05-Dec-2017 author: Yang Cao 
#-----------------inputs-----------------
# EspMatrix: n(n-1)*n matrix, contains transition points
# qM: n(n-1)*n matrix, contains q w.r.t. the corresponding transition point
# dM: n(n-1)*n matrix, contains d w.r.t. the corresponding transition point
#-----------------outputs-----------------
# maxPL: maximum incremental privacy leakage
# q,d: the ones satisfying theorem 4 in our ICDE/TKDE paper.
#=========================================


import numpy as np

def calcPLbyPreComp(a, EspMatrix, qM, dM):
    qdIdx = EspMatrix > a
    jA = np.sum(qdIdx, axis=1)
    rLen = EspMatrix.shape[0]
    iA = np.arange(1, rLen + 1)
    id = (jA - 1) * rLen + iA
    id = id[id > 0]  # in case q==d
    qArr = qM[id]
    dArr = dM[id]

    bplArr = np.log((qArr * (np.exp(a) - 1) + 1) / (dArr * (np.exp(a) - 1) + 1))
    maxPL = np.max(bplArr)
    idx = np.argmax(bplArr)
    q = qArr[idx]
    d = dArr[idx]

    return maxPL, q, d

