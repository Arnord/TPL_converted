# =========================================
# This recursive function obtain L(a) function.
# 05-Dec-2017 author: Yang Cao
# -----------------inputs-----------------
# a1 and an: specify the range, i.e., definition domain, of such L(a) function
# EspMatrix: n(n-1)*n matrix,  matrix version of "aArr", contains transition points. EspMatrix = [\alpha_1, ...,  \alpha_{k-1},NaN,..]
# qM: n(n-1)*n matrix, matrix version of "qArr", contains q w.r.t. the corresponding transition point
# dM: n(n-1)*n matrix, matrix version of "dArr", contains d w.r.t. the corresponding transition point
# -----------------outputs-----------------
# aArrMax: vector, contains transition points,defines domains on each segment of L(a)
# qArrMax: vector, contains values of q, defines parameters of L(a)
# dArrMax: vector, contains values of d, defines parameters of L(a)
# =========================================

import numpy as np
from numpy import log, exp


def genLFunc(a1, an, EspMatrix, qM, dM):
    eps = np.finfo(float).eps  # Machine epsilon in Python

    if np.all(np.isnan(EspMatrix)):
        # if uniform matrix
        # np.all(np.isnan(EspMatrix)) will be True
        return [], [], []

    if an < a1:
        raise ValueError('error: an<a1')

    if a1 == 0:
        a1 = 0.0001

    maxBPL1, q1, d1 = calcPLbyPreComp(a1, EspMatrix, qM, dM)
    maxBPLn, qn, dn = calcPLbyPreComp(an, EspMatrix, qM, dM)

    k = (qn + d1 - q1 - dn) / (q1 * dn - qn * d1)

    # n = EspMatrix.shape[0]
    # aArrMax = np.zeros(n)
    # qArrMax = np.zeros(n)
    # dArrMax = np.zeros(n)

    if (abs(a1 - an) <= eps or  # same a, one func
            abs(maxBPLn - log(
                (q1 * (exp(an) - 1) + 1) / (d1 * (exp(an) - 1) + 1))) <= eps * 10):  # two func intersect at an
        aArrMax = [an]
        qArrMax = [q1]
        dArrMax = [d1]
    elif (abs(maxBPL1 - log(
            (qn * (exp(a1) - 1) + 1) / (dn * (exp(a1) - 1) + 1))) <= eps * 10 or  # two func intersect at a1
          k <= 0):  # two func without intersection at a>0 (~ the same func)
        aArrMax = [an]
        qArrMax = [qn]
        dArrMax = [dn]
    else:
        ak = log(k + 1)
        aArr_k, qArr_k, dArr_k = genLFunc(a1, ak, EspMatrix, qM, dM)
        aArr_n, qArr_n, dArr_n = genLFunc(ak, an, EspMatrix, qM, dM)
        aArrMax = aArr_k + aArr_n
        qArrMax = qArr_k + qArr_n
        dArrMax = dArr_k + dArr_n

    return aArrMax, qArrMax, dArrMax
