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
from tools import *
from calcPLbyPreComp import calcPLbyPreComp


def genLFunc(a1, an, EspMatrix, qM, dM):
    if np.all(np.isnan(EspMatrix)):
        # 如果 EspMatrix 全是 NaN，直接返回空数组
        aArrMax = np.array([])
        qArrMax = np.array([])
        dArrMax = np.array([])
    else:
        if an < a1:
            raise ValueError('error: an < a1')

        if a1 == 0:
            a1 = 0.0001

        # 计算 a1 和 an 对应的 maxBPL、q 和 d
        maxBPL1, q1, d1 = calcPLbyPreComp(a1, EspMatrix, qM, dM)
        maxBPLn, qn, dn = calcPLbyPreComp(an, EspMatrix, qM, dM)

        # 计算 k
        with np.errstate(divide='ignore', invalid='ignore'):
            k = (qn + d1 - q1 - dn) / (q1 * dn - qn * d1)

        # 判断是否需要分段
        if (abs(a1 - an) <= np.finfo(float).eps or  # 相同 a，一个函数
                abs(maxBPLn - np.log((q1 * (np.exp(an) - 1) + 1) / (d1 * (np.exp(an) - 1) + 1))) <= np.finfo(
                    float).eps * 10):  # 两个函数在 an 处相交
            aArrMax = np.array([an])
            qArrMax = np.array([q1])
            dArrMax = np.array([d1])
        elif (abs(maxBPL1 - np.log((qn * (np.exp(a1) - 1) + 1) / (dn * (np.exp(a1) - 1) + 1))) <= np.finfo(
                float).eps * 10 or  # 两个函数在 a1 处相交
              k <= 0):  # 两个函数在 a > 0 处没有交点
            aArrMax = np.array([an])
            qArrMax = np.array([qn])
            dArrMax = np.array([dn])
        else:
            # 递归分段处理
            ak = np.log(k + 1)
            aArr_k, qArr_k, dArr_k = genLFunc(a1, ak, EspMatrix, qM, dM)
            aArr_n, qArr_n, dArr_n = genLFunc(ak, an, EspMatrix, qM, dM)
            aArrMax = np.concatenate((aArr_k, aArr_n))
            qArrMax = np.concatenate((qArr_k, qArr_n))
            dArrMax = np.concatenate((dArr_k, dArr_n))

    return aArrMax, qArrMax, dArrMax
