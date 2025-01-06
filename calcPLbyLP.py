# Converted from MATLAB
#=========================================
# this function calculate the incremental privacy leakag, i.e. L(a), by Linear
# programming. note that this function only calc L(a), while BPL or FPL = L(a)+ eps_t
#-----------------inputs-----------------
# TM: transition matrix
# a: previouts BPL or the next FPL
# method: string, 'gurobi' or 'cplex', or 'matlab'.
#         gurobi works for ONLY matlab2016b, prolematic for matlab2017b
#-----------------outputs-----------------
# maxPL: maximum incremental privacy leakage L(a)
# maxPL_ij: 2*1 array, ith row is the q vector, jth row is the d vector
#=========================================


import numpy as np
from itertools import combinations
from tools import *

def calcPLbyLP(TM, a, method):
    n = TM.shape[1]  # n = size(TM, 2) same?
    maxPL = -1
    maxPL_ij = []

    # pairs = VChooseK(int16(1:n), 2)
    # pairs = [pairs; filplr(pairs)];
    pairs = list(combinations(range(1, n + 1), 2))  # 生成所有 2-组合，索引从 1 开始
    # 将 pairs 转换为 numpy 数组
    pairs = np.array(pairs)
    # 对 pairs 的每一行进行翻转
    flipped_pairs = np.fliplr(pairs)
    # 将 pairs 和 flipped_pairs 垂直拼接
    pairs = np.vstack((pairs, flipped_pairs))

    for eachPair in pairs:
        v1 = TM[eachPair[0] - 1, :].reshape(1, -1)  # Adjust for 0-based indexing
        v2 = TM[eachPair[1] - 1, :].reshape(1, -1)  # Adjust for 0-based indexing

        if method == 'cplex':
            bpl, x = calcPLbyLP_cplex(v1, v2, a)
        elif method == 'gurobi':
            bpl, x = calcPLbyLP_gu(v1, v2, a)
        elif method == 'matlab':
            bpl, x = calcPLbyLP_matlab(v1, v2, a)

        if maxPL < bpl:
            maxPL = bpl
            maxPL_ij = eachPair

    return maxPL, maxPL_ij



