# Converted from MATLAB
#=========================================
# This function calculate the incremental privacy leakag, i.e. L(a), by L(a) function. 
#
# 05-Dec-2017 author: Yang Cao 
#-----------------inputs-----------------
# a: the previous BPL or the next FPL
# aArrMax: vector, contains transition points,defines domains on each segment of L(a)
# qArrMax: vector, contains values of q, defines parameters of L(a)
# dArrMax: vector, contains values of d, defines parameters of L(a)
#-----------------outputs-----------------
# maxPL: maximum incremental privacy leakage L(a)
# q,d: two scalars that satisfy Theorem 4 in our TKDE paper
#=========================================

# function[maxBPL, q, d] = calcPLbyFunc(a, aArrMax, qArrMax, dArrMax)
#
# % aArrMax = evalin('base', 'aArrMax');
# % qArrMax = evalin('base', 'qArrMax');
# % dArrMax = evalin('base', 'dArrMax');
#
# if isempty(aArrMax)
#     maxBPL = 0;
#     q = 0;
#     d = 0;
# else
#     if a > aArrMax(end)
#         error('a should be in the range of aArrMax');
#     end
#     idx = sum(aArrMax < a) + 1;
#     q = qArrMax(idx);
#     d = dArrMax(idx);
#     maxBPL = log((q * (exp(a) - 1) + 1) / (d * (exp(a) - 1) + 1));
#
# end

import numpy as np

def calcPLbyFunc(a, aArrMax, qArrMax, dArrMax):
    if len(aArrMax) == 0:
        maxBPL = 0
        q = 0
        d = 0
    else:
        if a > aArrMax[-1]:
            raise ValueError('a should be in the range of aArrMax')

        # 找到 a 在 aArrMax 中的位置
        idx = np.sum(aArrMax < a)

        # 提取对应的 q 和 d
        q = qArrMax[idx]
        d = dArrMax[idx]

        # 计算 maxBPL
        maxBPL = np.log((q * (np.exp(a) - 1) + 1) / (d * (np.exp(a) - 1) + 1))

    return maxBPL, q, d





