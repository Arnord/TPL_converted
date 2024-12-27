#=========================================
# plot two figures of privacy budget allocation schemes
# for stisfying the desired TPL
# 05-Dec-2017 author: Yang Cao
#=========================================
import sys
sys.path.append('tools/')

import numpy as np
import matplotlib.pyplot as plt

T = 30

## initialize random TMs
n = 10
m = np.abs(np.random.normal(1, 1, (n, n)))
di = np.sum(m, axis=1)
TM_B = m / di[:, np.newaxis]

m = np.abs(np.random.normal(2, 1, (n, n)))
di = np.sum(m, axis=1)
TM_F = m / di[:, np.newaxis]

## precomputation
a1 = 0
an = 100
EspMatrix_B, qM_B, dM_B, QDplusInd_B = preCompQDMatrix(TM_B)
aArrMax_B, qArrMax_B, dArrMax_B = genLFunc(a1, an, EspMatrix_B, qM_B, dM_B)

a1 = 0
an = 100
EspMatrix_F, qM_F, dM_F, QDplusInd_F = preCompQDMatrix(TM_F)
aArrMax_F, qArrMax_F, dArrMax_F = genLFunc(a1, an, EspMatrix_F, qM_F, dM_F)

## alloc budget by upper bound
a = 1
e = allocEspByUpperBound(a, TM_B, TM_F)
eArr = np.ones(T) * e

print('\033[94malloc budget by upper bound\033[0m')
printTPL = 1
plotTPL(eArr, TM_B, TM_F, printTPL)

## alloc budget by quantification
a = 1
e_s, e_mid, e_end = allocEspByQuantify(a, TM_B, TM_F)
eArr_mid = np.ones(T-2) * e_mid
eArr = np.concatenate(([e_s], eArr_mid, [e_end]))

print('\033[94malloc budget by quantification\033[0m')
printTPL = 1
plotTPL(eArr, TM_B, TM_F, printTPL)
