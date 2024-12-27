import numpy as np
import matplotlib.pyplot as plt
from preCompQDMatrix import preCompQDMatrix
#=========================================
# plot two figures of privacy budget allocation schemes
# for satisfying the desired TPL
# 05-Dec-2017 author: Yang Cao
#=========================================

T = 30

# initialize random TMs
n = 10
m = np.abs(np.random.normal(1, 1, (n, n)))
di = np.sum(m, axis=1)
TM_B = m / di[:, np.newaxis]

m = np.abs(np.random.normal(2, 1, (n, n)))
di = np.sum(m, axis=1)
TM_F = m / di[:, np.newaxis]

# precomputation
a1 = 0
an = 100
EspMatrix_B, qM_B, dM_B, QDplusInd_B = preCompQDMatrix(TM_B)
aArrMax_B, qArrMax_B, dArrMax_B = genLFunc(a1, an, EspMatrix_B, qM_B, dM_B)

a1 = 0
an = 100
EspMatrix_F, qM_F, dM_F, QDplusInd_F = preCompQDMatrix(TM_F)
aArrMax_F, qArrMax_F, dArrMax_F = genLFunc(a1, an, EspMatrix_F, qM_F, dM_F)

# alloc budget by upper bound
a = 1
e = allocEspByUpperBound(a, TM_B, TM_F)
eArr = np.ones(T) * e

print('alloc budget by upper bound')
printTPL = 1
plotTPL(eArr, TM_B, TM_F, printTPL)

# alloc budget by quantification
a = 1
e_s, e_mid, e_ = allocEspByQuantify(a, TM_B, TM_F)
eArr_mid = np.ones(T - 2) * e_mid
eArr = np.concatenate(([e_s], eArr_mid, [e_]))

print('alloc budget by quantification')
printTPL = 1
plotTPL(eArr, TM_B, TM_F, printTPL)

