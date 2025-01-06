# %=========================================
# % test findSup. comparing the results of findSup with the one calculated
# % step by step until the BPL (or FPL) is stable.
# %
# % There are different test cases.



import numpy as np
from tools import *
from preCompQDMatrix import preCompQDMatrix
from genLFunc import genLFunc
from findSup import findSup
from calcPLbyFunc import calcPLbyFunc
import matplotlib.pyplot as plt

# Test case 1 of sup (q!=0/1 d!=0/1)
n = 5
m = np.abs(np.random.normal(1, 1, (n, n)))
di = np.sum(m, axis=1)
TM = m / di[:, np.newaxis]

e = 0.1

# FindSup
a1 = 0
an = 10
EspMatrix, qM, dM, QDplusInd = preCompQDMatrix(TM)
aArrMax, qArrMax, dArrMax = genLFunc(a1, an, EspMatrix, qM, dM)

# Timing the execution
import time
start_time = time.time()
maxSup, q_sup, d_sup = findSup(TM, e, qM, dM, QDplusInd)
end_time = time.time()

# Print results
# print(maxSup)
# print(q_sup)
# print(d_sup)

# Calculate by TM
T = 200
aArr = np.zeros(T)
aArr[0] = e
for i in range(1, T):
    # print(f't={i}')
    maxBPL, q, d = calcPLbyFunc(aArr[i-1], aArrMax, qArrMax, dArrMax)
    aArr[i] = maxBPL + e

aArr[T-1]
plt.plot(aArr)  # Uncomment to plot if needed
plt.show()

