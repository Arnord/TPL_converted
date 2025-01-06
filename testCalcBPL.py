#=========================================
# test correctness of calcBPL
#
# 05-Dec-2017 author: Yang Cao
#=========================================
# import sys
# sys.path.append('tools/')

import numpy as np
from time import time
# from tools import *
from calcPLbyLP import calcPLbyLP
from calcPLbyFunc import calcPLbyFunc
from calcPLbyThm import calcPLbyThm
from calcPLbyPreComp import calcPLbyPreComp
from preCompQDMatrix import preCompQDMatrix
from genLFunc import genLFunc

# Clear variables (not needed in Python as variables are scoped)

a = 0.1

# initialize transition matrix
n = 30
m = np.abs(np.random.normal(1, 1, (n, n)))
di = np.sum(m, axis=1)
TM = m / di[:, np.newaxis]  # equivalent to bsxfun(@rdivide, m, di)

# calc by cplex
start_time = time()
maxBPL_cplex = calcPLbyLP(TM, a, 'cplex')
print(f"maxBPL_cplex = {maxBPL_cplex}")
print(f"Time elapsed: {time() - start_time} seconds\n")

# calc by theorem 4
start_time = time()
maxBPL1, _, _ = calcPLbyThm(TM, a)
print(f"maxBPL1 = {maxBPL1}")
print(f"Time elapsed: {time() - start_time} seconds\n")

# calc by precomputation
EspMatrix, qM, dM, _ = preCompQDMatrix(TM)
start_time = time()
maxBPL2, _, _ = calcPLbyPreComp(a, EspMatrix, qM, dM)
print(f"maxBPL2 = {maxBPL2}")
print(f"Time elapsed: {time() - start_time} seconds\n")

# calc by function L(a)
a1 = 0
an = a
aArrMax, qArrMax, dArrMax = genLFunc(a1, an, EspMatrix, qM, dM)
start_time = time()
maxBPL3 = calcPLbyFunc(a, aArrMax, qArrMax, dArrMax)
print(f"maxBPL3 = {maxBPL3}")
print(f"Time elapsed: {time() - start_time} seconds\n")
