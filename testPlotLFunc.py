import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import sys

# Add tools directory to path
sys.path.append('tools/')

# Clear all variables (Python doesn't need this, but keeping comment for documentation)
# Clear console (optional, system dependent)
os.system('cls' if os.name == 'nt' else 'clear')

# Initialize transition matrix
TM = np.array([[0.1, 0.2, 0.7],
               [0.3, 0.3, 0.4],
               [0.5, 0.3, 0.2]])

# Alternative random matrix generation (commented out)
# n = 5
# m = np.abs(norm.rvs(loc=1, scale=1, size=(n,n)))
# di = np.sum(m, axis=1)
# TM = m / di[:, np.newaxis]

dim = TM.shape[1]

# Assuming preCompQDMatrix is defined in tools
EspMatrix, qM, dM, QDplusInd = preCompQDMatrix(TM)

# Common for plot and fplot
maxEsp = np.max(EspMatrix[~np.isinf(EspMatrix) & ~np.isnan(EspMatrix)])
if maxEsp is not None:
    maxEsp = maxEsp + 3.5
else:
    maxEsp = 100

EspMatrix[np.isinf(EspMatrix)] = maxEsp
EspMatrix[np.isnan(EspMatrix)] = 0

# Plot
# Get (q d) pairs
qArr = qM[QDplusInd]
dArr = dM[QDplusInd]

# Get x ∈ [x_l, x_b] for each (q d) pair
x_u = EspMatrix[QDplusInd]
x_l = EspMatrix[np.roll(QDplusInd, 1, axis=1)]

plt.figure()
plotPoints = 1000
linewidth = 1
StemSpec = ':o'
plotFuncs(qArr, dArr, x_l, x_u, plotPoints, linewidth, StemSpec, 1)

a1 = 0
an = maxEsp
EspMatrix, qM, dM, QDplusInd = preCompQDMatrix(TM)
aArrMax, qArrMax, dArrMax = genLFunc(a1, an, EspMatrix, qM, dM)
x_uMax = aArrMax.T
x_lMax = np.concatenate(([a1], x_uMax))
x_lMax = x_lMax[:-1]
linewidth = 3

qArrMax = qArrMax.T
dArrMax = dArrMax.T
rNums = len(qArrMax)

# Get x data
x = np.zeros((rNums, plotPoints))
for i in range(rNums):
    x[i,:] = np.linspace(x_lMax[i], x_uMax[i], plotPoints)

y = np.log((qArrMax[:,np.newaxis] * (np.exp(x) - 1) + 1) /
           (dArrMax[:,np.newaxis] * (np.exp(x) - 1) + 1))

x_end = np.concatenate((x_lMax, x_uMax))
y_end = np.log((qArrMax * (np.exp(x_end) - 1) + 1) /
               (dArrMax * (np.exp(x_end) - 1) + 1))

for xi, yi in zip(x, y):
    plt.plot(xi, yi, linewidth=linewidth)

ax = plt.gca()
ax.tick_params(labelsize=20)
ax.set_xlabel(r'$\alpha$', fontsize=20)
ax.set_ylabel('Incremental Privacy leakage', fontsize=20)

# Plot special point & annotation
x_intersec = x_lMax[1]
y_intersec = np.log((qArrMax * (np.exp(x_intersec) - 1) + 1) /
                    (dArrMax * (np.exp(x_intersec) - 1) + 1))
plt.plot(x_intersec, y_intersec, 'p', markersize=18)

plt.annotate(r'$\mathcal{L}(\alpha)$',
            xy=(0.4, 0.7),
            xytext=(0.5, 0.7),
            fontsize=18,
            arrowprops=dict(arrowstyle='->'),
            bbox=dict(boxstyle='round,pad=0.5'))

plt.show()
