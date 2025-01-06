import numpy as np
import matplotlib.pyplot as plt
from preCompQDMatrix import preCompQDMatrix
from genLFunc import genLFunc
from calcPLbyFunc import calcPLbyFunc

def plotTPL(eArr, TM_B, TM_F, printTPL):
    EspMatrix_B, qM_B, dM_B, QDplusInd_B = preCompQDMatrix(TM_B)
    EspMatrix_B[np.isnan(EspMatrix_B)] = 0

    EspMatrix_F, qM_F, dM_F, QDplusInd_F = preCompQDMatrix(TM_F)
    EspMatrix_F[np.isnan(EspMatrix_F)] = 0

    a1 = 0
    an = 100
    aArrMax_B, qArrMax_B, dArrMax_B = genLFunc(a1, an, EspMatrix_B, qM_B, dM_B)
    aArrMax_F, qArrMax_F, dArrMax_F = genLFunc(a1, an, EspMatrix_F, qM_F, dM_F)

    T = eArr.shape[0]

    bplArr = np.zeros(T)
    fplArr = np.zeros(T)

    bplArr[0] = eArr[0]
    fplArr[T-1] = eArr[T-1]

    for t in range(1, T):
        bplArr[t] = calcPLbyFunc(bplArr[t-1], aArrMax_B, qArrMax_B, dArrMax_B) + eArr[t]

    for t in range(T-2, -1, -1):
        fplArr[t] = calcPLbyFunc(fplArr[t+1], aArrMax_F, qArrMax_F, dArrMax_F) + eArr[t]

    tplArr = bplArr + fplArr - eArr

    if printTPL:
        print(tplArr)

    plt.figure()
    plt.plot(range(1, T+1), eArr, '-k*', label='budget')
    plt.plot(range(1, T+1), bplArr, '--bo', label='BPL')
    plt.plot(range(1, T+1), fplArr, '--ms', label='FPL')
    plt.plot(range(1, T+1), tplArr, '-r^', label='TPL', linewidth=1)
    ax = plt.gca()
    ax.set_yscale('linear')
    ax.tick_params(labelsize=12)
    ax.set_xticks(range(1, T+1))
    ax.set_ylim(0, np.max(tplArr) + 0.1)
    ax.set_xlabel('time')
    ax.set_ylabel('privacy loss')
    plt.legend(fontsize=16)
    plt.show()

