import numpy as np
from itertools import combinations
import tools

def preCompQDMatrix(TM):
    """
    Precompute results that are common for each time point w.r.t. the same transition matrix.

    Parameters:
    -----------
    TM : ndarray
        Transition matrix

    Returns:
    --------
    EspMatrix : ndarray
        n(n-1)*n matrix, matrix version of "aArr", contains transition points.
        EspMatrix = [alpha_1, ..., alpha_{k-1}, NaN,..]
    qM : ndarray
        n(n-1)*n matrix, matrix version of "qArr", contains q w.r.t. the corresponding transition point
    dM : ndarray
        n(n-1)*n matrix, matrix version of "dArr", contains d w.r.t. the corresponding transition point
    QDplusInd : ndarray
        n(n-1)*n matrix, contains 0 or 1 means whether the position exist a transition points

    Author: Yang Cao (05-Dec-2017)
    Python translation
    """

    n = TM.shape[0]

    # Generate all pairs of indices
    pairs = np.array(list(combinations(range(1, n + 1), 2)))  # 生成所有 2-组合，索引从 1 开始
    # 对 pairs 的每一行进行翻转
    flipped_pairs = np.fliplr(pairs)
    # 将 pairs 和 flipped_pairs 垂直拼接
    pairs = np.vstack((pairs, flipped_pairs))

    # Extract QM and DM from transition matrix
    QM = TM[pairs[0, :] - 1, :]
    DM = TM[pairs[1, :] - 1, :]

    # Sort QM and DM in descending order based on their ratio
    QM, DM = tools.sortRatioM1M2_DES(QM, DM)

    # Create indicator matrix for Q > D
    QDplusInd = QM > DM

    # Apply indicator matrix
    QM = QM * QDplusInd
    DM = DM * QDplusInd

    # Compute cumulative sums
    qM = np.cumsum(QM, axis=1)
    dM = np.cumsum(DM, axis=1)

    # Apply indicator matrix again
    qM = qM * QDplusInd
    dM = dM * QDplusInd

    # Compute EspMatrix
    with np.errstate(divide='ignore', invalid='ignore'):   # ignore compute error
        EspMatrix = np.log((QM - DM) / (qM * DM - QM * dM) + 1)
    # EspMatrix = np.log((QM - DM) / (qM * DM - QM * dM) + 1)

    return EspMatrix, qM, dM, QDplusInd

