# Converted from MATLAB
#=========================================
# this function calculate the incremental privacy leakag, i.e. L(a), by
# pre-computating the common results
# 05-Dec-2017 author: Yang Cao 
#-----------------inputs-----------------
# EspMatrix: n(n-1)*n matrix, contains transition points
# qM: n(n-1)*n matrix, contains q w.r.t. the corresponding transition point
# dM: n(n-1)*n matrix, contains d w.r.t. the corresponding transition point
#-----------------outputs-----------------
# maxPL: maximum incremental privacy leakage
# q,d: the ones satisfying theorem 4 in our ICDE/TKDE paper.
#=========================================


import numpy as np

# def calcPLbyPreComp(a, EspMatrix, qM, dM):
#     qdIdx = EspMatrix > a
#     jA = np.sum(qdIdx, axis=1)
#     rLen = EspMatrix.shape[0]
#     iA = np.arange(1, rLen + 1)
#     id = (jA - 1) * rLen + iA
#     id = id[id > 0]  # in case q==d
#     qArr = qM[id - 1]
#     dArr = dM[id - 1]
#
#     bplArr = np.log((qArr * (np.exp(a) - 1) + 1) / (dArr * (np.exp(a) - 1) + 1))
#     maxPL = np.max(bplArr)
#     idx = np.argmax(bplArr)
#     q = qArr[idx]
#     d = dArr[idx]
#
#     return maxPL, q, d

def calcPLbyPreComp(a, EspMatrix, qM, dM):
    # 找到 EspMatrix 中大于 a 的元素
    qdIdx = EspMatrix > a

    # 计算每行中大于 a 的元素数量
    jA = np.sum(qdIdx, axis=1)

    # 获取 EspMatrix 的行数
    rLen = EspMatrix.shape[0]

    # 生成行索引
    iA = np.arange(1, rLen + 1)  # 不需要减 1，因为用于计算

    # 计算 id
    id = (jA - 1) * rLen + iA
    id = id[id > 0]  # 过滤掉 id <= 0 的情况

    # 确保 id - 1 在 qM 的有效范围内
    valid_indices = (id - 1 < qM.size) & (id - 1 >= 0)
    id = id[valid_indices]

    # 提取 qArr 和 dArr
    qArr = qM.flat[id - 1]  # 使用线性索引提取元素 # TODO need to ensure if or not use linear idx
    dArr = dM.flat[id - 1]

    # 计算 bplArr
    bplArr = np.log((qArr * (np.exp(a) - 1) + 1) / (dArr * (np.exp(a) - 1) + 1))

    # 找到最大值及其索引
    maxPL = np.max(bplArr)
    idx = np.argmax(bplArr)

    # 提取对应的 q 和 d
    q = qArr[idx]
    d = dArr[idx]

    return maxPL, q, d

