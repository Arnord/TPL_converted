import numpy as np

def sortRatioM1M2_DES(m1, m2):
    rNum = m1.shape[0]
    cNum = m1.shape[1]

    # 计算 m1 / m2，并将除零结果设为 NaN
    ratio = np.divide(m1, m2, out=np.full_like(m1, np.nan), where=m2 != 0)

    # 对每一行进行降序排序，并返回排序后的索引
    idx = np.argsort(-ratio, axis=1)

    # 将索引转换为线性索引
    idx = (idx - 1) * rNum + np.tile(np.arange(rNum).reshape(-1, 1), (1, cNum))

    # 使用线性索引提取排序后的值
    m1 = m1.flatten()[idx]
    m2 = m2.flatten()[idx]

    return m1, m2