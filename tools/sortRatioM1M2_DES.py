import numpy as np

def sort_ratio_m1_m2_des(m1, m2):
    rNum = m1.shape[0]
    cNum = m1.shape[1]

    idx = np.argsort(np.divide(m1, m2, out=np.full_like(m1, np.nan), where=m2!=0), axis=1)[:, ::-1]

    idx = (idx - 1) * rNum + np.tile(np.arange(1, rNum + 1).reshape(-1, 1), (1, cNum))

    m1 = m1.flatten()[idx]
    m2 = m2.flatten()[idx]

    return m1, m2