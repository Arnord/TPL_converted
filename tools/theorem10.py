import numpy as np

def theorem10(q, d, e):
    eps = np.finfo(float).eps
    if abs(q - d) < eps:
        sup = e
    elif d != 0:
        sup = np.log(((4 * d * np.exp(e) * (1 - q) + (d + q * np.exp(e) - 1)**2)**0.5 + d + q * np.exp(e) - 1) / (2 * d))
    elif q != 1 and e <= np.log(1 / q):
        sup = np.log(((1 - q) * np.exp(e)) / (1 - q * np.exp(e)))
    elif q != 1 and e > np.log(1 / q):
        sup = np.inf
    else:
        sup = np.inf
    return sup

