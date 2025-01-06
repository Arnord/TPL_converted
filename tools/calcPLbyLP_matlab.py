import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from itertools import combinations


def calcPLbyLP_matalb(Q, D, a):
    """
    Can be also used for calculate FPL
    Note: some precision problem could happen when a is large (e.g. a>30)
    calcBPLbyPreComp.m and calcBPLbyFunc.m have no precision problem
    """
    n = Q.shape[1]  # size of variables yi
    m = n * (n - 1) // 2  # size of contraints  exp(-a)<=yi  OR   yj<=exp(a)

    # constraints  exp(-a)<=yi/yj<=exp(a)
    # i.e.:  exp(-a)*yj -yi<=0    yi-exp(a)*yj<=0
    row = np.tile(np.arange(1, 2 * m + 1), (2, 1))
    row = row.flatten(order='F')

    # Generate combinations for columns
    col = np.array(list(combinations(range(1, n + 1), 2))).T
    col = np.concatenate([col.flatten(), col.flatten()])

    # Create values array
    val = np.concatenate([
        np.tile([-np.exp(a), 1], m),
        np.tile([1, -np.exp(a)], m)
    ])

    # Create sparse matrix A
    A = sparse.csr_matrix((val, (row - 1, col - 1)))
    A = A.toarray()

    # Create b vector
    b = np.zeros(2 * m)

    # constraints D'*yi=1
    Aeq = D
    beq = np.array([1])

    # min f
    f = -1 * Q

    # Solve linear programming problem
    result = linprog(
        c=f,
        A_ub=A,
        b_ub=b,
        A_eq=Aeq,
        b_eq=beq,
        method='simplex',
        options={'disp': False}
    )

    x = result.x
    bpl = np.log(-1 * result.fun)  # linprog finds min, we multiply -1 * f

    return bpl, x
