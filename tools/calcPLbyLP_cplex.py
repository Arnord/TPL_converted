import numpy as np
from scipy.sparse import csr_matrix
from cplex import Cplex


def calcPLbyLP_cplex(Q, D, a):
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

    # Generate combinations for column indices
    col_idx = []
    for i in range(n):
        for j in range(i + 1, n):
            col_idx.append([i, j])
    col = np.array(col_idx).T
    col = np.concatenate([col.flatten(), col.flatten()])

    val = np.concatenate([np.tile([-np.exp(a), 1], m), np.tile([1, -np.exp(a)], m)])

    # Create sparse matrix for inequality constraints
    Aineq = csr_matrix((val, (row - 1, col)), shape=(2 * m, n))
    bineq = np.zeros(2 * m)

    # Create CPLEX problem
    prob = Cplex()

    # Minimize objective function
    prob.objective.set_sense(prob.objective.sense.minimize)

    # Add variables
    prob.variables.add(obj=(-Q).flatten())

    # Add inequality constraints
    prob.linear_constraints.add(
        lin_expr=[[[range(n), row.tolist()] for row in Aineq.toarray()]],
        senses=["L"] * (2 * m),
        rhs=bineq.tolist()
    )

    # Add equality constraint D'*yi=1
    prob.linear_constraints.add(
        lin_expr=[[[range(n), D.tolist()]]],
        senses=["E"],
        rhs=[1.0]
    )

    # Solve the problem
    prob.solve()

    # Get solution
    x = np.array(prob.solution.get_values())
    fval = prob.solution.get_objective_value()

    bpl = np.log(-fval)  # linprog find min, we multiply -1* f

    return bpl, x

