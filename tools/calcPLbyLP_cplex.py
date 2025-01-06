import numpy as np
import scipy.sparse as sp
import cplex
from itertools import combinations

def calcPLbyLP_cplex(Q, D, a):
    n = Q.shape[1]  # size of variables yi
    m = n * (n - 1) // 2  # size of constraints exp(-a) <= yi OR yj <= exp(a)

    # Generate all pairs (i,j) where i < j
    pairs = np.array(list(combinations(range(n), 2)))
    row = np.repeat(np.arange(2 * m), 2)
    col = np.hstack([pairs, pairs[:, ::-1]]).flatten()
    val = np.concatenate([np.tile([-np.exp(a), 1], m), np.tile([1, -np.exp(a)], m)])

    Aineq = sp.csr_matrix((val, (row, col)), shape=(2 * m, n))

    bineq = np.zeros(2 * m)  # NOTE: COLUMN vec for cplexlp (diff from linprog)

    # constraints D'*yi = 1
    Aeq = D.T
    beq = np.array([1])

    # min f
    f = -Q.T  # NOTE: COLUMN vec for cplexlp (diff from linprog)

    # Set up the CPLEX problem
    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)
    prob.variables.add(obj=f.flatten(), lb=[0]*n, ub=[cplex.infinity]*n)
    prob.linear_constraints.add(rhs=bineq.tolist(), senses="L" * len(bineq), lin_expr=[cplex.SparsePair(ind=list(range(n)), val=Aineq[i, :].toarray().flatten().tolist()) for i in range(Aineq.shape[0])])
    prob.linear_constraints.add(rhs=beq.tolist(), senses="E" * len(beq), lin_expr=[cplex.SparsePair(ind=list(range(n)), val=Aeq.flatten().tolist())])

    # Solve the problem
    prob.solve()

    # Get the solution
    x = prob.solution.get_values()
    fval = prob.solution.get_objective_value()

    bpl = np.log(-fval)  # linprog finds min, we multiply -1 * f

    return bpl, x

# Example usage
# Q = np.array([[1, 2], [3, 4]])
# D = np.array([[1], [1]])
# a = 1
# bpl, x = calcPLbyLP_cplex(Q, D, a)
# print("bpl:", bpl)
# print("x:", x)