# NOTE: Gurobi 7.5/7.0 is runable only for Matlab 2016b or former
# can be also used for calculate FPL
# some precision problem could happen when  a is large (e.g. a>30)
# calcBPLbyPreComp.m and calcBPLbyFunc.m have no precision problem

import numpy as np
from itertools import combinations
from gurobipy import *
import scipy.sparse as sp


def calcBPLbyLP_gu(Q, D, a):
    n = Q.shape[1]
    m = n * (n - 1) // 2

    # constraints  exp(-a)<=yi/yj<=exp(a); D'*yi = 1;
    row = np.tile(np.arange(1, 2 * m + 1), (2, 1))
    row = np.vstack([row.flatten(), np.full(n, 2 * m + 1)])

    # Generate combinations for columns
    col_pairs = np.array(list(combinations(range(1, n + 1), 2))).T
    col = np.vstack([col_pairs.flatten(), col_pairs.flatten(), np.arange(1, n + 1)])

    # Create values array
    val_part1 = np.tile([-np.exp(a), 1], m)
    val_part2 = np.tile([1, -np.exp(a)], m)
    val = np.concatenate([val_part1, val_part2, D.T.flatten()])

    # Create sparse matrix
    A = sp.coo_matrix((val, (row.flatten() - 1, col.flatten() - 1))).tocsr()

    # Create and setup Gurobi model
    model = Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Method', -1)  # auto

    # Create variables
    x = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")

    # Set objective
    obj = QuadExpr()
    for i in range(n):
        obj.add(Q[i] * x[i])
    model.setObjective(obj, GRB.MAXIMIZE)

    # Add constraints
    rhs = np.zeros(2 * m)
    rhs = np.append(rhs, 1)

    # Add constraints from sparse matrix A
    for i in range(A.shape[0]):
        row = A.getrow(i)
        expr = LinExpr()
        for j, val in zip(row.indices, row.data):
            expr.add(x[j] * val)
        if i < 2 * m:
            model.addConstr(expr <= 0)
        else:
            model.addConstr(expr == rhs[i])

    try:
        model.optimize()

        if model.status == GRB.OPTIMAL:
            x_sol = np.array([x[i].X for i in range(n)])
            bpl = np.log(model.ObjVal)
            return bpl, x_sol
        else:
            print(f"Optimization status: {model.status}")
            return float('inf'), float('inf')

    except GurobiError as e:
        print(f"Error: {e}")
        return float('inf'), float('inf')

