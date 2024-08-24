# Optimization for Engineers - Dr.Johannes Hild
# projected BFGS descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k is the BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the inverse BFGS matrix is reset.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

# myObjective = nonlinearObjective()
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[0.1], [0.1]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

# myObjective = nonlinearObjective()
# a = np.array([[-2], [-2]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[1.5], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[-0.26],[0.21]] (if it is close to [[0.26],[-0.21]] then maybe your reduction is done wrongly)

# myObjective = bananaValleyObjective()
# a = np.array([[-10], [-10]])
# b = np.array([[10], [10]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[0], [1]], dtype=float)
# eps = 1.0e-6
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]] in less than 30 iterations. If you have too much iterations, then maybe the hessian is used wrongly.


import numpy as np
import projectedBacktrackingSearch as PB


def matrnr():
    # set your matriculation number here
    matrnr = 23159043
    return matrnr


def projectedBFGSDescent(f, P, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedBFGSDescent...')

    countIter = 0
    xp = P.project(x0)
    gradx = f.gradient(xp)
    Bk = np.eye(len(xp))
    while  np.linalg.norm(xp - P.project(xp - f.gradient(xp))) > eps :
        a = P.activeIndexSet(xp)
        Bk[ a , : ]  = Bk[ a , : ] 
        Bk[: , a]= Bk[: , a]
        d = - Bk @ ( f.gradient(xp))
        if ( f.gradient(xp).T) @ d >= 0:
            d = (-gradx)
            Bk = np.eye(len(xp))
          
        t = PB.projectedBacktrackingSearch(f, P, xp, d)
        xpold = xp
        xp = P.project(xp + t * d)
        delta_xp = xp - xpold
        delta_gp = f.gradient(xp) - f.gradient(xpold)
        rp = delta_xp - Bk @ delta_gp
        Bk = Bk + (((rp @ delta_xp.T)+(delta_xp @ rp.T))/ (delta_gp.T @ delta_xp)) - ((rp.T @ delta_gp)/((delta_gp.T @ delta_xp)**2)) * (delta_xp @ delta_xp.T)

       
        countIter +=1

    



    if verbose:
        print('projectedBFGSDescent terminated after ', countIter, ' steps with norm of stationarity =',
              np.linalg.norm(xp - P.project(xp - f.gradient(xp))))

    return xp
