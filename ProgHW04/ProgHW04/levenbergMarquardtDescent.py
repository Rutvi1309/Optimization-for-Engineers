# Optimization for Engineers - Dr.Johannes Hild
# Levenberg-Marquardt descent

# Purpose: Find pmin to satisfy norm(jacobian_R.T @ R(pmin))<=eps

# Input Definition:
# R: error vector class with methods .residual() and .jacobian()
# p0: column vector in R**n (parameter point), starting point.
# eps: positive value, tolerance for termination. Default value: 1.0e-4.
# alpha0: positive value, starting value for damping. Default value: 1.0e-3.
# beta: positive value bigger than 1, scaling factor for alpha. Default value: 100.
# verbose: bool, if set to true, verbose information is displayed.

# Output Definition:
# pmin: column vector in R**n (parameter point)

# Required files:
# d = CGSolver(A,b) from CGSolver.py

# Test cases:
# p0 = np.array([[180],[0]])
# myObjective =  simpleValleyObjective(p0)
# xk = np.array([[0, 0], [1, 2]])
# fk = np.array([[2, 3]])
# myErrorVector = leastSquaresObjective(myObjective, xk, fk)
# eps = 1.0e-4
# alpha0 = 1.0e-3
# beta = 100
# pmin = levenbergMarquardtDescent(myErrorVector, p0, eps, alpha0, beta, 1)
# should return pmin close to [[1], [1]]

# points = io.loadmat('measurePoints.mat')
# results = io.loadmat('measureResults.mat')
# p0 = np.array([[0],[0],[0]])
# myObjective =  benchmarkObjective(p0)
# xk = np.array(points, dtype=float)
# fk = np.array(results, dtype=float)
# myErrorVector = leastSquaresObjective(myObjective, xk, fk)
# eps = 1.0e-4
# alpha0 = 1.0e-3
# beta = 100
# pmin = levenbergMarquardtDescent(myErrorVector, p0, eps, alpha0, beta, 1)
# should return pmin close to [[3], [2], [16]]

import numpy as np
import CGSolver as CG


def matrnr():
    # set your matriculation number here
    matrnr = 23159043
    return matrnr


def levenbergMarquardtDescent(R, p0: np.array, eps=1.0e-4, alpha0=1.0e-3, beta=100, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if alpha0 <= 0:
        raise TypeError('range of alpha0 is wrong!')
    if beta <= 1:
        raise TypeError('range of beta is wrong!')

    if verbose:
        print('Start levenbergMarquardtDescent...')

    countIter = 0
    p = p0.copy()
    resp = R.residual(p)
    jacp = R.jacobian(p)
    lambda_ = alpha0

    while np.linalg.norm(jacp.T @ resp) > eps:
        countIter += 1

       
        
        A = jacp.T @ jacp + lambda_ * np.eye(jacp.shape[1])
        g = - jacp.T @ resp
        dp = CG.CGSolver(A, g)
        p_new = p + dp
        R_new = R.residual(p_new)
        J_new = R.jacobian(p_new)
        
        

        if 0.5 * R_new.T @ R_new < 0.5 * resp.T @ resp:
            p = p_new
            jacp = J_new 
            resp = R_new  
            lambda_ = alpha0
        else:
            lambda_ *= beta
        
    if verbose:
        print('levenbergMarquardtDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(jacp.T @ resp))

    return p
