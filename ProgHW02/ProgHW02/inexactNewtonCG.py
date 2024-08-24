# Optimization for Engineers - Dr.Johannes Hild
# inexact Newton descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = directionalHessApprox(f, x, d) from directionalHessApprox.py
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = nonlinearObjective()
# x0 = np.array([[-0.01], [0.01]])
# eps = 1.0e-6
# xmin = inexactNewtonCG(myObjective, x0, eps, 1)
# should return
# xmin close to [[0.26],[-0.21]]

# myObjective = nonlinearObjective()
# x0 = np.array([[-0.6], [0.6]])
# eps = 1.0e-3
# xmin = inexactNewtonCG(myObjective, x0, eps, 1)
# should return
# xmin close to [[-0.26],[0.21]]

# myObjective = nonlinearObjective()
# x0 = np.array([[0.6], [-0.6]])
# eps = 1.0e-3
# xmin = inexactNewtonCG(myObjective, x0, eps, 1)
# should return
# xmin close to [[-0.26],[0.21]]


import numpy as np
import WolfePowellSearch as WP
import directionalHessApprox as DHA

def matrnr():
    23159043
    matrnr = 23159043
    return matrnr


def inexactNewtonCG(f, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start inexactNewtonCG...')

    countIter = 0
    x = x0.copy()
    gradx = f.gradient(x)
    eta = np.min([0.5,np.sqrt(np.linalg.norm(f.gradient(x)))]) *np.linalg.norm(f.gradient(x))
    while np.linalg.norm(f.gradient(x)) > eps:
        dk =  -f.gradient(x)
        dh = DHA.directionalHessApprox(f,x,dk)
        
        rhok = dk.T @ dh
        
        if rhok > eps*np.dot(dk.T, dk):
            rj = f.gradient(x)
            dj = -rj
            xj = x
            da =dh
            rhoj = rhok
            tj = np.dot(rj.T, rj)/rhoj
            xj = xj+tj*dj
            rold = rj
            rj = rold+tj*da
            betaj = np.dot(rj.T, rj)/np.dot(rold.T, rold)
            dj = -rj+betaj*dj
            while np.linalg.norm(rj)> eta:
                da = DHA.directionalHessApprox(f,x,dj)
                rhoj = dj.T @ da
                tj = np.dot(rj.T, rj)/rhoj
                xj = xj+tj*dj
                rold = rj
                rj = rold+tj*da
                betaj = np.dot(rj.T, rj)/np.dot(rold.T, rold)
                dj = -rj+betaj*dj
            dk = xj-x
        t = WP.WolfePowellSearch(f, x, dk)
        x = x + t * dk
        eta = np.min([0.5,np.sqrt(np.linalg.norm(f.gradient(x)))]) *np.linalg.norm(f.gradient(x))
        countIter += 1
            
            
        

    if verbose:
        gradx = f.gradient(x)
        print('inexactNewtonCG terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradx))

    return x
