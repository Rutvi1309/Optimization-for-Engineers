# Optimization for Engineers - Dr.Johannes Hild
# scaled unit central simplex gradient

# Purpose: Approximates gradient of f on a scaled unit central simplex

# Input Definition:
# f: objective class with methods .objective()
# x: column vector in R ** n(domain point)
# h: simplex edge length
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# grad_f_h: simplex gradient
# stenFail: 0 by default, but 1 if stencil failure shows up

# Required files:
# < none >

# Test cases:
# myObjective = nonlinearObjective()
# x = np.array([[-0.015793], [0.012647]], dtype=float)
# h = 1.0e-6
# myGradient = SUCSimplexGradient(myObjective, x, h)
# should return
# myGradient close to [[0],[0]]

# myObjective = multidimensionalObjective()
# x = np.array([[1.02614],[0],[0],[0],[0],[0],[0],[0]], dtype=float)
# h = 1.0e-6
# myGradient = SUCSimplexGradient(myObjective, x, h)
# should return
# myGradient close to [[0],[0],[0],[0],[0],[0],[0],[0]]

import numpy as np


def matrnr():
    
    matrnr = 23159043
    return matrnr


def SUCSimplexGradient(f, x: np.array, h, verbose=0):

    if verbose:
        print('Start SUCSimplexGradient...')

    grad_f_h = x.copy()
    n = x.shape[0]
    for i in range(n):
        e_j = x.copy()
        e_i = x.copy()
        e_i[i] += h
        e_j[i] -= h
        grad_f = (f.objective(e_i) - f.objective(e_j)) 
        grad_f_h[i] =grad_f / (2 * h)
    
    

    if verbose:
        print('SUCSimplexGradient terminated with gradient =', grad_f_h)

    return grad_f_h


def SUCStencilFailure(f, x: np.array, h, verbose=0):

    if verbose:
        print('Check for SUCStencilFailure...')

    stenFail = 1
    n = x.shape[0]
    for i in range(n):
        e_j = x.copy()
        e_i = x.copy()
        e_i[i] += h
        e_j[i] -= h
        a = f.objective(x)
        b = f.objective(e_i)
        c = f.objective(e_j)
        if b <= a or c <= a: 
            stenFail = 0
            
            break
            
            
     
    

    if verbose:
        print('SUCStencilFailure check returns ', stenFail)

    return stenFail
