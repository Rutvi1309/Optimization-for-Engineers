# Optimization for Engineers - Dr.Johannes Hild
# Wolfe-Powell line search

# Purpose: Find t to satisfy f(x+t*d)<=f(x) + t*sigma*gradf(x).T@d and gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set, such that t satisfies both Wolfe - Powell conditions

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=1

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.2], [1]])
# d = np.array([[0.1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=16

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-0.2], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=0.25

# myObjective = nonlinearObjective()
# x = np.array([[0.53], [-0.29]])
# d = np.array([[-3.88], [1.43]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=0.0938


import numpy as np

def checkw1(f, x, d, t, sigma):
    return f.objective(x + t * d) <= f.objective(x) + t * sigma * f.gradient(x).T @ d 

def checkw2(f, x, d, t, rho):
    return f.gradient(x + t * d).T @ d >= rho * f.gradient(x).T @ d

def matrnr():
    # set your matriculation number here
    matrnr = 23159043
    return matrnr


def WolfePowellSearch(f, x: np.array, d: np.array, sigma=1.0e-3, rho=1.0e-2, verbose=0):
    fx = f.objective(x)
    gradx = f.gradient(x)
    descent = gradx.T @ d


    if descent >= 0:
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5:
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1:
        raise TypeError('range of rho is wrong!')

    if verbose:
        print('Start WolfePowellSearch...')

    t = 1

    xt = x + t * d
    fxt = f.objective(xt)
    gradxt = f.gradient(xt)

    if checkw1(f, x, d, t, sigma) == False:
        t /= 2
        
        while checkw1(f, x, d, t, sigma) == False:
            t /= 2
            
        t_minus = t
        t_plus = 2 * t
        
    elif checkw2(f, x, d, t, rho) == True:
        return t
    
    else:
        t = 2 * t
        
        while checkw1(f, x, d, t, sigma) == True:
            t = 2 * t
            
        t_minus = t / 2
        t_plus = t
    
    t = t_minus
    
    while checkw2(f, x, d, t, rho) == False:
        
        t = (t_minus + t_plus) / 2
        
        if checkw1(f, x, d, t, sigma) == True:
            
            t_minus = t
            
    return t_minus
        

    if verbose:
        xt = x + t * d
        fxt = f.objective(xt)
        gradxt = f.gradient(xt)
        print('WolfePowellSearch terminated with t=', t)
        print('Wolfe-Powell: ', fxt, '<=', fx+t*sigma*descent, ' and ', gradxt.T @ d, '>=', rho*descent)

    return t
