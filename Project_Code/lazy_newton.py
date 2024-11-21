import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm

def driver():
    Nmax = 100
    x0 = np.array([1.5, 0.5, 2.5])
    tol = 1e-6
    
    [xstar, gval, ier, errors] = LazyNewtonMethod(x0, tol, Nmax)
    print("The Lazy Newton method found the solution:", xstar)
    print("g evaluated at this point is:", gval)
    print("ier is:", ier)

    # Plotting log(error) vs. iterations
    iterations = range(len(errors))
    log_errors = np.log(errors)  # Compute log of errors

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, log_errors, marker='o')
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'$\log(e_k)$', fontsize=14)
    plt.title("Lazy Newton Method Performance", fontsize=16)
    plt.show()
    

def evalF(x):
    F = np.zeros(3)
    F[0] = x[0]**2 - 2.
    F[1] = np.exp(x[1]) - 1.
    F[2] = x[2]**3 - 3*x[2] + 2
    return F

def evalJ(x):
    J = np.array([
        [2*x[0], 0, 0],
        [0, np.exp(x[1]), 0],
        [0, 0, 3*x[2]**2 - 3]
    ])
    return J

def evalf(x):
    F = evalF(x)
    return np.dot(F,F)

def evalg(x):
    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)
    return 2 * np.dot(J.T, F)

def eval_hessianf(x):
    F = evalF(x)
    J = evalJ(x)
    n = len(F)
    m = len(x)
    second_derivatives = np.zeros((m, m))
    for i in range(n):
        Fi_hessian = eval_hessianFi(x, i)  # Hessian of each F_i
        second_derivatives += 2 * F[i] * Fi_hessian
    
    # Full Hessian
    return 2 * np.dot(J.T, J) + second_derivatives

def eval_hessianFi(x, i):
    H = np.zeros((3, 3))  # Replace with second derivatives of F_i
    return H

###############################
### Lazy Newton method

def LazyNewtonMethod(x, tol, Nmax):
    errors = []
    H_inv = None  # Cached inverse of the Hessian
    for its in range(Nmax):
        gradf = eval_gradg(x)

        # Compute Hessian only on the first iteration or periodically
        if H_inv is None or its % 5 == 0:  # Recompute Hessian every 5 iterations
            H = eval_hessianf(x)
            try:
                H_inv = np.linalg.inv(H)  # Cache the inverse of the Hessian
            except np.linalg.LinAlgError:
                print("Singular Hessian matrix")
                ier = 1
                return [x, evalf(x), ier, errors]

        # Compute Newton step using cached Hessian inverse
        delta = -np.dot(H_inv, gradf)

        # Update solution
        x = x + delta
        
        # Evaluate the function norm to check convergence
        gval = norm(gradf)
        errors.append(gval)
        if gval < tol:
            ier = 0
            return [x, gval, ier, errors]
    
    print('Max iterations exceeded')
    ier = 1
    return [x, evalg(x), ier, errors]

if __name__ == '__main__':
    driver()
