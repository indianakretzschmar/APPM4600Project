import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

def driver():
    # THIS CODE IS FOR BFGS METHOD

    Nmax = 100
    x0 = np.array([0, 0, 0])  # Initial guess
    tol = 1e-6

    [xstar, gval, ier, errors] = BFGSMethod(x0, tol, Nmax)
    print("The BFGS method found the solution:", xstar)
    print("g evaluated at this point is:", gval)
    print("ier is:", ier)

def evalF(x):
    # F(X)
    F = np.zeros(3)
    F[0] = x[0]**2 - x[1]
    F[1] = x[0] * x[1] + x[1]**2
    F[2] = x[2]**3
    return F

def evalJ(x):
    # JACOBIAN
    J = np.array([
        [2*x[0], -1, 0],
        [x[1], x[0] + 2*x[1], 0],
        [0, 0, 3*x[2]**2]
    ])
    return J

def evalg(x):
    # g(x) = ||F(x)||^2
    F = evalF(x)
    return np.dot(F, F)

def eval_gradg(x):
    # GRADIENT
    F = evalF(x)
    J = evalJ(x)
    return 2 * np.dot(J.T, F)

def BFGSMethod(x, tol, Nmax):
    # SOLVING F(X) = 0 USING BFGS
    errors = []
    n = len(x)
    B = np.eye(n)  # Initialize Hessian approximation as identity matrix

    for its in range(Nmax):
        gradf = eval_gradg(x)

        # Check for convergence
        gval = norm(gradf)
        errors.append(gval)
        if gval < tol:
            ier = 0
            return [x, gval, ier, errors]

        # Compute search direction
        try:
            delta = -np.linalg.solve(B, gradf)
        except np.linalg.LinAlgError:
            print("Singular approximation matrix B")
            ier = 1
            return [x, evalg(x), ier, errors]

        # Update the solution
        x_new = x + delta
        gradf_new = eval_gradg(x_new)

        # Update the Hessian approximation using BFGS formula
        s = x_new - x
        y = gradf_new - gradf
        sy = np.dot(s, y)

        # Avoid division by zero in the BFGS update
        if np.abs(sy) < 1e-10:
            print("Skipping BFGS update due to small sy")
            x = x_new
            continue

        Bs = np.dot(B, s)
        B = B + np.outer(y, y) / sy - np.outer(Bs, Bs) / np.dot(s, Bs)

        # Move to next iteration
        x = x_new

    print('Max iterations exceeded')
    ier = 1
    return [x, evalg(x), ier, errors]

if __name__ == '__main__':
    driver()



