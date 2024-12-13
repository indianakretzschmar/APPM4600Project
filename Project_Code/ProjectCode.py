import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import time

# Fetch historical data for five stocks over one month
def get_data():
    stocks = ["NVDA", "CVX", "LLY", "MCD", "COST"]  # Example stocks
    start_date = "2023-01-01"
    end_date = "2023-01-31"

    # Download adjusted close prices
    data = yf.download(stocks, start=start_date, end=end_date)["Adj Close"]
    returns = data.pct_change().dropna()
    mu = returns.mean().values  # Mean returns
    Sigma = returns.cov().values  # Covariance matrix
    R_f = 0.02 / 252  # Daily risk-free rate
    return mu, Sigma, R_f

def evalF(w, mu, Sigma, R_f):
    # Objective function: Negative Sharpe ratio
    Rp = w @ mu
    sigma_p = np.sqrt(w @ Sigma @ w)
    if sigma_p == 0:
        return np.inf
    return -(Rp - R_f) / sigma_p

def eval_gradg(w, mu, Sigma, R_f):
    # Gradient of the negative Sharpe ratio
    Rp = w @ mu
    sigma_p = np.sqrt(w @ Sigma @ w)
    grad_Rp = mu
    grad_sigma_p = Sigma @ w / sigma_p
    grad = -grad_Rp / sigma_p + (Rp - R_f) * grad_sigma_p / (sigma_p ** 2)
    return grad

def eval_hessianf(w, mu, Sigma, R_f):
    # Hessian of the negative Sharpe ratio
    sigma_p = np.sqrt(w @ Sigma @ w)
    grad_sigma_p = Sigma @ w / sigma_p
    hess_sigma_p = Sigma / sigma_p - np.outer(grad_sigma_p, grad_sigma_p) / sigma_p
    hessian = (2 * np.outer(grad_sigma_p, grad_sigma_p) - hess_sigma_p) / (sigma_p ** 2)
    return hessian

def NewtonMethod(x, mu, Sigma, R_f, tol=1e-4, grad_tol=1e-3, max_iter=1000):
    errors = []
    sharpe_values = []
    weight_history = [x.copy()]
    prev_grad_norm = np.inf

    for its in range(max_iter):
        gradf = eval_gradg(x, mu, Sigma, R_f)
        H = eval_hessianf(x, mu, Sigma, R_f)

        try:
            delta = -np.linalg.solve(H, gradf)
        except np.linalg.LinAlgError:
            H += np.eye(len(x)) * 1e-6  # Add small regularization
            delta = -np.linalg.solve(H, gradf)

        # Update weights
        x = x + delta

        # Enforce constraints: weights >= 0.1 and normalize
        x = np.maximum(x, 0.1)  # Enforce minimum weight of 0.1
        x /= np.sum(x)  # Normalize weights to sum to 1

        # Track Sharpe ratio and weights
        sharpe_values.append(-evalF(x, mu, Sigma, R_f))
        weight_history.append(x.copy())

        # Track gradient norm
        grad_norm = np.linalg.norm(gradf)
        errors.append(grad_norm)

        # Check convergence based on gradient tolerance
        if abs(prev_grad_norm - grad_norm) < grad_tol:
            print(f"Converged after {its + 1} iterations.")
            break
        prev_grad_norm = grad_norm

    return x, errors, sharpe_values, weight_history

def plot_sharpe_progression(sharpe_values):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sharpe_values)), sharpe_values, marker='o', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Sharpe Ratio")
    plt.title("Sharpe Ratio Progression During Optimization")
    plt.grid()
    plt.show()

def plot_weights_progression(weight_history):
    weight_history = np.array(weight_history)
    plt.figure(figsize=(10, 6))
    for i in range(weight_history.shape[1]):
        plt.plot(range(weight_history.shape[0]), weight_history[:, i], label=f"Stock {i + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.title("Portfolio Weights Progression")
    plt.legend()
    plt.grid()
    plt.show()

def plot_gradient_convergence(errors):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(errors)), errors, marker='o', color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Convergence During Optimization")
    plt.grid()
    plt.show()

def driver():
    # Fetch data
    mu, Sigma, R_f = get_data()

    # Initial weights (equal allocation)
    x0 = np.ones(len(mu)) / len(mu)

    # Run Newton's method
    start_time = time.perf_counter()
    xstar, errors, sharpe_values, weight_history = NewtonMethod(x0, mu, Sigma, R_f, grad_tol=1e-3)
    end_time = time.perf_counter()

    # Results
    expected_return = xstar @ mu  # Calculate expected portfolio return
    print("Optimal Weights:", xstar)
    print("Sharpe Ratio of Optimal Portfolio:", sharpe_values[-1])
    print("Expected Portfolio Return:", expected_return*252)
    print("Elapsed Time:", end_time - start_time)

    # Plot Sharpe ratio progression
    plot_sharpe_progression(sharpe_values)

    # Plot weights progression
    plot_weights_progression(weight_history)

    # Plot gradient convergence
    plot_gradient_convergence(errors)

if __name__ == "__main__":
    driver()
