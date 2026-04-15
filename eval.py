import matplotlib.pyplot as plt
import numpy as np
import util

def plot_F_surface(args, resolution=80):
    """
    Plots F(alpha,beta,gamma) over the simplex alpha+beta+gamma=1.
    """

    alpha_vals = np.linspace(1e-6, 1-1e-6, resolution)
    beta_vals  = np.linspace(1e-6, 1-1e-6, resolution)

    A, B = np.meshgrid(alpha_vals, beta_vals)
    F = np.full_like(A, np.nan)

    for i in range(resolution):
        for j in range(resolution):
            alpha = A[i, j]
            beta = B[i, j]
            gamma = 1 - alpha - beta

            if gamma > 0:  # inside simplex
                F[i, j] = util.func_F_phi(alpha, beta, gamma, args)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(A, B, F, linewidth=0, antialiased=True)

    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_zlabel("F(alpha,beta,gamma)")
    ax.set_title("Surface of F on simplex (gamma = 1 - alpha - beta)")

    plt.show()

def plot_loo_validation(y_actual, y_pred, title_suffix=""):
    """
    Plots a timeline comparison and a scatter plot of Actual vs. Predicted values.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Sequential Timeline
    axs[0].plot(y_actual, label='Actual', marker='o', linestyle='-', color='black')
    axs[0].plot(y_pred, label='LOO Prediction', marker='x', linestyle='--', color='blue')
    axs[0].set_title(f"LOO-CV Predictions over Data Index {title_suffix}")
    axs[0].set_xlabel("Data Index")
    axs[0].set_ylabel("Response (y)")
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.7)

    # Plot 2: Actual vs Predicted Scatter
    axs[1].scatter(y_actual, y_pred, color='red', alpha=0.7, edgecolors='k')

    # 1:1 Ideal Fit Line
    max_val = max(max(y_actual), max(y_pred))
    min_val = min(min(y_actual), min(y_pred))
    axs[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ideal Fit (y=x)')

    axs[1].set_title("Actual vs. Predicted Performance")
    axs[1].set_xlabel("Actual Value")
    axs[1].set_ylabel("Predicted Value")
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()