from typing import List

import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt


def plot_bbb_regression_predictions(
    X_train_arr: Tensor,
    X_val_arr: Tensor,
    Y_val_pred_mean: Tensor,
    Y_val_pred_var: Tensor,
    Y_val_pred_quartiles: np.array,
    save_path: str
):
    """Plot the regression predictions made by BBB.

    :param X_train_arr: training data
    :type X_train_arr: Tensor
    :param X_val_arr: evaluation data
    :type X_val_arr: Tensor
    :param Y_val_pred_mean: mean of predictions
    :type Y_val_pred_mean: Tensor
    :param Y_val_pred_var: variance of predictions
    :type Y_val_pred_var: Tensor
    """
    # Initialise the figure
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    # Plot the data points
    ax.plot(X_train_arr[:,0], X_train_arr[:,1], label='Original', ls='', marker='x')
    
    # Plot the predictive mean
    ax.plot(X_val_arr[:,0], Y_val_pred_mean, marker='x', label='Prediction')

    # Two standard deviations
    ax.fill_between(X_val_arr[:,0], Y_val_pred_mean-2*np.sqrt(Y_val_pred_var), Y_val_pred_mean+2*np.sqrt(Y_val_pred_var), color='tab:green', alpha=0.2)

    # 0th and 100th percentile
    ax.fill_between(X_val_arr[:,0], Y_val_pred_quartiles[0,:], Y_val_pred_quartiles[3,:], color='tab:blue', alpha=0.25)
    
    # 25th and 75th percentile
    ax.fill_between(X_val_arr[:,0], Y_val_pred_quartiles[1,:], Y_val_pred_quartiles[2,:], color='tab:orange', alpha=0.50)

    # Formatting of plot    
    ax.legend()
    # ax.set_ylim([-1.5, 1.5])
    # ax.set_xlim([-0.6, 1/.4])

    # Display the plot
    plt.show()

    # Save the figure
    plt.savefig(save_path)


def plot_weight_samples(weight_samples: List[Tensor]):
    """Plot a historgram of the passed weights.

    :param weight_samples: List of sampled weights
    :type weight_samples: List[Tensor]
    """
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    for i, weights in enumerate(weight_samples):
        ax.hist(weights.flatten().detach().cpu().numpy(), density=True, alpha=0.5, label=f'Layer: {i}, Weights: {weights.shape[0]}')
    
    ax.legend()
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    plt.show()
