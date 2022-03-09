from typing import List

import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt


def plot_bbb_regression_predictions(X_train_arr: Tensor, X_val_arr: Tensor, Y_val_pred_mean: Tensor, Y_val_pred_var: Tensor):
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
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    ax.plot(X_train_arr[:,0], X_train_arr[:,1], label='Original')
    
    ax.plot(X_val_arr[:,0], Y_val_pred_mean, marker='x', label='Prediction')
    ax.fill_between(X_val_arr[:,0], Y_val_pred_mean-2*np.sqrt(Y_val_pred_var), Y_val_pred_mean+2*np.sqrt(Y_val_pred_var), alpha=0.5)
    
    ax.legend()
    plt.show()


def plot_weight_samples(weight_samples: List[Tensor]):
    """Plot a historgram of the passed weights.

    :param weight_samples: List of sampled weights
    :type weight_samples: List[Tensor]
    """
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    for i, weights in enumerate(weight_samples):
        ax.hist(weights.flatten().detach().numpy(), density=True, alpha=0.5, label=f'Layer: {i}, Weights: {weights.shape[0]}')
    
    ax.legend()
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    plt.show()

# def plot_bandit()
