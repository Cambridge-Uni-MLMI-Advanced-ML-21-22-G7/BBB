import os
from typing import List

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt


def plot_bbb_regression_predictions(
    X_train_arr: Tensor,
    X_val_arr: Tensor,
    Y_val_pred_mean: Tensor,
    Y_val_pred_var: Tensor,
    Y_val_pred_quartiles: np.array,
    save_dir: str
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
    :param Y_val_pred_quartiles: quartiles of predictions
    :type Y_val_pred_quartiles: Tensor
    :param save_dir: directory to save the plot to
    :type save_dir: str
    """
    # Initialise the figure
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    # Plot the data points
    ax.plot(X_train_arr[:,0], X_train_arr[:,1], label='Original', ls='', marker='x')
    
    # Plot the predictive mean
    ax.plot(X_val_arr[:,0], Y_val_pred_mean, marker='x', label='Prediction')

    # Two standard deviations
    # ax.fill_between(X_val_arr[:,0], Y_val_pred_mean-2*np.sqrt(Y_val_pred_var), Y_val_pred_mean+2*np.sqrt(Y_val_pred_var), color='tab:green', alpha=0.2)

    # 0th and 100th percentile
    ax.fill_between(X_val_arr[:,0], Y_val_pred_quartiles[0,:], Y_val_pred_quartiles[3,:], color='tab:blue', alpha=0.25)
    
    # 25th and 75th percentile
    ax.fill_between(X_val_arr[:,0], Y_val_pred_quartiles[1,:], Y_val_pred_quartiles[2,:], color='tab:orange', alpha=0.50)

    # Formatting of plot
    ax.set_xlim([-0.2, 1.3])
    ax.set_ylim([-0.5, 1.3])

    # Add legend
    ax.legend()

    # Display the plot
    plt.show()

    # Save the figure
    plt.savefig(os.path.join(save_dir, 'plot.png'))


def plot_dnn_regression_predictions(
    X_train_arr: Tensor,
    X_val_arr: Tensor,
    Y_val_pred: Tensor,
    save_dir: str
):
    """Plot the regression predictions made by BBB.

    :param X_train_arr: training data
    :type X_train_arr: Tensor
    :param X_val_arr: evaluation data
    :type X_val_arr: Tensor
    :param Y_val_pred: predictions
    :type Y_val_pred: Tensor
    :param save_dir: directory to save the plot to
    :type save_dir: str
    """
    # Initialise the figure
    fig, ax = plt.subplots(1, 1, figsize=(10,10))

    # Plot the data points
    ax.plot(X_train_arr[:,0], X_train_arr[:,1], label='Original', ls='', marker='x')
    
    # Plot the predictive mean
    ax.plot(X_val_arr[:,0], Y_val_pred, marker='x', label='Prediction')

    # Formatting of plot
    ax.set_xlim([-0.2, 1.3])
    ax.set_ylim([-0.5, 1.3])

    # Add legend
    ax.legend()

    # Display the plot
    plt.show()

    # Save the figure
    plt.savefig(os.path.join(save_dir, 'plot.png'))


def plot_weight_samples(
    weight_samples: List[Tensor],
    save_dir: str
):
    """Plot a historgram of the passed weights.

    :param weight_samples: List of sampled weights
    :type weight_samples: List[Tensor]
    :param save_dir: directory to save the plots to
    :type save_dir: str
    """
    histogram_args = {
        'density': True,
        'bins': 50,
        'alpha': 0.5
    }

    ######################
    # Plots for each layer
    ######################

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    for i, weights in enumerate(weight_samples):
            ax.hist(weights.detach().cpu().numpy(), label=f'Layer: {i}, Weights: {weights.shape[0]}', **histogram_args)
    
    # Formatting of plot
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')

    # Add legend
    ax.legend()

    # Display the plot
    plt.show()

    # Save the figure
    plt.savefig(os.path.join(save_dir, 'weights_plot.png'))

    ################
    # Combined plots
    ################

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    comb_weight_samples = torch.hstack(weight_samples)
    ax.hist(comb_weight_samples.flatten().detach().cpu().numpy(), **histogram_args)

    # Formatting of plot
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')

    # Display the plot
    plt.show()

    # Save the figure
    plt.savefig(os.path.join(save_dir, 'comb_weights_plot.png'))
