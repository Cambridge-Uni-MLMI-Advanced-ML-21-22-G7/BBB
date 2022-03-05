import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from bbb.utils.pytorch_setup import DEVICE
from bbb.utils.tqdm import train_with_tqdm
from bbb.config.constants import KL_REWEIGHTING_TYPES, PRIOR_TYPES, VP_VARIANCE_TYPES
from bbb.config.parameters import Parameters, PriorParameters
from bbb.models.dnn import DNN
from bbb.models.bnn import RegressionBNN
from bbb.data import generate_regression_data


logger = logging.getLogger(__name__)


#############
# BBB Methods
#############

def _bbb_regression_evaluation(net: nn.Module, X_train: DataLoader = None, X_val: DataLoader = None):
    if X_train is None or X_val is None:
        X_train = generate_regression_data(train=True, size=10*BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
        X_val = generate_regression_data(train=False, size=BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    rmse = net.eval(X_val)
    logger.info(f'RMSE: {rmse}')

    X_train_arr = np.array(X_train.dataset, dtype=float)
    X_val_arr = np.array(X_val.dataset, dtype=float)

    Y_val_pred_mean, Y_val_pred_var = net.predict(X_val.dataset[:][0])
    Y_val_pred_mean, Y_val_pred_var = Y_val_pred_mean.detach().numpy().flatten(), torch.sqrt(Y_val_pred_var).detach().numpy().flatten()
    
    plt.plot(X_train_arr[:,0], X_train_arr[:,1], label='Original')
    plt.plot(X_val_arr[:,0], Y_val_pred_mean, marker='x', label='Prediction')
    plt.fill_between(X_val_arr[:,0], Y_val_pred_mean-2*Y_val_pred_var, Y_val_pred_mean+2*Y_val_pred_var, alpha=0.5)
    plt.legend()
    plt.show()


BNN_REGRESSION_PARAMETERS = Parameters(
    name = "BBB_regression",
    input_dim = 1,
    output_dim = 1,
    weight_mu = [-0.2, 0.2],
    weight_rho = [-5, -4],
    prior_params = PriorParameters(
        w_sigma=1.,
        b_sigma=1.,
        w_sigma_2=0.2,
        b_sigma_2=0.2,
        w_mixture_weight=0.5,
        b_mixture_weight=0.5,
    ),
    hidden_units = 400,
    hidden_layers=3,
    batch_size = 100,
    lr = 1e-3,
    epochs = 100,
    elbo_samples = 5,
    inference_samples = 10,
    prior_type=PRIOR_TYPES.mixture,
    kl_reweighting_type=KL_REWEIGHTING_TYPES.simple,
    vp_variance_type=VP_VARIANCE_TYPES.simple
)

def run_bbb_regression_training():
    logger.info('Beginning regression training...')
    net = RegressionBNN(params=BNN_REGRESSION_PARAMETERS).to(DEVICE)

    X_train = generate_regression_data(train=True, size=10*BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
    X_val = generate_regression_data(train=False, size=BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    train_with_tqdm(net=net, train_data=X_train, eval_data=X_val, epochs=BNN_REGRESSION_PARAMETERS.epochs)

    _bbb_regression_evaluation(net, X_train=X_train, X_val=X_val)


def run_bbb_regression_evaluation():
    logger.info('Beginning regression evaluation...')
    net = RegressionBNN(params=BNN_REGRESSION_PARAMETERS).to(DEVICE)
    net.load_saved()
    _bbb_regression_evaluation(net)


#############
# DNN Methods
#############

def _dnn_regression_evaluation(net: nn.Module, X_train: DataLoader = None, X_val: DataLoader = None):
    if X_train is None or X_val is None:
        X_train = generate_regression_data(train=True, size=10*BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
        X_val = generate_regression_data(train=False, size=BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    rmse = net.eval(X_val)
    logger.info(f'RMSE: {rmse}')

    X_train_arr = np.array(X_train.dataset, dtype=float)
    X_val_arr = np.array(X_val.dataset, dtype=float)

    Y_val_pred, _ = net.predict(X_val.dataset[:][0])
    Y_val_pred = Y_val_pred.detach().numpy().flatten()
    
    plt.plot(X_train_arr[:,0], X_train_arr[:,1], label='Original')
    plt.plot(X_val_arr[:,0], Y_val_pred, marker='x', label='Prediction')
    plt.xlim(-0.2, 1.4)
    plt.legend()
    plt.show()

DNN_REGRESSION_PARAMETERS = Parameters(
    name = "DNN_regression",
    input_dim = 1,
    output_dim = 1,
    hidden_layers = 3,
    hidden_units = 50,
    batch_size = 100,
    lr = 0.01,
    epochs = 100,
    early_stopping=True,
    early_stopping_thresh=1e-4
)

def run_dnn_regression_training():
    logger.info('Beginning regression training...')
    net = DNN(params=DNN_REGRESSION_PARAMETERS).to(DEVICE)

    X_train = generate_regression_data(train=True, size=10*BNN_REGRESSION_PARAMETERS.batch_size, batch_size=DNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
    X_val = generate_regression_data(train=False, size=DNN_REGRESSION_PARAMETERS.batch_size, batch_size=DNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    train_with_tqdm(net=net, train_data=X_train, eval_data=X_val, epochs=DNN_REGRESSION_PARAMETERS.epochs)

    _dnn_regression_evaluation(net, X_train=X_train, X_val=X_val)


def run_dnn_regression_evaluation():
    logger.info('Beginning regression evaluation...')
    net = DNN(params=DNN_REGRESSION_PARAMETERS).to(DEVICE)
    net.load_saved()
    _dnn_regression_evaluation(net)
