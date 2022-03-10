import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from bbb.utils.pytorch_setup import DEVICE
from bbb.utils.tqdm import train_with_tqdm
from bbb.utils.plotting import plot_weight_samples, plot_bbb_regression_predictions
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
    Y_val_pred_mean, Y_val_pred_var = Y_val_pred_mean.detach().cpu().numpy().flatten(), torch.sqrt(Y_val_pred_var).detach().cpu().numpy().flatten()

    plot_bbb_regression_predictions(X_train_arr=X_train_arr, X_val_arr=X_val_arr, Y_val_pred_mean=Y_val_pred_mean, Y_val_pred_var=Y_val_pred_var, save_path=net.save_plot_path)


BNN_REGRESSION_PARAMETERS = Parameters(
    name = "BBB_regression",
    input_dim = 1,
    output_dim = 1,
    weight_mu_range = [-0.2, 0.2],
    weight_rho_range = [-5, -4],
    prior_params = PriorParameters(
        w_sigma=1.,
        b_sigma=1.,
        w_sigma_2=0.2,
        b_sigma_2=0.2,
        w_mixture_weight=0.5,
        b_mixture_weight=0.5,
    ),
    hidden_units = 400,
    hidden_layers = 4,
    batch_size = 128,
    lr = 1e-3,
    epochs = 1000,
    elbo_samples = 5,
    inference_samples = 10,
    prior_type = PRIOR_TYPES.single,
    kl_reweighting_type = KL_REWEIGHTING_TYPES.simple,
    vp_variance_type = VP_VARIANCE_TYPES.paper,
    local_reparam_trick = True
)

def run_bbb_regression_training():
    logger.info('Beginning regression training...')
    net = RegressionBNN(params=BNN_REGRESSION_PARAMETERS).to(DEVICE)

    logger.info(BNN_REGRESSION_PARAMETERS)

    X_train = generate_regression_data(train=True, size=10*BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
    X_val = generate_regression_data(train=False, size=BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    train_with_tqdm(net=net, train_data=X_train, eval_data=X_val, epochs=BNN_REGRESSION_PARAMETERS.epochs)

    weight_samples = net.weight_samples()
    plot_weight_samples(weight_samples)

    _bbb_regression_evaluation(net, X_train=X_train, X_val=X_val)


def run_bbb_regression_evaluation(model_path: str):
    logger.info(f'Beginning regression evaluation against {model_path}...')
    
    logger.info(BNN_REGRESSION_PARAMETERS)

    net = RegressionBNN(params=BNN_REGRESSION_PARAMETERS).to(DEVICE)
    net.load_saved(model_path=model_path)
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
    hidden_layers = 4,
    hidden_units = 400,
    batch_size = 100,
    lr = 1e-3,
    epochs = 100,
    early_stopping = False,
    early_stopping_thresh = 1e-4
)

def run_dnn_regression_training():
    logger.info('Beginning regression training...')
    net = DNN(params=DNN_REGRESSION_PARAMETERS).to(DEVICE)

    X_train = generate_regression_data(train=True, size=10*BNN_REGRESSION_PARAMETERS.batch_size, batch_size=DNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
    X_val = generate_regression_data(train=False, size=DNN_REGRESSION_PARAMETERS.batch_size, batch_size=DNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    train_with_tqdm(net=net, train_data=X_train, eval_data=X_val, epochs=DNN_REGRESSION_PARAMETERS.epochs)

    _dnn_regression_evaluation(net, X_train=X_train, X_val=X_val)


def run_dnn_regression_evaluation(model_path: str):
    logger.info(f'Beginning regression evaluation against {model_path}...')
    net = DNN(params=DNN_REGRESSION_PARAMETERS).to(DEVICE)
    net.load_saved(model_path=model_path)
    _dnn_regression_evaluation(net)
