import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from bbb.utils.pytorch_setup import DEVICE
from bbb.utils.tqdm import train_with_tqdm
from bbb.utils.plotting import plot_weight_samples, plot_bbb_regression_predictions, plot_dnn_regression_predictions
from bbb.config.constants import KL_REWEIGHTING_TYPES, PRIOR_TYPES, VP_VARIANCE_TYPES
from bbb.config.parameters import Parameters, PriorParameters
from bbb.models.dnn import RegressionDNN
from bbb.models.bnn import RegressionBNN
from bbb.data import generate_regression_data


logger = logging.getLogger(__name__)


#############
# BBB Methods
#############

def _bbb_regression_evaluation(net: nn.Module, X_train: DataLoader = None, X_val: DataLoader = None):
    if X_train is None or X_val is None:
        X_train = generate_regression_data(train=True, size=8*BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
        X_val = generate_regression_data(train=False, size=BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    rmse = net.evaluate(X_val)
    logger.info(f'RMSE: {rmse}')

    X_train_arr = np.array(X_train.dataset, dtype=float)
    X_val_arr = np.array(X_val.dataset, dtype=float)

    Y_val_pred_mean, Y_val_pred_var, Y_val_pred_quartiles = net.predict(X_val.dataset[:][0])
    Y_val_pred_mean, Y_val_pred_var, Y_val_pred_quartiles = (
        Y_val_pred_mean.detach().cpu().numpy().flatten(),
        Y_val_pred_var.detach().cpu().numpy().flatten(),
        Y_val_pred_quartiles.detach().cpu().numpy()[:,:,0]
    )

    plot_bbb_regression_predictions(
        X_train_arr=X_train_arr,
        X_val_arr=X_val_arr,
        Y_val_pred_mean=Y_val_pred_mean,
        Y_val_pred_var=Y_val_pred_var,
        Y_val_pred_quartiles=Y_val_pred_quartiles,
        save_dir=net.model_save_dir
    )


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
    regression_likelihood_noise = 0.1,
    hidden_units = 400,
    hidden_layers = 3,
    batch_size = 128,
    lr = 1e-3,
    epochs = 1000,
    step_size = 250,
    elbo_samples = 5,
    inference_samples = 10,
    prior_type = PRIOR_TYPES.single,
    kl_reweighting_type = KL_REWEIGHTING_TYPES.paper,
    vp_variance_type = VP_VARIANCE_TYPES.paper,
    local_reparam_trick = False
)


def run_bbb_regression_training():
    logger.info('Beginning regression training...')
    net = RegressionBNN(params=BNN_REGRESSION_PARAMETERS).to(DEVICE)

    logger.info(BNN_REGRESSION_PARAMETERS)

    X_train = generate_regression_data(train=True, size=8*BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
    X_val = generate_regression_data(train=False, size=BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    train_with_tqdm(net=net, train_data=X_train, eval_data=X_val, epochs=BNN_REGRESSION_PARAMETERS.epochs)

    weight_samples = net.weight_samples()
    plot_weight_samples(weight_samples, save_dir=net.model_save_dir)

    _bbb_regression_evaluation(net, X_train=X_train, X_val=X_val)


def run_bbb_regression_evaluation(model_path: str):
    logger.info(f'Beginning regression evaluation against {model_path}...')
    
    logger.info(BNN_REGRESSION_PARAMETERS)

    net = RegressionBNN(params=BNN_REGRESSION_PARAMETERS, eval_mode=True).to(DEVICE)
    net.load_saved(model_path=model_path)

    weight_samples = net.weight_samples()
    plot_weight_samples(weight_samples, save_dir=net.model_save_dir)

    _bbb_regression_evaluation(net)


#############
# DNN Methods
#############

def _dnn_regression_evaluation(net: nn.Module, X_train: DataLoader = None, X_val: DataLoader = None):
    if X_train is None or X_val is None:
        X_train = generate_regression_data(train=True, size=8*BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
        X_val = generate_regression_data(train=False, size=BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    rmse = net.evaluate(X_val)
    logger.info(f'RMSE: {rmse}')

    X_train_arr = np.array(X_train.dataset, dtype=float)
    X_val_arr = np.array(X_val.dataset, dtype=float)

    Y_val_pred, _, _ = net.predict(X_val.dataset[:][0])
    Y_val_pred = Y_val_pred.detach().cpu().numpy().flatten()
    
    plot_dnn_regression_predictions(
        X_train_arr=X_train_arr,
        X_val_arr=X_val_arr,
        Y_val_pred=Y_val_pred,
        save_dir=net.model_save_dir
    )

DNN_REGRESSION_PARAMETERS = Parameters(
    name = "DNN_regression",
    input_dim = 1,
    output_dim = 1,
    hidden_layers = 3,
    hidden_units = 400,
    batch_size = 128,
    lr = 1e-3,
    epochs = 1000,
    step_size = 250,
    early_stopping = False,
    early_stopping_thresh = 1e-4
)

def run_dnn_regression_training():
    logger.info('Beginning regression training...')
    net = RegressionDNN(params=DNN_REGRESSION_PARAMETERS).to(DEVICE)

    X_train = generate_regression_data(train=True, size=8*BNN_REGRESSION_PARAMETERS.batch_size, batch_size=DNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
    X_val = generate_regression_data(train=False, size=DNN_REGRESSION_PARAMETERS.batch_size, batch_size=DNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    train_with_tqdm(net=net, train_data=X_train, eval_data=X_val, epochs=DNN_REGRESSION_PARAMETERS.epochs)

    weight_samples = net.weight_samples()
    plot_weight_samples(weight_samples, save_dir=net.model_save_dir)

    _dnn_regression_evaluation(net, X_train=X_train, X_val=X_val)


def run_dnn_regression_evaluation(model_path: str):
    logger.info(f'Beginning regression evaluation against {model_path}...')

    net = RegressionDNN(params=DNN_REGRESSION_PARAMETERS, eval_mode=True).to(DEVICE)
    net.load_saved(model_path=model_path)

    weight_samples = net.weight_samples()
    plot_weight_samples(weight_samples, save_dir=net.model_save_dir)

    _dnn_regression_evaluation(net)
