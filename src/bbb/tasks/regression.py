import logging

import numpy as np
import matplotlib.pyplot as plt

from bbb.utils.pytorch_setup import DEVICE
from bbb.utils.tqdm import train_with_tqdm
from bbb.config.constants import KL_REWEIGHTING_TYPES
from bbb.config.parameters import Parameters, PriorParameters
from bbb.models.dnn import DNN
from bbb.models.bnn import RegressionBNN
from bbb.data import generate_regression_data


logger = logging.getLogger(__name__)


BNN_REGRESSION_PARAMETERS = Parameters(
    name = "BBB_regression",
    input_dim = 1,
    output_dim = 1,
    weight_mu = [-0.2, 0.2],
    weight_rho = [-5, -4],
    prior_params = PriorParameters(
        w_sigma=1,
        b_sigma=2,
    ),
    hidden_units = 400,
    batch_size = 100,
    lr = 1e-3,
    epochs = 1000,
    elbo_samples = 5,
    inference_samples = 10,
    kl_reweighting_type=KL_REWEIGHTING_TYPES.simple,
)

def run_bbb_regression():
    logger.info('Beginning regression training...')
    net = RegressionBNN(params=BNN_REGRESSION_PARAMETERS).to(DEVICE)

    X_train = generate_regression_data(size=1000, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
    X_val = generate_regression_data(size=BNN_REGRESSION_PARAMETERS.batch_size, batch_size=BNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    train_with_tqdm(net=net, train_data=X_train, eval_data=X_val, epochs=BNN_REGRESSION_PARAMETERS.epochs)

    mse = net.eval(X_val)
    logger.info(f'MSE: {mse}')

    Y_pred = net.predict(X_train.dataset[:][0])

    X_train_arr = np.array(X_train.dataset)
    plt.plot(X_train_arr[:,0], X_train_arr[:,1])
    plt.plot(X_train_arr[:,0], Y_pred.detach().numpy(), marker='x')
    plt.show()


DNN_REGRESSION_PARAMETERS = Parameters(
    name = "DNN_regression",
    input_dim = 1,
    output_dim = 1,
    hidden_layers = 3,
    hidden_units = 50,
    batch_size = 10,
    lr = 0.01,
    epochs = 100,
)

def run_dnn_regression():
    logger.info('Beginning regression training...')
    net = DNN(params=DNN_REGRESSION_PARAMETERS).to(DEVICE)

    X_train = generate_regression_data(size=100, batch_size=DNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
    X_val = generate_regression_data(size=DNN_REGRESSION_PARAMETERS.batch_size, batch_size=DNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    train_with_tqdm(net=net, train_data=X_train, eval_data=X_val, epochs=DNN_REGRESSION_PARAMETERS.epochs)

    mse = net.eval(X_val)
    logger.info(f'MSE: {mse}')

    Y_pred = net.predict(X_train.dataset[:][0])

    X_train_arr = np.array(X_train.dataset, dtype=object)
    plt.plot(X_train_arr[:,0], X_train_arr[:,1])
    plt.plot(X_train_arr[:,0], Y_pred.detach().numpy(), marker='x')
    plt.show()
