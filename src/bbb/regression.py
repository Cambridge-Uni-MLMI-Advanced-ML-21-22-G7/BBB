import logging

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from bbb.parameters import Parameters
from bbb.models.dnn import DNN
from bbb.data import generate_regression_data, RegressionDataset


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DNN_REGRESSION_PARAMETERS = Parameters(
    name = "DNN",
    input_dim = 1,
    output_dim = 1,
    hidden_layers = 3,
    hidden_units = 50,
    batch_size = 10,
    lr = 0.01,
    epochs = 100,
)

def run_dnn_regression():
    logger.info('Beginning classification training...')
    net = DNN(params=DNN_REGRESSION_PARAMETERS)

    X_train = generate_regression_data(size=100, batch_size=DNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)
    X_val = generate_regression_data(size=DNN_REGRESSION_PARAMETERS.batch_size, batch_size=DNN_REGRESSION_PARAMETERS.batch_size, shuffle=True)

    epochs = DNN_REGRESSION_PARAMETERS.epochs
    for epoch in tqdm(range(epochs)):
        loss = net.train(X_train)
        logger.info(f'[Epoch {epoch+1}/{epochs}] - Loss: {loss}')

    mse = net.eval(X_val)
    logger.info(f'MSE: {mse}')

    Y_pred = net.predict(X_train.dataset[:][0])

    X_train_arr = np.array(X_train.dataset)
    plt.plot(X_train_arr[:,0], X_train_arr[:,1])
    plt.plot(X_train_arr[:,0], Y_pred.detach().numpy(), marker='x')
    plt.show()

if __name__ == '__main__':
    logger.info('Beginning execution...')
    run_dnn_regression()
    logger.info('Completed execution')