import logging
import logging

import torch
from tqdm import tqdm 

from bbb.parameters import Parameters, PriorParameters
from bbb.models import BNN
from bbb.data import load_mnist


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CLASSIFY_PARAMETERS = Parameters(
    name = "test",
    input_dim = 28*28,
    output_dim = 10,
    hidden_units = 1200,
    weight_mu = [-0.2, 0.2],
    weight_rho = [-5, -4],
    prior_params = PriorParameters(
        w_sigma=1,
        b_sigma=2,
    ),
    batch_size = 128,
    lr = 1e-4,
    epochs = 300,
    elbo_samples = 2,
    inference_samples = 10,
)

def run_mnist_classification():
    logger.info('Beginning classification training...')
    net = BNN(params=CLASSIFY_PARAMETERS)

    X_train = load_mnist(train=True, batch_size=CLASSIFY_PARAMETERS.batch_size, shuffle=True)
    X_val = load_mnist(train=False, batch_size=CLASSIFY_PARAMETERS.batch_size, shuffle=True)

    epochs = CLASSIFY_PARAMETERS.epochs
    for epoch in tqdm(range(epochs)):
        net.train(X_train)
        
        # If you want to check the parameter values, switch log level to debug
        logger.debug(net.optimizer.param_groups)
        
        net.optimizer.step()
        net.scheduler.step()
        net.eval(X_val)

        logger.info(f'[Epoch {epoch}/{epochs}] - acc: {net.acc}')
        if net.best_acc  and net.acc > net.best_acc:
            net.best_acc = net.acc
            torch.save(net.model.state_dict(), net.save_model_path)

    logger.info('Completed classification training...')

if __name__ == '__main__':
    logger.info('Beginning execution...')
    run_mnist_classification()
    logger.info('Completed execution')
