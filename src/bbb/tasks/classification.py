import logging

from bbb.utils.pytorch_setup import DEVICE
from bbb.utils.tqdm import train_with_tqdm
from bbb.config.constants import KL_REWEIGHTING_TYPES
from bbb.config.parameters import Parameters, PriorParameters
from bbb.models.bnn import ClassificationBNN
from bbb.models.cnn import CNN
from bbb.data import load_mnist


logger = logging.getLogger(__name__)


BBB_CLASSIFY_PARAMETERS = Parameters(
    name = "BBB_classification",
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
    kl_reweighting_type=KL_REWEIGHTING_TYPES.simple,
)

def run_bbb_mnist_classification_training():
    logger.info('Beginning classification training...')
    net = ClassificationBNN(params=BBB_CLASSIFY_PARAMETERS).to(DEVICE)

    X_train = load_mnist(train=True, batch_size=BBB_CLASSIFY_PARAMETERS.batch_size, shuffle=True)
    X_val = load_mnist(train=False, batch_size=BBB_CLASSIFY_PARAMETERS.batch_size, shuffle=True)

    train_with_tqdm(net=net, train_data=X_train, epochs=BBB_CLASSIFY_PARAMETERS.epochs, eval_data=X_val)

    logger.info('Completed classification training...')


CNN_CLASSIFY_PARAMETERS = Parameters(
    name = "CNN_classification",
    input_dim = 28*28,
    output_dim = 10,
    hidden_units = 1200,
    batch_size = 128,
    lr = 0.01,
    epochs = 10,
)

def run_cnn_mnist_classification_training():
    logger.info('Beginning classification training...')
    net = CNN(params=CNN_CLASSIFY_PARAMETERS).to(DEVICE)

    X_train = load_mnist(train=True, batch_size=CNN_CLASSIFY_PARAMETERS.batch_size, shuffle=True)
    X_val = load_mnist(train=False, batch_size=CNN_CLASSIFY_PARAMETERS.batch_size, shuffle=True)

    train_with_tqdm(net=net, train_data=X_train, epochs=CNN_CLASSIFY_PARAMETERS.epochs, eval_data=X_val)

    accuracy = net.eval(X_val)
    logger.info(f'Accuracy: {accuracy}')
