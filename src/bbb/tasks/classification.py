import logging

from bbb.utils.pytorch_setup import DEVICE
from bbb.utils.tqdm import train_with_tqdm
from bbb.utils.plotting import plot_weight_samples
from bbb.config.constants import KL_REWEIGHTING_TYPES, PRIOR_TYPES, VP_VARIANCE_TYPES
from bbb.config.parameters import Parameters, PriorParameters
from bbb.models.bnn import ClassificationBNN
from bbb.models.dnn import ClassificationDNN
from bbb.data import load_mnist

logger = logging.getLogger(__name__)


#############
# BBB Methods
#############

BBB_CLASSIFY_PARAMETERS = Parameters(
    name = "BBB_classification",
    batch_size = 128,
    epochs = 300,
    # Architecture
    input_dim = 28*28,
    output_dim = 10,
    hidden_units = 1200,
    hidden_layers = 3,
    # Paper choices
    prior_type = PRIOR_TYPES.single,
    kl_reweighting_type = KL_REWEIGHTING_TYPES.paper,
    vp_variance_type = VP_VARIANCE_TYPES.paper,
    local_reparam_trick=False,
    # Variational Inference
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
    elbo_samples = 2,
    inference_samples = 10,
    # Optimiser
    opt_choice = 'Adam',
    lr = 1e-4,
    # LR Scheduler
    step_size = 75,
)

def run_bbb_mnist_classification_training():
    logger.info('Beginning classification training...')
    net = ClassificationBNN(params=BBB_CLASSIFY_PARAMETERS).to(DEVICE)

    logger.info('Initialized BNN...')

    X_train = load_mnist(train=True, batch_size=BBB_CLASSIFY_PARAMETERS.batch_size, shuffle=True)
    X_val = load_mnist(train=False, batch_size=BBB_CLASSIFY_PARAMETERS.batch_size, shuffle=True)
    
    logger.info('Loaded MNIST...')
    
    train_with_tqdm(net=net, train_data=X_train, epochs=BBB_CLASSIFY_PARAMETERS.epochs, eval_data=X_val)

    logger.info('Completed classification training...')

    accuracy = net.evaluate(X_val)
    logger.info(f'Accuracy: {accuracy}')

    weight_samples = net.weight_samples()
    plot_weight_samples(weight_samples, save_dir=net.model_save_dir)

def run_bbb_mnist_classification_evaluation(model_path: str):
    logger.info(f'Beginning classification evaluation against {model_path}...')
    
    net = ClassificationBNN(params=BBB_CLASSIFY_PARAMETERS, eval_mode=True).to(DEVICE)
    net.load_saved(model_path=model_path)

    weight_samples = net.weight_samples()
    plot_weight_samples(weight_samples, save_dir=net.model_save_dir)


#############
# DNN Methods
#############

DNN_CLASSIFY_PARAMETERS = Parameters(
    name = "DNN_classification",
    batch_size = 128,
    epochs = 300,
    # Architecture
    input_dim = 28*28,
    output_dim = 10,
    hidden_units = 1200,
    hidden_layers = 3,
    # Optimiser
    opt_choice = 'Adam',
    lr = 1e-4,
    # LR Scheduler
    step_size = 75,
    gamma = 0.1,
    # Dropout
    dropout = True,
    dropout_p = 0.5,
)

def run_dnn_mnist_classification_training():
    logger.info('Beginning classification training...')
    net = ClassificationDNN(params=DNN_CLASSIFY_PARAMETERS).to(DEVICE)

    logger.info('Initialized DNN...')

    X_train = load_mnist(train=True, batch_size=DNN_CLASSIFY_PARAMETERS.batch_size, shuffle=True)
    X_val = load_mnist(train=False, batch_size=DNN_CLASSIFY_PARAMETERS.batch_size, shuffle=True)

    logger.info('Loaded MNIST...')

    train_with_tqdm(net=net, train_data=X_train, epochs=DNN_CLASSIFY_PARAMETERS.epochs, eval_data=X_val)

    logger.info('Completed classification training...')

    accuracy = net.evaluate(X_val)
    logger.info(f'Accuracy: {accuracy}')

    weight_samples = net.weight_samples()
    plot_weight_samples(weight_samples, save_dir=net.model_save_dir)

def run_dnn_mnist_classification_evaluation(model_path: str):
    logger.info(f'Beginning classification evaluation against {model_path}...')
    
    net = ClassificationDNN(params=DNN_CLASSIFY_PARAMETERS, eval_mode=True).to(DEVICE)
    net.load_saved(model_path=model_path)

    weight_samples = net.weight_samples()
    plot_weight_samples(weight_samples, save_dir=net.model_save_dir)
