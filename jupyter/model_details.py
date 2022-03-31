import os
import json
from glob import glob
from collections import namedtuple

import torch
from bbb.models.dnn import ClassificationDNN
from bbb.models.bnn import ClassificationBNN
from bbb.config.parameters import Parameters, PriorParameters

ModelDetails = namedtuple('ModelDetails', 'dir mclass description')

MODEL_DETAILS_DICT = {
    # BNN - Singe Gaussian Prior (300 epochs)
    "bnn_sgp_1200": ModelDetails("../saved_models/BBB_classification/baseline/single_gaussian_prior/300_epochs/1200/2022-03-15-09.18.07", ClassificationBNN, "BNN - 1200 hidden units, SGP, 300 epochs"),
    "bnn_sgp_800": ModelDetails("../saved_models/BBB_classification/baseline/single_gaussian_prior/300_epochs/800/2022-03-15-14.25.46", ClassificationBNN, "BNN - 800 hidden units, SGP, 300 epochs"),
    "bnn_sgp_400": ModelDetails("../saved_models/BBB_classification/baseline/single_gaussian_prior/300_epochs/400/2022-03-15-14.26.34", ClassificationBNN, 
    "BNN - 400 hidden units, SGP, 300 epochs"),

    # BNN - Singe Gaussian Prior (600 epochs)
    "bnn_sgp_600_1200": ModelDetails("../saved_models/BBB_classification/baseline/single_gaussian_prior/600_epochs/1200/2022-03-27-20.30.22", ClassificationBNN, "BNN - 1200 hidden units, SGP, 600 epochs"),
    "bnn_sgp_600_800": ModelDetails("../saved_models/BBB_classification/baseline/single_gaussian_prior/600_epochs/800/2022-03-29-17.41.40", ClassificationBNN, "BNN - 800 hidden units, SGP, 600 epochs"),
    "bnn_sgp_600_400": ModelDetails("../saved_models/BBB_classification/baseline/single_gaussian_prior/600_epochs/400/2022-03-29-17.35.15", ClassificationBNN, "BNN - 400 hidden units, SGP, 600 epochs"),

    # BNN - Singe Gaussian Prior (600 epochs), eval every step
    "bnn_sgp_600_1200_every": ModelDetails("../saved_models/BBB_classification/baseline/mog_prior/sigma_1_1_sigma_2_exp_7/1200/600_epochs/2022-03-29-11.30.06", ClassificationBNN, "BNN - 1200 hidden units, SGP, 600 epochs"),

    # BNN - MoG (600 epochs)
    # "bnn_mog_600_1200": ModelDetails("../saved_models/BBB_classification/baseline/mog_prior/sigma_1_1_sigma_2_exp_7/1200/600_epochs/2022-03-27-16.24.44", ClassificationBNN, "BNN - 1200 hidden units, MoG, 600 epochs"), # only eval per 20 steps
    "bnn_mog_600_1200": ModelDetails("../saved_models/BBB_classification/baseline/mog_prior/sigma_1_1_sigma_2_exp_7/1200/600_epochs/2022-03-29-11.30.06", ClassificationBNN, "BNN - 1200 hidden units, MoG, 600 epochs"),
    "bnn_mog_600_800": ModelDetails("../saved_models/BBB_classification/baseline/mog_prior/sigma_1_1_sigma_2_exp_7/800/600_epochs/2022-03-29-17.25.29", ClassificationBNN, "BNN - 800 hidden units, MoG, 600 epochs"),
    "bnn_mog_600_400": ModelDetails("../saved_models/BBB_classification/baseline/mog_prior/sigma_1_1_sigma_2_exp_7/400/600_epochs/2022-03-29-17.41.59", ClassificationBNN, "BNN - 400 hidden units, MoG, 600 epochs"),
    
    # BNN - MoG (300 epochs)
    "bnn_mog_1200": ModelDetails("../saved_models/BBB_classification/baseline/mog_prior/sigma_1_1_sigma_2_exp_7/1200/300_epochs/2022-03-27-07.54.00", ClassificationBNN, "BNN - 1200 hidden units, MoG, 300 epochs"),
    "bnn_mog_800": ModelDetails("../saved_models/BBB_classification/baseline/mog_prior/sigma_1_1_sigma_2_exp_7/800/300_epochs/2022-03-27-10.10.21", ClassificationBNN, "BNN - 800 hidden units, MoG, 300 epochs"),
    "bnn_mog_400": ModelDetails("../saved_models/BBB_classification/baseline/mog_prior/sigma_1_1_sigma_2_exp_7/400/300_epochs/2022-03-27-07.55.07", ClassificationBNN, 
    "BNN - 400 hidden units, MoG, 300 epochs"),

    # BNN - Laplace (600 epochs)
    "bnn_lap_600_1200":  ModelDetails("../saved_models/BBB_classification/baseline/laplace_prior/b_0.15/1200/600_epochs/2022-03-28-21.50.25", ClassificationBNN, "BNN - 1200 hidden units, Laplace, 600 epochs"),

    # DNN - no dropout (300 epochs)
    "dnn_1200": ModelDetails("../saved_models/DNN_classification/baseline/1200/2022-03-15-14.28.25", ClassificationDNN, "DNN - 1200 hidden units, 300 epochs"),
    "dnn_800": ModelDetails("../saved_models/DNN_classification/baseline/800/2022-03-15-16.06.09", ClassificationDNN, "DNN - 800 hidden units, 300 epochs"),
    "dnn_400": ModelDetails("../saved_models/DNN_classification/baseline/400/2022-03-15-16.10.34", ClassificationDNN, "DNN - 400 hidden units, 300 epochs"),

    # DNN - no dropout (600 epochs)
    "dnn_600_1200": ModelDetails("../saved_models/DNN_classification/baseline/1200/600_epochs/2022-03-29-11.30.46", ClassificationDNN, "DNN - 1200 hidden units, 600 epochs"),
    
    # DNN - dropout (300 epochs)
    "dnn_do_400": ModelDetails("../saved_models/DNN_classification/dropout/0.5/1200/2022-03-15-15.21.46", ClassificationDNN, "Dropout - 400 hidden units, 0.5 dropout, 300 epochs"),
    "dnn_do_800": ModelDetails("../saved_models/DNN_classification/dropout/0.5/800/2022-03-15-15.58.04", ClassificationDNN, "Dropout - 800 hidden units, 0.5 dropout, 300 epochs"),
    "dnn_do_1200": ModelDetails("../saved_models/DNN_classification/dropout/0.5/400/2022-03-15-16.26.18", ClassificationDNN, "Dropout - 1200 hidden units, 0.5 dropout, 300 epochs"),
    
    # DNN - dropout (600 epochs)
    "dnn_do_600_1200": ModelDetails("../saved_models/DNN_classification/dropout/0.5/1200/600_epochs/2022-03-29-11.30.56", ClassificationDNN, "Dropout - 1200 hidden units, 0.5 dropout, 600 epochs"),

}

def load_model(MODEL, MODEL_DETAILS=None):
    if MODEL_DETAILS == None:
        MODEL_DETAILS = MODEL_DETAILS_DICT[MODEL]
    
    with open(os.path.join(MODEL_DETAILS.dir, 'params.txt'), 'r') as f:
        params_dict = json.load(f)

    # Need to deserialise the prior_params into a PriorParameters object
    if params_dict['prior_params']:
        params_dict['prior_params'] = PriorParameters(**params_dict['prior_params'])

    params = Parameters(**params_dict)
    
    # Load model
    # net = MODEL_DETAILS.mclass(params=params, eval_mode=True).to(DEVICE)
    net = MODEL_DETAILS.mclass(params=params, eval_mode=True)
    net.model.load_state_dict(torch.load(os.path.join(MODEL_DETAILS.dir, 'model.pt'), map_location=torch.device('cpu')))

    return net, params